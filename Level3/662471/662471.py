"""
OBELIX Level-3  Dyna-Q Training
=================================
Algorithm : Dyna-Q  (Sutton & Barto Ch.8)
Level 3 adaptations:
  1. Pursuit momentum  – when box visible, keep moving toward it
                         even if sonar drops for 1-2 steps (box moved)
  2. No blind station-keep – for moving box, keep MOVING in last direction
     (not oscillate) – intercepts moving box better
  3. No IR probe        – Level 3 box moves, probing loses it instantly
  4. Staged training    – diff=0→diff=2→diff=3 with curriculum
  5. n_planning=50      – 50 imaginary updates per real step
  
State (10-bit):
  fwd_far, fwd_near, left_any, right_any,
  ir, stuck, recent_saw, was_close, is_blind,
  moving_blind  ← NEW: was-visible-and-now-blind-AND-was-recently-moving

Runtime: ~30 min CPU
"""

import sys, os, random, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from collections import defaultdict, deque
from obelix import OBELIX

# ── Hyperparameters ────────────────────────────────────────────────────────────
ACTIONS     = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS   = 5

ALPHA       = 0.10
GAMMA       = 0.999
Q_INIT      = 5.0
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.9985
BLINK_MEM   = 30          # longer memory for moving box (blink cycle up to 60)
N_PLANNING  = 50          # 50 imaginary steps per real step

# Escape: L45×4 = 180° reversal, then FW×2 — works for wall corners
_ESCAPE_SEQ  = [0, 0, 0, 0, 2, 2]

# Reward shaping potential weights
_PHI = np.array([1,2,1,2, 3,6,3,6,3,6,3,6, 1,2,1,2, 10,0], dtype=np.float64)


def potential(obs):
    return float(np.dot(obs, _PHI))

def get_alpha(n):
    return ALPHA / (1.0 + 0.5 * n)    # decaying alpha per visit

def make_state(obs, recent_saw, was_close, is_blind, moving_blind):
    """10-bit state — adds moving_blind for Level 3 box tracking"""
    return (
        int(np.any(obs[[5, 7, 9, 11]])),   # fwd_far
        int(np.any(obs[[4, 6, 8, 10]])),   # fwd_near
        int(np.any(obs[0:4])),             # left_any
        int(np.any(obs[12:16])),           # right_any
        int(obs[16]),                      # ir
        int(obs[17]),                      # stuck
        int(recent_saw),
        int(was_close),
        int(is_blind),
        int(moving_blind),                 # NEW: blind after recent contact
    )

def epsilon_greedy(q_vals, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))

def escape_action(step):
    return _ESCAPE_SEQ[step % len(_ESCAPE_SEQ)]


def train(
    eps_stage1=3000,
    eps_stage2=3000,
    eps_stage3=5000,    # more episodes for hardest level
    max_steps=1000,
    n_planning=N_PLANNING,
    seed=42,
    save_path="q_table_l3.pkl",
):
    rng = np.random.default_rng(seed)

    Q = {m: defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
         for m in ("finder", "pusher", "escape")}

    # Dyna-Q world model
    Model       = {}
    visited_sa  = []

    # Curriculum: easy → medium → hard
    stages = [
        (eps_stage1, 0, False, EPS_START),  # diff=0 no wall (learn basics)
        (eps_stage2, 0, True,  0.40),        # diff=0 with wall
        (eps_stage3, 3, True,  0.50),        # diff=3 moving+blink+wall ← MAIN
    ]

    all_rwds = []

    for si, (n_eps, difficulty, wall, eps_init) in enumerate(stages):
        epsilon = eps_init
        print(f"\n{'='*65}")
        print(f" STAGE {si+1}/3 | diff={difficulty} | wall={wall} "
              f"| n_eps={n_eps} | n_plan={n_planning}")
        print(f"{'='*65}")

        env = OBELIX(
            scaling_factor=5, arena_size=500,
            max_steps=max_steps, wall_obstacles=wall,
            difficulty=difficulty, seed=seed,
        )

        stage_rwds = []
        win_count  = 0

        for ep in range(n_eps):
            obs         = env.reset(seed=int(rng.integers(0, 100_000)))
            history     = deque(maxlen=BLINK_MEM)
            enable_push = False
            esc_step    = 0
            last_dir    = 2     # 1=L22 2=FW 3=R22
            pursuit_steps = 0   # steps since last sonar contact (pursuit momentum)
            total       = 0.0

            for _ in range(max_steps):
                history.append(obs.copy())

                # Last known direction (sonar only, not IR)
                if   any(obs[0:4]):    last_dir = 1; pursuit_steps = 0
                elif any(obs[4:12]):   last_dir = 2; pursuit_steps = 0
                elif any(obs[12:16]):  last_dir = 3; pursuit_steps = 0
                else:
                    pursuit_steps += 1

                any_sens    = bool(np.any(obs[:17]))
                recent_saw  = any(bool(np.any(h[:17])) for h in history)
                was_close   = any(bool(h[16] == 1)     for h in history)
                is_blind    = (not any_sens) and recent_saw
                # moving_blind: box recently visible, now gone, within 15 steps
                # Robot should keep moving (box is nearby, just blinked)
                moving_blind = int(is_blind and pursuit_steps <= 15)
                stuck       = bool(obs[17])

                state  = make_state(obs, recent_saw, was_close, is_blind, moving_blind)
                module = "escape" if stuck else ("pusher" if enable_push else "finder")

                # ── Action selection ──────────────────────────────────────────
                if stuck:
                    action_idx = escape_action(esc_step)
                    esc_step  += 1
                    rule_based = True

                elif is_blind and was_close and not enable_push:
                    # Level 3: KEEP MOVING in last direction (box is moving too!)
                    # Do NOT oscillate — just keep pursuing
                    action_idx = last_dir
                    rule_based = True
                    esc_step   = 0

                else:
                    esc_step   = 0
                    action_idx = epsilon_greedy(Q[module][state], epsilon, rng)
                    rule_based = False

                action             = ACTIONS[action_idx]
                next_obs, rew, done = env.step(action, render=False)
                total             += rew

                if not enable_push and rew > 50:
                    enable_push = True

                if done and total > 500:
                    win_count += 1

                # Next state
                if   any(next_obs[0:4]):    n_last_dir = 1
                elif any(next_obs[4:12]):   n_last_dir = 2
                elif any(next_obs[12:16]):  n_last_dir = 3
                else:                       n_last_dir = last_dir

                n_ps     = pursuit_steps + 1 if not any(next_obs[:16]) else 0
                n_any    = bool(np.any(next_obs[:17]))
                n_recent = any(bool(np.any(h[:17])) for h in history) or n_any
                n_close  = any(bool(h[16] == 1) for h in history) or bool(next_obs[16])
                n_blind  = (not n_any) and n_recent
                n_mb     = int(n_blind and n_ps <= 15)
                n_state  = make_state(next_obs, n_recent, n_close, n_blind, n_mb)
                n_module = "escape" if next_obs[17] else ("pusher" if enable_push else "finder")

                # ── Dyna-Q update ─────────────────────────────────────────────
                if not rule_based:
                    shaped    = rew + GAMMA * potential(next_obs) - potential(obs)
                    best_next = float(np.max(Q[n_module][n_state]))
                    td_err    = shaped + GAMMA * best_next - Q[module][state][action_idx]
                    Q[module][state][action_idx] += ALPHA * td_err

                    # Model update
                    key = (module, state, action_idx)
                    if key not in Model:
                        visited_sa.append(key)
                    Model[key] = (shaped, n_module, n_state)

                    # Planning (50 imaginary steps)
                    if visited_sa:
                        for pk in random.choices(visited_sa, k=n_planning):
                            pr, pnm, pns   = Model[pk]
                            pm, ps, pa     = pk
                            p_best         = float(np.max(Q[pnm][pns]))
                            Q[pm][ps][pa] += ALPHA * (pr + GAMMA * p_best - Q[pm][ps][pa])

                obs = next_obs
                if done: break

            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            stage_rwds.append(total)
            all_rwds.append(total)

            if (ep + 1) % 200 == 0:
                mr   = np.mean(stage_rwds[-200:])
                prev = stage_rwds[-400:-200] if len(stage_rwds) > 200 else [mr - 1]
                tr   = "✅" if mr > np.mean(prev) else "⏳"
                print(
                    f"[S{si+1} Ep{ep+1:>5}/{n_eps}] "
                    f"R:{mr:>8.1f} | W:{win_count:>3}/200 | ε:{epsilon:.4f} | "
                    f"Model:{len(Model):>4} | "
                    f"F:{len(Q['finder'])} P:{len(Q['pusher'])} E:{len(Q['escape'])} {tr}"
                )
                win_count = 0

    # Save (same format as all previous agents)
    data = {
        "Q_finder": {str(k): v.tolist() for k, v in Q["finder"].items()},
        "Q_pusher": {str(k): v.tolist() for k, v in Q["pusher"].items()},
        "Q_escape": {str(k): v.tolist() for k, v in Q["escape"].items()},
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✅ Saved: {save_path}")
    print(f"   F:{len(Q['finder'])} P:{len(Q['pusher'])} E:{len(Q['escape'])}")
    print(f"   Model: {len(Model)} transitions | {len(visited_sa)} unique (s,a)")
    return all_rwds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eps_stage1", type=int, default=3000)
    p.add_argument("--eps_stage2", type=int, default=3000)
    p.add_argument("--eps_stage3", type=int, default=5000)
    p.add_argument("--max_steps",  type=int, default=1000)
    p.add_argument("--n_planning", type=int, default=50)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--save_path",  type=str, default="q_table_l3.pkl")
    a = p.parse_args()
    train(**vars(a))