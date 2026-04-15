"""
Level 2 — Q(lambda) with Blink-Context Augmented State
Curriculum: Stage1=difficulty 0 → Stage2=difficulty 2
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "env"))
import pickle
import sys
import os
import numpy as np
from collections import defaultdict, deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

# ── Hyperparameters ───────────────────────────────────────────
LAMBDA        = 0.92
ALPHA_0       = 0.15
ALPHA_C       = 2.0
GAMMA         = 0.999
Q_INIT        = 30.0
TRACE_MIN     = 1e-3
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.998
BLINK_MEMORY  = 20   # steps to remember if box was seen

# ── Potential: near weights > far weights ─────────────────────
_PHI = np.array([
    1, 2,  1, 2,               # left  sonar: far=1, near=2
    2, 4,  2, 4,  2, 4,  2, 4, # fwd   sonar: far=2, near=4
    1, 2,  1, 2,               # right sonar: far=1, near=2
    8, 0,                      # IR=8, stuck=0
], dtype=np.float64)


def potential(obs):
    return float(np.dot(obs, _PHI))

def get_alpha(n):
    return ALPHA_0 / (1.0 + ALPHA_C * n)

def make_state(obs, recent_saw, was_close, is_blind):
    return tuple(obs.astype(int).tolist()) + (int(recent_saw), int(was_close), int(is_blind))

def epsilon_greedy(q_vals, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))

# Escape sequence: turn left twice then forward
_ESCAPE_SEQ = [0, 0, 2, 0, 0, 2]  # L45,L45,FW,L45,L45,FW

def escape_action(step):
    return _ESCAPE_SEQ[step % len(_ESCAPE_SEQ)]

def station_action(step):
    # Oscillate gently: L22 then R22
    return 1 if (step % 2 == 0) else 3


def train(
    eps_stage1=2000,
    eps_stage2=3000,
    max_steps=1000,
    seed=42,
    save_path="q_table_l2.pkl",
):
    rng = np.random.default_rng(seed)

    Q = {m: defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
         for m in ("finder", "pusher", "escape")}
    N = {m: defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))
         for m in ("finder", "pusher", "escape")}

    epsilon    = EPSILON_START
    all_rwds   = []

    stages = [(eps_stage1, 0, False), (eps_stage2, 2, True)]

    for stage_idx, (n_eps, difficulty, wall) in enumerate(stages):
        print(f"\n{'='*55}")
        print(f" STAGE {stage_idx+1} | difficulty={difficulty} | "
              f"wall={wall} | episodes={n_eps}")
        print(f"{'='*55}")

        env = OBELIX(
            scaling_factor=5, arena_size=500,
            max_steps=max_steps, wall_obstacles=wall,
            difficulty=difficulty, seed=seed,
        )

        stage_rwds = []

        for ep in range(n_eps):
            ep_seed    = int(rng.integers(0, 100_000))
            obs        = env.reset(seed=ep_seed)
            history    = deque(maxlen=BLINK_MEMORY)
            enable_push = False
            esc_step   = 0
            blk_step   = 0
            total      = 0.0
            e          = {}

            for _ in range(max_steps):
                history.append(obs.copy())

                # ── Blink context ──────────────────────────────
                any_sens   = bool(np.any(obs[:17]))
                recent_saw = any(bool(np.any(h[:17])) for h in history)
                was_close  = any(bool(h[16] == 1)     for h in history)
                is_blind   = (not any_sens) and recent_saw

                state  = make_state(obs, recent_saw, was_close, is_blind)
                stuck  = bool(obs[17] == 1)
                module = "escape" if stuck else ("pusher" if enable_push else "finder")

                # ── Action selection ───────────────────────────
                if stuck:
                    action_idx = escape_action(esc_step)
                    esc_step  += 1
                    blk_step   = 0
                elif is_blind and was_close and not enable_push:
                    action_idx = station_action(blk_step)
                    blk_step  += 1
                    esc_step   = 0
                else:
                    esc_step   = 0
                    blk_step   = 0
                    q_vals     = Q[module][state]
                    action_idx = epsilon_greedy(q_vals, epsilon, rng)

                action = ACTIONS[action_idx]
                next_obs, reward, done = env.step(action, render=False)
                total += reward

                # Detect attachment (reward jumps ~+100)
                if not enable_push and reward > 50:
                    enable_push = True

                # ── Next state ─────────────────────────────────
                n_any      = bool(np.any(next_obs[:17]))
                n_recent   = any(bool(np.any(h[:17])) for h in history) or n_any
                n_close    = any(bool(h[16] == 1) for h in history) or (next_obs[16] == 1)
                n_blind    = (not n_any) and n_recent
                next_state = make_state(next_obs, n_recent, n_close, n_blind)

                n_push   = enable_push
                n_module = "escape" if next_obs[17]==1 else ("pusher" if n_push else "finder")

                # ── Shaped reward ──────────────────────────────
                shaped = reward + GAMMA * potential(next_obs) - potential(obs)

                # ── Q(λ) update ────────────────────────────────
                best_next = float(np.max(Q[n_module][next_state]))
                delta     = shaped + GAMMA * best_next - Q[module][state][action_idx]

                N[module][state][action_idx] += 1
                key    = (module, state, action_idx)
                e[key] = e.get(key, 0.0) + 1.0   # accumulating traces

                for k in list(e.keys()):
                    mk, sk, ak = k
                    Q[mk][sk][ak] += get_alpha(int(N[mk][sk][ak])) * delta * e[k]
                    e[k] *= GAMMA * LAMBDA
                    if e[k] < TRACE_MIN:
                        del e[k]

                # Cut traces on non-greedy or module switch
                greedy = int(np.argmax(Q[module][state]))
                if action_idx != greedy or n_module != module:
                    e = {}

                obs = next_obs
                if done:
                    break

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            stage_rwds.append(total)
            all_rwds.append(total)

            if (ep + 1) % 100 == 0:
                mean_r = np.mean(stage_rwds[-100:])
                trend  = "✅" if len(stage_rwds) > 100 and mean_r > np.mean(stage_rwds[-200:-100]) else "⏳"
                print(f"[S{stage_idx+1} Ep {ep+1:>4}/{n_eps}] "
                      f"Mean: {mean_r:>8.1f} | ε: {epsilon:.4f} | "
                      f"F/P/E: {len(Q['finder'])}/{len(Q['pusher'])}/{len(Q['escape'])} {trend}")

    # ── Save ──────────────────────────────────────────────────
    data = {
        "Q_finder": {str(k): v.tolist() for k, v in Q["finder"].items()},
        "Q_pusher": {str(k): v.tolist() for k, v in Q["pusher"].items()},
        "Q_escape": {str(k): v.tolist() for k, v in Q["escape"].items()},
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✅ Saved: {save_path}")
    print(f"   F:{len(Q['finder'])} | P:{len(Q['pusher'])} | E:{len(Q['escape'])}")
    return all_rwds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eps_stage1", type=int, default=2000)
    p.add_argument("--eps_stage2", type=int, default=3000)
    p.add_argument("--max_steps",  type=int, default=1000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--save_path",  type=str, default="q_table_l2.pkl")
    args = p.parse_args()
    train(**vars(args))
