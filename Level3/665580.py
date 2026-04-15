"""
train.py — experiments/Level_3/exp06
======================================
Algorithm  : Dyna-Q + Temporal Q(λ) Hybrid
Level      : All (curriculum: 0 → 0+wall → 2+wall → 3+wall)

Design Philosophy (combining best of exp01, exp02, exp04):
═══════════════════════════════════════════════════════════
CLAUDE THOR
FROM EXP04 (best base):
  ✅ enable_push IN STATE (not external)
  ✅ Post-escape cooldown (don't immediately re-enter wall zone)
  ✅ 4-stage curriculum
  ✅ Replacing eligibility traces

FROM EXP02 (best features):
  ✅ Wall/box spread discrimination (sides_any, spread_high)
  ✅ fast_grow (sudden sensor explosion = wall warning)
  ✅ vel_hint (temporal velocity estimation)

FROM EXP01 (best ideas):
  ✅ Dyna-Q planning (30 simulated steps per real step)
  ✅ moving_blind (keep moving in last direction, not oscillate)
  ✅ pursuit momentum counter

NEW FIXES (bugs from all three):
  ✅ FIX1: Boundary-aware stuck
           push+stuck → FW (box near boundary = success!)
           not-push+stuck → escape sequence

  ✅ FIX2: enable_push auto-reset
           if push=True + 15 consecutive stuck → force reset push
           (prevents ghost-push wall loop)

  ✅ FIX3: Blind mode = PURSUIT not oscillation
           keep moving in last_dir during blink
           NOT L22/R22 alternate (that was wrong for moving box)

  ✅ FIX4: FW bias in zero-sensor exploration
           80% FW, 10% L22, 10% R22 (not 50/50 rotate)

  ✅ FIX5: Shorter escape + cooldown
           L45 L45 FW FW → re-evaluate (4 steps, not 6-step loop)
           cooldown_steps after escape = don't trust sensors briefly

State space:
  12 features → max ~600 practical states
  All features proven useful from prior experiments

Folder:
    experiments/Level_3/exp06/
        train.py
        agent.py
        q_table_best.pkl
        q_table_final.pkl
        rewards_log.csv
        videos/

Run:
    cd experiments/Level_3/exp06
    python train.py
    python train.py --eps1 3000 --eps2 3000 --eps3 2000 --eps4 6000
    
Note: Pure NumPy — no GPU needed. Fast on CPU.
      Dyna-Q gives 30x sample efficiency boost.
"""

import sys, os, random, pickle, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "env"))

import numpy as np
from collections import defaultdict, deque
import cv2
from obelix import OBELIX


# ─────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────
ACTIONS     = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS   = 5

ALPHA_0     = 0.15        # base learning rate
ALPHA_C     = 1.0         # decay constant
GAMMA       = 0.999       # high → terminal reward survives 600+ steps
LAMBDA      = 0.97        # high → long horizon credit
Q_INIT      = 2.0         # mild optimism (not 50 — avoids wall preference)
TRACE_MIN   = 1e-3

EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.9988      # slow decay — large state space needs exploration

BLINK_MEM   = 30          # longer — moving box blinks up to 30 steps
VEL_WINDOW  = 6           # frames for velocity estimation

N_PLANNING  = 30          # Dyna-Q: 30 simulated steps per real step

# Escape config
_ESCAPE_SEQ      = [0, 0, 2, 2]   # L45 L45 FW FW — SHORT then re-evaluate
COOLDOWN_STEPS   = 8               # after escape, trust sensors less
PUSH_STUCK_LIMIT = 15              # consecutive push+stuck → reset push

VIDEO_EVERY = 500
VIDEO_FPS   = 15
VIDEO_DIR   = os.path.join(_HERE, "videos")


# ─────────────────────────────────────────────────────────────
# State Construction — Combined Best Features
# ─────────────────────────────────────────────────────────────
def sensor_centroid(obs):
    """Left vs Right balance of forward sensors → box drift direction."""
    lf = int(obs[4]) + int(obs[5]) + int(obs[6])  + int(obs[7])
    rf = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    total = lf + rf
    if total == 0: return None
    if lf > rf + 1: return -1   # box left of center
    if rf > lf + 1: return +1   # box right of center
    return 0                     # centered


def make_state(obs, history, enable_push, moving_blind):
    """
    12-feature temporal abstract state.
    Combines best features from all experiments.

    Features:
      stuck        : wall contact (but see boundary-aware logic)
      ir           : IR sensor (box ~20px ahead)
      fwd_near     : forward near sensors
      fwd_far      : forward far sensors
      sides_any    : any side sensors (wall signature — EXP02)
      spread_high  : wide spread > narrow (wall vs box — EXP02)
      fast_grow    : sudden sensor explosion (wall warning — EXP02)
      is_blind     : currently blind but recently saw something
      was_close    : IR was active in recent history
      vel_hint     : box drift direction -1/0/+1 (EXP02 temporal)
      enable_push  : attached to box? (EXP04 — in state!)
      moving_blind : blind but recently moving (EXP01 pursuit)
    """
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    # Wall vs Box discrimination (EXP02)
    count_fwd  = sum(int(obs[i]) for i in range(4, 12))
    count_side = (sum(int(obs[i]) for i in range(0, 4)) +
                  sum(int(obs[i]) for i in range(12, 16)))
    sides_any   = int(count_side > 0)
    spread_high = int(count_side > 2 and count_fwd >= 2)

    hist_list  = list(history)
    n          = len(hist_list)

    recent_saw = int(any(bool(np.any(h[:17])) for h in hist_list))
    was_close  = int(any(bool(h[16] == 1)     for h in hist_list))
    any_now    = bool(np.any(obs[:17]))
    is_blind   = int(not any_now and recent_saw)

    # Velocity hint — box drift direction (EXP02)
    vel_hint = 0
    if n >= 2:
        centroids = []
        for h in hist_list[-VEL_WINDOW:]:
            c = sensor_centroid(h)
            if c is not None:
                centroids.append(c)
        if len(centroids) >= 3:
            half       = len(centroids) // 2
            avg_first  = np.mean(centroids[:half])
            avg_second = np.mean(centroids[half:])
            drift      = avg_second - avg_first
            if drift > 0.4:   vel_hint = +1
            elif drift < -0.4: vel_hint = -1

    # Sudden sensor explosion = wall warning (EXP02)
    fast_grow = 0
    if n >= 2:
        prev_cnt  = sum(int(x) for x in hist_list[-2][:16])
        curr_cnt  = sum(int(x) for x in obs[:16])
        fast_grow = int((curr_cnt - prev_cnt) >= 4)

    return (
        stuck,              # 2
        ir,                 # 2
        fwd_near,           # 2
        fwd_far,            # 2
        sides_any,          # 2  ← EXP02
        spread_high,        # 2  ← EXP02
        fast_grow,          # 2  ← EXP02
        is_blind,           # 2
        was_close,          # 2
        vel_hint + 1,       # 3  ← EXP02
        int(enable_push),   # 2  ← EXP04
        int(moving_blind),  # 2  ← EXP01
    )
    # Max theoretical: 2^11 × 3 = 6144
    # Practical reachable: ~300-500 states


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_alpha(n):
    return ALPHA_0 / (1.0 + ALPHA_C * n)


def epsilon_greedy(q_vals, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


def save_tables(Q, N_visits, Model, path):
    data = {}
    for m in ("finder", "pusher", "escape"):
        data[f"Q_{m}"] = {str(k): v.tolist() for k, v in Q[m].items()}
    data["Model"] = {str(k): v for k, v in Model.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  💾 {os.path.basename(path)}")


def log_rewards(path, rewards):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i + 1, f"{r:.2f}"])


# ─────────────────────────────────────────────────────────────
# Action Logic — Fixed Behaviors
# ─────────────────────────────────────────────────────────────
def escape_action(esc_step):
    """
    SHORT escape sequence — re-evaluate after 4 steps.
    FIX5: Was 6-step loop that could return to same wall.
    """
    return _ESCAPE_SEQ[esc_step % len(_ESCAPE_SEQ)]


def pursuit_action(last_dir, vel_hint, step):
    """
    FIX3: During blink, KEEP MOVING in last known direction.
    NOT oscillating L22/R22.

    Also uses velocity hint to slightly adjust direction.
    EXP01: pursuit momentum concept.
    """
    # Adjust for box movement
    if vel_hint == +1:
        # Box was drifting right → lean right
        return 3 if step % 4 == 0 else last_dir   # occasional R22
    elif vel_hint == -1:
        # Box was drifting left → lean left
        return 1 if step % 4 == 0 else last_dir   # occasional L22
    return last_dir   # straight pursuit


def explore_action(rng, step):
    """
    FIX4: FW-biased exploration in zero-sensor state.
    80% FW, 10% L22, 10% R22 — not 50/50 rotate.
    Also adds systematic arc every 40 steps.
    """
    if step % 40 == 0:
        return 1   # L22 — systematic arc to cover arena
    probs = np.array([0.0, 0.10, 0.80, 0.10, 0.0])
    return int(rng.choice(N_ACTIONS, p=probs))


# ─────────────────────────────────────────────────────────────
# Video Recording
# ─────────────────────────────────────────────────────────────
def record_episode(env, Q, N_vis, Model, ep_num, difficulty, seed, max_steps):
    """Greedy episode recorded as MP4."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fname  = os.path.join(VIDEO_DIR, f"ep{ep_num:05d}_d{difficulty}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    rng_v  = np.random.default_rng(seed + 77777)

    obs          = env.reset(seed=seed)
    history      = deque(maxlen=BLINK_MEM)
    history.append(obs.copy())
    enable_push  = False
    esc_step     = 0
    cooldown     = 0
    push_stuck   = 0
    last_dir     = 2
    pursuit_step = 0
    total        = 0.0

    for step in range(max_steps):
        env._update_frames(show=False)
        frame = cv2.flip(env.frame.copy(), 0)
        st_str = ("PUSH" if enable_push else
                  "STUCK" if obs[17] else
                  "IR" if obs[16] else "FIND")
        cv2.putText(
            frame, f"Stp:{step:4d} Scr:{total:7.0f} [{st_str}]",
            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1
        )
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        # Update last_dir from current obs
        if   any(obs[0:4]):   last_dir = 1; pursuit_step = 0
        elif any(obs[4:12]):  last_dir = 2; pursuit_step = 0
        elif any(obs[12:16]): last_dir = 3; pursuit_step = 0
        else: pursuit_step += 1

        stuck   = obs[17] == 1
        ir_on   = obs[16] == 1 and not stuck
        any_vis = bool(np.any(obs[:17]))
        recent  = any(bool(np.any(h[:17])) for h in history)
        blind   = not any_vis and recent
        moving_blind = int(blind and pursuit_step <= 20)
        state   = make_state(obs, history, enable_push, moving_blind)
        vel     = state[9] - 1
        cooldown = max(0, cooldown - 1)

        # FIX2: push stuck auto-reset
        if enable_push and stuck:
            push_stuck += 1
            if push_stuck >= PUSH_STUCK_LIMIT:
                enable_push = False
                push_stuck  = 0
        elif not stuck:
            push_stuck = 0

        # FIX1: Boundary-aware stuck
        if stuck and enable_push:
            action_idx = 2   # FW — almost success!
        elif stuck:
            action_idx = escape_action(esc_step)
            esc_step  += 1
        elif ir_on and not enable_push and cooldown == 0:
            action_idx = 2   # FW → attach
        elif enable_push and not stuck:
            action_idx = 2   # FW → push
        elif blind and recent:
            action_idx = pursuit_action(last_dir, vel, step)
        elif any_vis and not stuck:
            module = "pusher" if enable_push else "finder"
            q_vals = Q[module][state]
            action_idx = int(np.argmax(q_vals)) if state in Q[module] \
                else int(rng_v.integers(0, N_ACTIONS))
        else:
            action_idx = explore_action(rng_v, step)

        obs, reward, done = env.step(ACTIONS[action_idx], render=False)
        history.append(obs.copy())
        total += reward
        if not enable_push and reward > 50:
            enable_push = True
            push_stuck  = 0
        if done: break

    if writer: writer.release()
    print(f"  🎬 {os.path.basename(fname)}  score={total:.0f}")
    return total


# ─────────────────────────────────────────────────────────────
# Core Training Loop
# ─────────────────────────────────────────────────────────────
def run_stage(
    env, Q, N_visits, Model, visited_sa,
    n_eps, epsilon, rng,
    stage_name, max_steps,
    best_path, best_mean_ref,
    all_rwd, total_ep_ref,
    video_ep_offset, difficulty,
    log_every=200,
):
    stage_rwd = []
    best_mean = best_mean_ref

    for ep in range(n_eps):
        ep_seed = int(rng.integers(0, 100_000))
        obs     = env.reset(seed=ep_seed)

        history      = deque(maxlen=BLINK_MEM)
        history.append(obs.copy())
        enable_push  = False
        esc_step     = 0
        cooldown     = 0
        push_stuck   = 0
        last_dir     = 2     # default: forward
        pursuit_step = 0     # steps since last sensor contact
        total        = 0.0
        e            = {}    # eligibility traces

        for step in range(max_steps):

            # ── Update last_dir (for pursuit) ─────────────────
            if   any(obs[0:4]):   last_dir = 1; pursuit_step = 0
            elif any(obs[4:12]):  last_dir = 2; pursuit_step = 0
            elif any(obs[12:16]): last_dir = 3; pursuit_step = 0
            else: pursuit_step += 1

            # ── Context variables ─────────────────────────────
            stuck   = obs[17] == 1
            ir_on   = obs[16] == 1 and not stuck
            any_vis = bool(np.any(obs[:17]))
            recent  = any(bool(np.any(h[:17])) for h in history)
            blind   = not any_vis and recent
            moving_blind = int(blind and pursuit_step <= 20)
            cooldown = max(0, cooldown - 1)

            state  = make_state(obs, history, enable_push, moving_blind)
            vel    = state[9] - 1   # vel_hint: -1/0/+1

            # ── FIX2: enable_push auto-reset ──────────────────
            if enable_push and stuck:
                push_stuck += 1
                if push_stuck >= PUSH_STUCK_LIMIT:
                    enable_push = False
                    push_stuck  = 0
                    cooldown    = COOLDOWN_STEPS
                    e           = {}   # reset traces on detach
            elif not stuck:
                push_stuck = 0

            # ── Module selection ──────────────────────────────
            module = ("escape" if stuck and not enable_push
                      else "pusher" if enable_push
                      else "finder")

            # ── Action selection ──────────────────────────────
            rule_based = False

            # FIX1: Boundary-aware stuck
            if stuck and enable_push:
                # push+stuck = box at boundary = keep pushing!
                action_idx = 2    # FW
                rule_based = True

            elif stuck and not enable_push:
                # exploration stuck = wall = escape
                action_idx = escape_action(esc_step)
                esc_step  += 1
                cooldown   = COOLDOWN_STEPS   # don't trust sensors right after escape
                rule_based = True
                e          = {}   # reset traces

            elif ir_on and not enable_push and cooldown == 0:
                # IR + not stuck + no cooldown = BOX confirmed → attach
                action_idx = 2    # FW
                rule_based = True
                esc_step   = 0

            elif enable_push and not stuck:
                # Attached, moving → push toward boundary
                action_idx = 2    # FW
                rule_based = True
                esc_step   = 0

            elif blind and recent:
                # FIX3: Pursuit (not oscillation) during blink
                action_idx = pursuit_action(last_dir, vel, step)
                rule_based = True
                esc_step   = 0

            elif any_vis and not stuck and cooldown == 0:
                # Sensors active, not stuck → Q-table
                esc_step   = 0
                action_idx = epsilon_greedy(Q[module][state], epsilon, rng)

            else:
                # FIX4: FW-biased exploration
                esc_step   = 0
                if rng.random() < epsilon:
                    action_idx = explore_action(rng, step)
                else:
                    action_idx = (int(np.argmax(Q[module][state]))
                                  if state in Q[module]
                                  else explore_action(rng, step))

            action = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action, render=False)
            history.append(next_obs.copy())
            total += reward

            if not enable_push and reward > 50:
                enable_push = True
                push_stuck  = 0
                esc_step    = 0

            # ── Next state ────────────────────────────────────
            n_any     = bool(np.any(next_obs[:17]))
            n_recent  = any(bool(np.any(h[:17])) for h in history) or n_any
            n_blind   = not n_any and n_recent
            n_mb      = int(n_blind and (pursuit_step + 1) <= 20)
            n_state   = make_state(next_obs, history, enable_push, n_mb)
            n_module  = ("pusher" if enable_push else "finder")

            # ── Q(λ) update (only for finder/pusher) ──────────
            if not rule_based and module in ("finder", "pusher"):
                best_next = float(np.max(Q[n_module][n_state]))
                delta     = (reward + GAMMA * best_next
                             - Q[module][state][action_idx])

                N_visits[module][state][action_idx] += 1
                n_vis  = int(N_visits[module][state][action_idx])
                key    = (module, state, action_idx)
                e[key] = 1.0   # REPLACING trace

                for k in list(e.keys()):
                    mk, sk, ak = k
                    Q[mk][sk][ak] += get_alpha(int(N_visits[mk][sk][ak])) * delta * e[k]
                    e[k]          *= GAMMA * LAMBDA
                    if e[k] < TRACE_MIN:
                        del e[k]

                if n_module != module:
                    e = {}   # reset on module switch

                # ── Dyna-Q: model update + planning ───────────
                mkey = (module, state, action_idx)
                if mkey not in Model:
                    visited_sa.append(mkey)
                Model[mkey] = (reward, n_module, n_state)

                if visited_sa and len(visited_sa) > 5:
                    plan_keys = random.choices(visited_sa, k=N_PLANNING)
                    for pk in plan_keys:
                        pr, pnm, pns = Model[pk]
                        pm, ps, pa   = pk
                        p_best       = float(np.max(Q[pnm][pns]))
                        Q[pm][ps][pa] += get_alpha(
                            int(N_visits[pm][ps][pa])
                        ) * (pr + GAMMA * p_best - Q[pm][ps][pa])

            obs = next_obs
            if done: break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        stage_rwd.append(total)
        all_rwd.append(total)
        total_ep_ref[0] += 1

        if (ep + 1) % log_every == 0:
            m_log = np.mean(stage_rwd[-log_every:])
            prev  = (stage_rwd[-(2*log_every):-log_every]
                     if len(stage_rwd) > log_every else [m_log - 1])
            trend = "✅" if m_log > np.mean(prev) else "⏳"
            succ  = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            print(
                f"[{stage_name} Ep {ep+1:>5}/{n_eps}] "
                f"Mean: {m_log:>8.1f} | ε: {epsilon:.4f} | "
                f"Succ/{log_every}: {succ:3d} | "
                f"F/P: {len(Q['finder'])}/{len(Q['pusher'])} | "
                f"Model: {len(Model):>5} {trend}"
            )
            if m_log > best_mean:
                best_mean = m_log
                save_tables(Q, N_visits, Model, best_path)

        if (ep + 1) % VIDEO_EVERY == 0:
            record_episode(
                env, Q, N_visits, Model,
                video_ep_offset + total_ep_ref[0],
                difficulty,
                int(rng.integers(0, 10000)),
                max_steps,
            )

    return epsilon, best_mean


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def train(
    eps1 = 2000,   # Stage 1: diff=0, no wall  — learn basic find+push
    eps2 = 3000,   # Stage 2: diff=0, wall     — wall avoidance
    eps3 = 2000,   # Stage 3: diff=2, wall     — blink handling
    eps4 = 6000,   # Stage 4: diff=3, wall     — moving+blink (MAIN)
    max_steps   = 1000,
    seed        = 42,
    save_prefix = "q_table",
):
    best_path  = os.path.join(_HERE, f"{save_prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{save_prefix}_final.pkl")
    log_path   = os.path.join(_HERE, "rewards_log.csv")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    Q = {m: defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
         for m in ("finder", "pusher", "escape")}
    N_visits = {m: defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))
                for m in ("finder", "pusher", "escape")}
    Model      = {}
    visited_sa = []

    epsilon   = EPS_START
    all_rwd   = []
    best_mean = -np.inf
    total_ep  = [0]

    stages = [
        (eps1, 0, False, "S1 (diff=0,no-wall)"),
        (eps2, 0, True,  "S2 (diff=0,+wall)  "),
        (eps3, 2, True,  "S3 (diff=2,blink)  "),
        (eps4, 3, True,  "S4 (diff=3,move+blink)"),
    ]
    eps_resets = [EPS_START, 0.50, 0.40, 0.50]  # epsilon restart per stage

    prev_offset = 0
    for si, ((n_eps, diff, wall, sname), eps_init) in enumerate(
        zip(stages, eps_resets)
    ):
        epsilon = max(eps_init, epsilon)
        print(f"\n{'═'*65}")
        print(f" {sname} | n_eps={n_eps} | diff={diff} | wall={wall}")
        print(f"{'═'*65}")

        env = OBELIX(
            scaling_factor=5, arena_size=500,
            max_steps=max_steps, wall_obstacles=wall,
            difficulty=diff, box_speed=2, seed=seed,
        )

        epsilon, best_mean = run_stage(
            env, Q, N_visits, Model, visited_sa,
            n_eps, epsilon, rng,
            sname, max_steps,
            best_path, best_mean,
            all_rwd, total_ep,
            prev_offset, diff,
        )
        prev_offset += n_eps

    # ── Final save ────────────────────────────────────────────
    save_tables(Q, N_visits, Model, final_path)
    log_rewards(log_path, all_rwd)

    total_succ = sum(1 for r in all_rwd if r > 500)
    total_eps  = total_ep[0]
    print(f"\n✅ Done!")
    print(f"   Best mean    : {best_mean:.1f}")
    print(f"   Success rate : {total_succ}/{total_eps} "
          f"({100*total_succ/max(total_eps,1):.1f}%)")
    print(f"   Model size   : {len(Model)} transitions")
    print(f"   Log          → rewards_log.csv")
    print(f"   Videos       → {VIDEO_DIR}/")

    # Final greedy videos for all difficulties
    print("\n🎬 Final evaluation videos...")
    for diff_v, wall_v in [(0, False), (0, True), (2, True), (3, True)]:
        env_v = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall_v, difficulty=diff_v, seed=seed,
        )
        for s in range(2):
            record_episode(
                env_v, Q, N_visits, Model,
                99000 + diff_v * 10 + s, diff_v, s * 137, max_steps
            )

    return all_rwd


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="EXP06 — Best Hybrid Agent")
    p.add_argument("--eps1",        type=int, default=2000)
    p.add_argument("--eps2",        type=int, default=3000)
    p.add_argument("--eps3",        type=int, default=2000)
    p.add_argument("--eps4",        type=int, default=6000)
    p.add_argument("--max_steps",   type=int, default=1000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save_prefix", type=str, default="q_table")
    a = p.parse_args()
    train(**vars(a))