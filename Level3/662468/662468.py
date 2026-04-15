"""
train.py — experiments/Level_3/exp01
======================================
Algorithm  : Temporal Q(λ) with Predictive Interception
Level      : 3 (moving + blinking box, wall ON)

Key Ideas:
  1. Temporal State     — last 6 obs → velocity estimation
  2. Sensor Spread      — lateral spread = wall signature
  3. Velocity Hint      — centroid shift = box direction
  4. Station Keeping    — hover during blink
  5. Predictive Intercept — turn toward box's future position
  6. Two-Stage Curriculum — Level 1 basics → Level 3 dynamics
  7. Replacing Traces   — no gradient explosion in loops
  8. Video Recording    — save MP4 every N episodes

Folder:
    experiments/Level_3/exp01/
        train.py
        agent.py
        q_table_best.pkl
        q_table_final.pkl
        rewards_log.csv
        videos/

Run:
    cd experiments/Level_3/exp01
    python train.py
    python train.py --eps1 5000 --eps3 8000 --max_steps 1000
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "env"))

import pickle, csv, random
import numpy as np
from collections import defaultdict, deque
import cv2
from obelix import OBELIX


# ─────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────
ACTIONS        = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS      = 5

LAMBDA         = 0.97     # high → long horizon credit assignment
ALPHA_0        = 0.15
ALPHA_C        = 1.5
GAMMA          = 0.999    # 0.999^600 = 0.55 → reward visible
Q_INIT         = 2.0      # mild optimism — avoids wall trap
TRACE_MIN      = 1e-3

EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9990   # slow decay for temporal state space

BLINK_MEMORY   = 30       # covers max blink off duration (30 steps)
VEL_WINDOW     = 6        # frames for velocity estimation

VIDEO_EVERY    = 500      # save video every N episodes
VIDEO_FPS      = 15
VIDEO_DIR      = os.path.join(_HERE, "videos")


# ─────────────────────────────────────────────────────────────
# Temporal State Construction
# ─────────────────────────────────────────────────────────────
def sensor_centroid(obs):
    """
    Left vs Right balance of forward sensors.
    Returns: -1=box_left, 0=center, +1=box_right, None=no_signal
    """
    # Left half of forward arc: indices 4,5,6,7
    # Right half of forward arc: indices 8,9,10,11
    left_fwd  = int(obs[4]) + int(obs[5]) + int(obs[6])  + int(obs[7])
    right_fwd = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    total     = left_fwd + right_fwd
    if total == 0:
        return None
    if left_fwd > right_fwd + 1:
        return -1   # box on left side of forward cone
    if right_fwd > left_fwd + 1:
        return +1   # box on right side of forward cone
    return 0        # centered


def make_state(obs, history):
    """
    Temporal abstract state for Q-table.

    obs     : current 18-bit numpy array
    history : deque of recent observations (BLINK_MEMORY length)

    Returns : tuple — hashable Q-table key
    Max theoretical states ~1500, practical ~150-200
    """
    # ── Current snapshot features ─────────────────────────────
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    # ── Lateral spread — Wall vs Box discriminator ────────────
    # Box:  narrow object → mostly FORWARD sensors fire
    # Wall: wide object  → SIDE sensors fire too
    count_fwd  = sum(int(obs[i]) for i in range(4, 12))
    count_side = (sum(int(obs[i]) for i in range(0, 4)) +
                  sum(int(obs[i]) for i in range(12, 16)))
    spread_high = int(count_side > 2 and count_fwd >= 2)
    sides_any   = int(count_side > 0)

    # ── Temporal / memory features ────────────────────────────
    hist_list  = list(history)
    n          = len(hist_list)

    recent_saw = int(any(bool(np.any(h[:17])) for h in hist_list))
    was_close  = int(any(bool(h[16] == 1)     for h in hist_list))
    any_now    = bool(np.any(obs[:17]))
    is_blind   = int(not any_now and recent_saw)

    # ── Velocity hint (KEY feature for Level 3) ───────────────
    # Track centroid of box in sensor field over VEL_WINDOW steps
    # Drifting centroid = box moving relative to robot
    vel_hint = 0
    if n >= 2:
        centroids = []
        window    = hist_list[-VEL_WINDOW:]
        for h in window:
            c = sensor_centroid(h)
            if c is not None:
                centroids.append(c)
        if len(centroids) >= 3:
            half      = len(centroids) // 2
            avg_first  = np.mean(centroids[:half])
            avg_second = np.mean(centroids[half:])
            drift      = avg_second - avg_first
            if drift > 0.4:
                vel_hint = +1   # drifting right → intercept right
            elif drift < -0.4:
                vel_hint = -1   # drifting left  → intercept left

    # ── Sudden sensor explosion (wall hit warning) ────────────
    fast_grow = 0
    if n >= 2:
        prev_count = sum(int(x) for x in hist_list[-2][:16])
        curr_count = sum(int(x) for x in obs[:16])
        fast_grow  = int((curr_count - prev_count) >= 4)

    return (
        stuck,           # 2 values
        ir,              # 2
        fwd_near,        # 2
        fwd_far,         # 2
        sides_any,       # 2  ← wall signature
        spread_high,     # 2  ← wide spread = wall
        is_blind,        # 2  ← blink detection
        was_close,       # 2  ← near-box memory
        vel_hint + 1,    # 3  (0=left, 1=center, 2=right drift)
        fast_grow,       # 2  ← sudden explosion = wall warning
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_alpha(n):
    return ALPHA_0 / (1.0 + ALPHA_C * n)


def epsilon_greedy(q_vals, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


def save_tables(Q, N, path):
    data = {}
    for m in ("finder", "pusher", "escape"):
        data[f"Q_{m}"] = {str(k): v.tolist() for k, v in Q[m].items()}
        data[f"N_{m}"] = {str(k): n.tolist() for k, n in N[m].items()}
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
# Deterministic Behaviors (NOT learned — hardcoded reflexes)
# ─────────────────────────────────────────────────────────────
_ESCAPE_SEQ = [0, 0, 2, 4, 4, 2]   # L45 L45 FW R45 R45 FW

def escape_action(esc_step):
    """Deterministic wall escape."""
    return _ESCAPE_SEQ[esc_step % len(_ESCAPE_SEQ)]


def station_action(step):
    """Hover in place during blink — wait for box to reappear."""
    return 1 if step % 2 == 0 else 3   # L22 / R22 alternate


def intercept_action(vel_hint, step):
    """
    Predictive interception for moving box.
    vel_hint : -1=moving_left, 0=stationary, +1=moving_right

    Instead of chasing current position, aim ahead.
    Pattern: 1 turn + 2 forward = curved intercept path
    """
    if vel_hint == +1:
        # Box drifting right → turn right + go forward
        return 3 if step % 3 == 0 else 2   # R22, FW, FW
    elif vel_hint == -1:
        # Box drifting left → turn left + go forward
        return 1 if step % 3 == 0 else 2   # L22, FW, FW
    else:
        return 2   # centered → straight forward


# ─────────────────────────────────────────────────────────────
# Video Recording
# ─────────────────────────────────────────────────────────────
def record_episode(env, Q, ep_num, difficulty, seed, max_steps):
    """
    Run one greedy episode and save as MP4.
    Robot behavior visible — use to debug policy.
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fname  = os.path.join(VIDEO_DIR, f"ep{ep_num:05d}_diff{difficulty}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    obs          = env.reset(seed=seed)
    history      = deque(maxlen=BLINK_MEMORY)
    history.append(obs.copy())
    enable_push  = False
    esc_step     = blink_step = 0
    total        = 0.0
    rng_v        = np.random.default_rng(seed + 77777)

    for step in range(max_steps):
        # Render
        env._update_frames(show=False)
        frame = cv2.flip(env.frame.copy(), 0)

        # Overlay text
        state_str = "PUSH" if enable_push else (
                    "STUCK" if obs[17] else (
                    "ATTACH" if obs[16] else "FIND"))
        cv2.putText(frame, f"Step:{step:4d}  Score:{total:7.0f}  [{state_str}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        # Greedy policy
        stuck   = obs[17] == 1
        ir_on   = obs[16] == 1 and not stuck
        any_vis = bool(np.any(obs[:17]))
        recent  = any(bool(np.any(h[:17])) for h in history)
        blind   = not any_vis and recent
        state   = make_state(obs, history)
        vel     = state[8] - 1
        module  = "escape" if stuck else ("pusher" if enable_push else "finder")

        if stuck:
            action_idx = escape_action(esc_step); esc_step += 1
        elif ir_on and not enable_push:
            action_idx = 2
        elif enable_push and not stuck:
            action_idx = 2
        elif blind and recent:
            action_idx = station_action(blink_step); blink_step += 1
        elif any_vis and vel != 0:
            action_idx = intercept_action(vel, step)
        else:
            q_vals = Q[module][state]
            action_idx = (int(np.argmax(q_vals)) if state in Q[module]
                          else int(rng_v.integers(0, N_ACTIONS)))

        obs, reward, done = env.step(action, render=False) \
            if False else env.step(ACTIONS[action_idx], render=False)
        history.append(obs.copy())
        total += reward
        if not enable_push and reward > 50:
            enable_push = True
        if done:
            break

    if writer:
        writer.release()
    print(f"  🎬 {os.path.basename(fname)}  score={total:.0f}")


# ─────────────────────────────────────────────────────────────
# Core Training Loop (reused for Stage 1 and Stage 2)
# ─────────────────────────────────────────────────────────────
def run_stage(env, Q, N, n_eps, epsilon, rng,
              stage_name, max_steps, best_path,
              best_mean_ref, all_rwd, total_ep_ref,
              video_ep_offset, difficulty, log_every=100):
    """
    Generic training loop for one curriculum stage.
    Modifies Q, N, all_rwd in-place.
    Returns updated epsilon, best_mean.
    """
    stage_rwd = []
    best_mean = best_mean_ref

    for ep in range(n_eps):
        ep_seed      = int(rng.integers(0, 100_000))
        obs          = env.reset(seed=ep_seed)
        history      = deque(maxlen=BLINK_MEMORY)
        history.append(obs.copy())
        enable_push  = False
        esc_step     = 0
        blink_step   = 0
        total        = 0.0
        e            = {}

        for step in range(max_steps):
            stuck   = obs[17] == 1
            ir_on   = obs[16] == 1 and not stuck
            any_vis = bool(np.any(obs[:17]))
            recent  = any(bool(np.any(h[:17])) for h in history)
            blind   = not any_vis and recent
            state   = make_state(obs, history)
            vel     = state[8] - 1
            module  = "escape" if stuck else (
                      "pusher" if enable_push else "finder")

            # ── Action selection ──────────────────────────────
            if stuck:
                action_idx = escape_action(esc_step)
                esc_step  += 1
                blink_step = 0
                e          = {}   # reset traces on escape

            elif ir_on and not enable_push:
                # IR + not stuck = BOX confirmed → attach
                action_idx   = 2   # FW
                esc_step = blink_step = 0

            elif enable_push and not stuck:
                # Attached → push to boundary (always FW)
                action_idx   = 2
                esc_step = blink_step = 0

            elif blind and recent:
                # Box blinked → station keep (don't wander)
                action_idx = station_action(blink_step)
                blink_step += 1
                esc_step   = 0

            elif any_vis and vel != 0 and not enable_push:
                # Box moving → INTERCEPT
                action_idx   = intercept_action(vel, step)
                esc_step = blink_step = 0

            else:
                esc_step = blink_step = 0
                q_vals     = Q[module][state]
                action_idx = epsilon_greedy(q_vals, epsilon, rng)

            action = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action, render=False)
            history.append(next_obs.copy())
            total += reward

            if not enable_push and reward > 50:
                enable_push = True

            # ── Q(λ) update — only for finder/pusher ──────────
            if module in ("finder", "pusher") and not stuck:
                n_module  = "pusher" if enable_push else "finder"
                n_state   = make_state(next_obs, history)
                best_next = float(np.max(Q[n_module][n_state]))
                delta     = (reward + GAMMA * best_next
                             - Q[module][state][action_idx])

                N[module][state][action_idx] += 1
                key    = (module, state, action_idx)
                e[key] = 1.0   # REPLACING trace (not accumulating)

                for k in list(e.keys()):
                    mk, sk, ak = k
                    Q[mk][sk][ak] += (
                        get_alpha(int(N[mk][sk][ak])) * delta * e[k]
                    )
                    e[k] *= GAMMA * LAMBDA
                    if e[k] < TRACE_MIN:
                        del e[k]

                if n_module != module:
                    e = {}

            obs = next_obs
            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        stage_rwd.append(total)
        all_rwd.append(total)
        total_ep_ref[0] += 1

        # ── Logging ───────────────────────────────────────────
        if (ep + 1) % log_every == 0:
            m100  = np.mean(stage_rwd[-log_every:])
            prev  = (stage_rwd[-(2*log_every):-log_every]
                     if len(stage_rwd) > log_every else [m100-1])
            trend = "✅" if m100 > np.mean(prev) else "⏳"
            succ  = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            print(
                f"[{stage_name} Ep {ep+1:>5}/{n_eps}] "
                f"Mean: {m100:>8.1f} | ε: {epsilon:.4f} | "
                f"Succ/{log_every}: {succ:3d} | "
                f"F/P/E: "
                f"{len(Q['finder'])}/{len(Q['pusher'])}/{len(Q['escape'])} "
                f"{trend}"
            )
            if m100 > best_mean:
                best_mean = m100
                save_tables(Q, N, best_path)

        # ── Video recording ───────────────────────────────────
        if (ep + 1) % VIDEO_EVERY == 0:
            record_episode(
                env, Q,
                video_ep_offset + total_ep_ref[0],
                difficulty,
                int(rng.integers(0, 10000)),
                max_steps,
            )

    return epsilon, best_mean


# ─────────────────────────────────────────────────────────────
# Main Train Function
# ─────────────────────────────────────────────────────────────
def train(
    eps1        = 5000,
    eps3        = 6000,
    max_steps   = 1000,
    seed        = 42,
    save_prefix = "q_table",
):
    best_path  = os.path.join(_HERE, f"{save_prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{save_prefix}_final.pkl")
    log_path   = os.path.join(_HERE, "rewards_log.csv")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    Q = {m: defaultdict(
            lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
         for m in ("finder", "pusher", "escape")}
    N = {m: defaultdict(
            lambda: np.zeros(N_ACTIONS, dtype=np.int32))
         for m in ("finder", "pusher", "escape")}

    epsilon    = EPSILON_START
    all_rwd    = []
    best_mean  = -np.inf
    total_ep   = [0]   # mutable ref

    # ── Stage 1: Level 1 (static box — learn wall avoidance + push) ──
    print("\n" + "═"*62)
    print(f" STAGE 1 | Level 1 (static box, wall ON) | {eps1} eps")
    print("═"*62)
    env1 = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=max_steps,
        wall_obstacles=True, difficulty=0, seed=seed,
    )
    epsilon, best_mean = run_stage(
        env1, Q, N, eps1, epsilon, rng,
        "S1", max_steps, best_path, best_mean,
        all_rwd, total_ep, 0, 0,
    )

    # ── Stage 2: Level 3 (moving + blinking) ─────────────────
    print("\n" + "═"*62)
    print(f" STAGE 2 | Level 3 (moving+blinking, wall ON) | {eps3} eps")
    print("═"*62)
    # Partial epsilon reset — re-explore with new dynamics
    epsilon = max(0.30, epsilon)

    env3 = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=max_steps,
        wall_obstacles=True, difficulty=3, box_speed=2, seed=seed,
    )
    epsilon, best_mean = run_stage(
        env3, Q, N, eps3, epsilon, rng,
        "S2", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps1, 3,
    )

    # ── Final save ────────────────────────────────────────────
    save_tables(Q, N, final_path)
    log_rewards(log_path, all_rwd)

    total_succ = sum(1 for r in all_rwd if r > 500)
    total_eps  = total_ep[0]
    print(f"\n✅ Done! | Best mean: {best_mean:.1f} | "
          f"Success: {total_succ}/{total_eps} "
          f"({total_succ/total_eps*100:.1f}%)")
    print(f"   Log    → rewards_log.csv")
    print(f"   Videos → {VIDEO_DIR}/")

    # Final evaluation videos
    print("\n🎬 Final evaluation videos (greedy)...")
    for diff, env_v in [(0, env1), (3, env3)]:
        for s in range(3):
            record_episode(env_v, Q, 99000 + s, diff, s * 137, max_steps)

    return all_rwd


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Level 3 Temporal Q(λ)")
    p.add_argument("--eps1",        type=int, default=5000)
    p.add_argument("--eps3",        type=int, default=6000)
    p.add_argument("--max_steps",   type=int, default=1000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save_prefix", type=str, default="q_table")
    args = p.parse_args()
    train(**vars(args))