"""
train_v2.py — Improved Q(λ) for OBELIX Level 3
================================================
Key fixes over exp01:

BUG FIXES:
  1. ESCAPE BUG FIXED: Old seq [L45,L45,FW,R45,R45,FW] returned robot
     to SAME direction facing the wall (only 5px away). New escape:
     adaptive 180° turn (4×45°) + 6 FW steps = 30px away.
  2. ADAPTIVE ESCAPE DIRECTION: turns AWAY from wall based on which
     side sensors are firing (left wall → turn right, right wall → left)
  3. POST-ESCAPE COOLDOWN: 8 FW steps after escape, Q-table silent,
     prevents immediate re-entry to wall zone

ALGORITHMIC IMPROVEMENTS:
  4. enable_push IN STATE: Q-table knows find vs push mode. Critical.
  5. SHAPED REWARD: dense sensor feedback during training (not one-time)
     removes non-Markovian artifact that confuses Q estimates
  6. LAMBDA = 0.70: reduced from 0.97 — targeted credit assignment,
     not "blame everyone 100 steps ago" for wall hits
  7. 4-STAGE CURRICULUM:
        S0: Level 0, no wall  → learn basic box navigation (3000 eps)
        S1: Level 0, wall ON  → add wall avoidance (3000 eps)
        S2: Level 2, blinking → add blink handling (2000 eps)
        S3: Level 3, full     → moving + blinking + wall (6000 eps)
     Partial epsilon reset between stages forces re-exploration

STATE DESIGN (max ~2304 unique states):
  (enable_push, stuck, ir, fwd_near, fwd_far,
   direction, wall_like, is_blind, vel_hint)

Run:
    python train_v2.py
    python train_v2.py --eps0 3000 --eps1 3000 --eps2 2000 --eps3 8000
"""

import sys, os, pickle, csv, random
import numpy as np
from collections import defaultdict, deque
import cv2

# ── Path setup ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# Try to find obelix.py: same dir, or ../env, or ../../env etc.
for _candidate in [_HERE,
                   os.path.join(_HERE, "env"),
                   os.path.join(_HERE, "..", "env"),
                   os.path.join(_HERE, "..", "..", "env"),
                   os.path.join(_HERE, "..", "..", "..", "env")]:
    if os.path.exists(os.path.join(_candidate, "obelix.py")):
        sys.path.insert(0, _candidate)
        break

from obelix import OBELIX  # noqa


# ══════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════
ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5

# Q(λ) parameters
LAMBDA        = 0.70   # reduced from 0.97 — cleaner credit assignment
ALPHA_0       = 0.25   # initial learning rate
ALPHA_C       = 2.0    # decay constant for N-step schedule
GAMMA         = 0.995  # 0.995^300 ≈ 0.22 — reward visible ~300 steps ahead
Q_INIT        = 0.0    # neutral init — let environment signal guide values
TRACE_MIN     = 1e-3

# Exploration
EPS_START     = 1.0
EPS_END       = 0.04
EPS_DECAY     = 0.9993  # decays over ~14000 total episodes

# Behavior parameters
BLINK_MEM       = 25   # cover max blink-off duration (30 steps)
VEL_WIN         = 5    # frames for velocity centroid estimation
ESC_SEQ_LEN     = 10   # 4 turns + 6 FW forward steps
POST_ESC_COOL   = 8    # FW cooldown after escape, Q silent

# Output
VIDEO_EVERY   = 500
VIDEO_FPS     = 15
VIDEO_DIR     = os.path.join(_HERE, "videos_v2")


# ══════════════════════════════════════════════════════════════
# Shaped Reward Wrapper  (training only — eval uses raw env)
# ══════════════════════════════════════════════════════════════
class RewardWrapper:
    """
    Wraps OBELIX to provide dense (non-one-time) sensor rewards.

    Problem in original env: sensor bonuses fire ONCE per episode.
    Same observation → different reward depending on claim history.
    This non-Markovian property confuses Q-table value estimates.

    Fix: add a small dense bonus proportional to active sensors,
    every step. The terminal bonuses (+100 attach, +2000 success,
    -200 stuck) are preserved unchanged.

    The shaped bonus is small enough that success (+2000) still
    dominates → policy prefers reaching boundary over lingering.
    Max dense bonus ≈ 8.5 per step vs 2000 for success.
    """
    # Per-sensor dense reward weights (fires every step)
    _W = np.array([
        0.3, 0.3, 0.3, 0.3,         # left sonar far/near × 2
        0.5, 0.8, 0.5, 0.8,         # fwd near/far × 4 (far→0.5, near→0.8)
        0.5, 0.8, 0.5, 0.8,         #   (bits 4-11)
        0.3, 0.3, 0.3, 0.3,         # right sonar far/near × 2
        1.5,                          # IR sensor
    ], dtype=np.float32)

    def __init__(self, raw_env: OBELIX):
        self._env       = raw_env
        self.enable_push = False

    def reset(self, seed=None):
        self.enable_push = False
        return self._env.reset(seed=seed)

    def step(self, action: str, render: bool = False):
        obs, reward, done = self._env.step(action, render=render)

        # Track attachment from reward signal (training only convenience)
        if reward > 50:
            self.enable_push = True

        # Dense sensor bonus (additive — fires every step)
        dense = float(np.dot(obs[:17].astype(np.float32), self._W))

        # Small extra bonus: forward motion toward detected box
        if action == "FW" and dense > 0 and obs[17] == 0:
            dense += 0.8

        return obs, reward + dense, done

    def __getattr__(self, name):
        return getattr(self._env, name)


# ══════════════════════════════════════════════════════════════
# State Construction  (MUST match agent_v2.py exactly)
# ══════════════════════════════════════════════════════════════
def make_state(obs: np.ndarray, history: deque, enable_push: bool) -> tuple:
    """
    Compact temporal state for Q-table lookup.

    Features (max ~2304 unique states):
      enable_push : push mode? (2)  ← CRITICAL, was missing in v1
      stuck       : wall/bound collision (2)
      ir          : box directly ahead, very close (2)
      fwd_near    : box in forward near zone (2)
      fwd_far     : box in forward far zone (2)
      direction   : box left / centered / right (3)
      wall_like   : many sensors → wide object = wall not box (2)
      is_blind    : box was visible, now gone = blink (2)
      vel_hint    : centroid drift = box moving left/right (3)

    Total: 2×2×2×2×2×3×2×2×3 = 2304
    """
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    left_s  = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_s = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s   = sum(int(obs[i]) for i in range(4, 12))
    total   = left_s + right_s + fwd_s

    # Direction to detected box
    if total == 0:
        direction = 1          # blind / unknown
    elif left_s > right_s + 1:
        direction = 0          # box on left
    elif right_s > left_s + 1:
        direction = 2          # box on right
    else:
        direction = 1          # centered / forward

    # Wall discriminator: ≥6 sensor bits active = wide object = wall
    wall_like = int(total >= 6)

    # Blink memory: was box visible recently but not now?
    hist      = list(history)
    any_now   = bool(np.any(obs[:17]))
    recent    = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind  = int(not any_now and recent)

    # Velocity hint for Level 3 moving box
    # Track centroid of box in left/right of forward cone over VEL_WIN frames
    vel_hint = 1   # default: no drift (0=left, 1=center, 2=right)
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6])  + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0:
                centroids.append((r - l) / t)   # +1=right, -1=left
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift >  0.3:
                vel_hint = 2   # drifting right → intercept ahead-right
            elif drift < -0.3:
                vel_hint = 0   # drifting left  → intercept ahead-left

    return (int(enable_push), stuck, ir, fwd_near, fwd_far,
            direction, wall_like, is_blind, vel_hint)


# ══════════════════════════════════════════════════════════════
# Deterministic Behaviors
# ══════════════════════════════════════════════════════════════
class EscapeController:
    """
    Adaptive 180° turn + sustained forward movement.

    WHY OLD CODE BROKE: [L45,L45,FW,R45,R45,FW] → robot ends at
    original angle, only 5px from wall → immediately re-hits wall.
    BUG CONFIRMED: turn net=0°, displacement=5px toward wall.

    NEW STRATEGY:
      1. Choose turn dir based on which side has more wall sensors
         (turn AWAY from wall: left sensors → turn right, vice versa)
      2. Execute 4 × 45° = 180° full reversal
      3. 6 × FW = 30px away from wall
    Robot now faces opposite direction, 30px clear. Safe.
    """
    SEQ_LEN = ESC_SEQ_LEN  # 10 steps

    def __init__(self):
        self.active  = False
        self._step   = 0
        self._turn   = 4   # action index: 0=L45, 4=R45

    def trigger(self, obs: np.ndarray):
        """Call when stuck_flag fires. Sets direction and resets counter."""
        left_w  = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
        right_w = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
        # Turn AWAY from the side with more wall sensors
        self._turn  = 4 if left_w >= right_w else 0   # 4=R45 if wall-left, 0=L45 if wall-right
        self._step  = 0
        self.active = True

    def next_action(self) -> int:
        """Returns action index for current escape step."""
        i = self._step
        self._step += 1
        if self._step >= self.SEQ_LEN:
            self.active = False
        return self._turn if i < 4 else 2   # turn×4 then FW×6


def station_action(step: int) -> int:
    """
    Gentle oscillation while box is blinking (invisible).
    Stays near last known box position, covers a small arc.
    Pattern: L22, L22, R22, R22, R22, R22, L22, L22 (period 8)
    """
    pattern = [1, 1, 3, 3, 3, 3, 1, 1]
    return pattern[step % len(pattern)]


def intercept_action(vel_hint: int, step: int) -> int:
    """
    Predictive interception for moving box (Level 3).
    Turn slightly toward where box WILL BE, not where it is.
    Pattern: 1 turn + 3 FW (curved approach path).
    """
    if vel_hint == 2:   # box drifting right → aim right
        return 3 if step % 4 == 0 else 2   # R22, FW, FW, FW
    elif vel_hint == 0: # box drifting left → aim left
        return 1 if step % 4 == 0 else 2   # L22, FW, FW, FW
    return 2            # centered → straight forward


# ══════════════════════════════════════════════════════════════
# Q-learning Helpers
# ══════════════════════════════════════════════════════════════
def get_alpha(n: int) -> float:
    """Decaying step size: α(n) = α₀ / (1 + c·n)."""
    return ALPHA_0 / (1.0 + ALPHA_C * n)


def eps_greedy(q_vals: np.ndarray, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


def save_Q(Q: dict, N: dict, path: str):
    data = {
        "Q": {str(k): v.tolist() for k, v in Q.items()},
        "N": {str(k): v.tolist() for k, v in N.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  💾 Saved {len(Q)} states → {os.path.basename(path)}")


def log_csv(path: str, rewards: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i + 1, f"{r:.2f}"])


# ══════════════════════════════════════════════════════════════
# Video Recording
# ══════════════════════════════════════════════════════════════
def record_episode(raw_env: OBELIX, Q: dict,
                   ep_num: int, difficulty: int,
                   seed: int, max_steps: int):
    """Run one greedy episode and save as MP4 for visual debugging."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fname  = os.path.join(VIDEO_DIR, f"ep{ep_num:05d}_diff{difficulty}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    obs        = raw_env.reset(seed=seed)
    history    = deque(maxlen=BLINK_MEM)
    history.append(obs.copy())
    enable_push = False
    esc         = EscapeController()
    cooldown    = 0
    blink_step  = 0
    total       = 0.0
    rng_v       = np.random.default_rng(seed + 999_999)

    for step in range(max_steps):
        raw_env._update_frames(show=False)
        frame = cv2.flip(raw_env.frame.copy(), 0)
        status = ("PUSH" if enable_push else
                  "ESC"  if esc.active   else
                  "COOL" if cooldown > 0 else "FIND")
        cv2.putText(frame,
                    f"S:{step:4d} R:{total:7.0f} [{status}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        stuck   = obs[17] == 1
        ir_on   = obs[16] == 1
        any_vis = bool(np.any(obs[:17]))
        recent  = any(bool(np.any(h[:17])) for h in history)
        blind   = not any_vis and recent
        state   = make_state(obs, history, enable_push)
        vel     = state[8]  # vel_hint

        # Trigger escape
        if stuck and not esc.active:
            esc.trigger(obs)
            cooldown = 0

        if esc.active:
            action_idx = esc.next_action()
        elif cooldown > 0:
            action_idx = 2; cooldown -= 1
            if not esc.active and cooldown == 0:
                cooldown = 0
        elif ir_on and not enable_push:
            action_idx = 2
        elif enable_push:
            action_idx = 2
        elif blind and recent:
            action_idx = station_action(blink_step); blink_step += 1
        elif any_vis and vel != 1 and not enable_push:
            action_idx = intercept_action(vel, step); blink_step = 0
        else:
            blink_step = 0
            if state in Q:
                action_idx = int(np.argmax(Q[state]))
            else:
                action_idx = int(rng_v.integers(0, N_ACTIONS))

        # Set cooldown after escape finishes
        if not esc.active and stuck:
            cooldown = POST_ESC_COOL

        obs, reward, done = raw_env.step(ACTIONS[action_idx], render=False)
        history.append(obs.copy())
        total += reward
        if reward > 50:
            enable_push = True
        if done:
            break

    if writer:
        writer.release()
    print(f"  🎬 {os.path.basename(fname)}  score={total:.0f}")


# ══════════════════════════════════════════════════════════════
# Core Training Stage
# ══════════════════════════════════════════════════════════════
def run_stage(
    wrapped_env, raw_env: OBELIX,
    Q: dict, N: dict,
    n_eps: int, epsilon: float, rng,
    name: str, max_steps: int,
    best_path: str, best_mean_ref: float,
    all_rwd: list, total_ep: list,
    vid_offset: int, difficulty: int,
    log_every: int = 100,
) -> tuple:
    """
    Generic Q(λ) training loop for one curriculum stage.

    Uses RewardWrapper (shaped rewards) for learning.
    Video recording uses raw environment (original rewards).
    Returns (epsilon, best_mean).
    """
    stage_rwd = []
    best_mean = best_mean_ref
    esc       = EscapeController()

    for ep in range(n_eps):
        ep_seed      = int(rng.integers(0, 200_000))
        obs          = wrapped_env.reset(seed=ep_seed)
        history      = deque(maxlen=BLINK_MEM)
        history.append(obs.copy())
        enable_push  = False
        cooldown     = 0
        blink_step   = 0
        total        = 0.0
        traces: dict = {}   # (state, action_idx) → eligibility value

        for step in range(max_steps):
            stuck   = obs[17] == 1
            ir_on   = obs[16] == 1
            any_vis = bool(np.any(obs[:17]))
            recent  = any(bool(np.any(h[:17])) for h in history)
            blind   = not any_vis and recent
            state   = make_state(obs, history, enable_push)
            vel     = state[8]

            # ── Trigger escape ────────────────────────────────
            if stuck and not esc.active:
                esc.trigger(obs)
                cooldown = 0
                traces   = {}   # clear traces: wall hit = start fresh

            # ── Action selection ──────────────────────────────
            use_q = False

            if esc.active:
                action_idx = esc.next_action()
                if not esc.active:          # just finished
                    cooldown = POST_ESC_COOL

            elif cooldown > 0:
                action_idx = 2              # FW to clear the wall zone
                cooldown  -= 1
                # Don't update Q during cooldown — robot in recovery

            elif ir_on and not enable_push:
                # IR active → box is RIGHT THERE → attach with FW
                action_idx = 2
                # Don't update Q for reflexive attach — clear value

            elif enable_push:
                # Attached: always push forward, Q handles stuck case
                action_idx = 2
                use_q      = True   # keep traces for push navigation

            elif blind and recent:
                # Box blinked out → station keep
                action_idx = station_action(blink_step)
                blink_step += 1

            elif any_vis and vel != 1 and not enable_push:
                # Box moving → predictive intercept
                action_idx = intercept_action(vel, step)
                blink_step = 0

            else:
                # ── Q-table decision ──────────────────────────
                blink_step = 0
                action_idx = eps_greedy(Q[state], epsilon, rng)
                use_q      = True

            action = ACTIONS[action_idx]
            next_obs, reward, done = wrapped_env.step(action, render=False)
            history.append(next_obs.copy())
            total += reward

            # Detect attachment (reward >50: attach +100 or success +2000)
            if reward > 50 and not enable_push:
                enable_push = True
                traces      = {}   # new phase: reset traces

            # ── Q(λ) with replacing traces ────────────────────
            if use_q and not esc.active and cooldown == 0:
                next_push  = wrapped_env.enable_push
                next_state = make_state(next_obs, history, next_push)
                best_next  = float(np.max(Q[next_state]))
                td_err     = reward + GAMMA * best_next - Q[state][action_idx]

                N[state][action_idx] += 1
                traces[(state, action_idx)] = 1.0   # REPLACING trace

                for (s, a), e in list(traces.items()):
                    Q[s][a] += get_alpha(int(N[s][a])) * td_err * e
                    traces[(s, a)] *= GAMMA * LAMBDA
                    if traces[(s, a)] < TRACE_MIN:
                        del traces[(s, a)]

                # Phase transition: reset traces
                if next_push != enable_push:
                    traces = {}

            obs = next_obs
            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        stage_rwd.append(total)
        all_rwd.append(total)
        total_ep[0] += 1

        # ── Logging ───────────────────────────────────────────
        if (ep + 1) % log_every == 0:
            m100  = np.mean(stage_rwd[-log_every:])
            prev  = (stage_rwd[-(2 * log_every):-log_every]
                     if len(stage_rwd) > log_every else [m100 - 1])
            trend = "✅" if m100 > np.mean(prev) else "⏳"
            succ  = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            min_s = min(stage_rwd[-log_every:])
            print(
                f"[{name} Ep {ep+1:>5}/{n_eps}] "
                f"Mean: {m100:>9.1f} | Min: {min_s:>9.0f} | "
                f"ε: {epsilon:.4f} | Succ/{log_every}: {succ:3d} | "
                f"States: {len(Q):5d}  {trend}"
            )
            if m100 > best_mean:
                best_mean = m100
                save_Q(Q, N, best_path)

        # ── Video recording ───────────────────────────────────
        if raw_env is not None and (ep + 1) % VIDEO_EVERY == 0:
            record_episode(
                raw_env, Q,
                vid_offset + total_ep[0],
                difficulty,
                int(rng.integers(0, 10_000)),
                max_steps,
            )

    return epsilon, best_mean


# ══════════════════════════════════════════════════════════════
# Main Training Function
# ══════════════════════════════════════════════════════════════
def train(
    eps0       = 3000,   # Stage 0: Level 0, no wall
    eps1       = 3000,   # Stage 1: Level 0, wall ON
    eps2       = 2000,   # Stage 2: Level 2, blinking
    eps3       = 6000,   # Stage 3: Level 3, full
    max_steps  = 1000,
    seed       = 42,
    prefix     = "q_table_v2",
    box_speed  = 2,
):
    best_path  = os.path.join(_HERE, f"{prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{prefix}_final.pkl")
    log_path   = os.path.join(_HERE, "rewards_v2.csv")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    Q = defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
    N = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))

    epsilon   = EPS_START
    all_rwd   = []
    best_mean = -np.inf
    total_ep  = [0]

    def make_env(wall: bool, diff: int, spd: int = 0) -> tuple:
        raw = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall, difficulty=diff,
            box_speed=spd, seed=seed,
        )
        return RewardWrapper(raw), raw

    # ── Stage 0: No wall — learn basic find + push ────────────
    print("\n" + "═" * 62)
    print(f" STAGE 0 | Level 0 | NO WALL | {eps0} eps")
    print(" Learn: locate box, navigate, attach, push to boundary")
    print("═" * 62)
    env0w, env0r = make_env(wall=False, diff=0)
    epsilon, best_mean = run_stage(
        env0w, env0r, Q, N, eps0, epsilon, rng,
        "S0", max_steps, best_path, best_mean,
        all_rwd, total_ep, 0, 0,
    )

    # ── Stage 1: Add wall — learn avoidance ──────────────────
    print("\n" + "═" * 62)
    print(f" STAGE 1 | Level 0 | WALL ON | {eps1} eps")
    print(" Learn: navigate through wall gap, avoid wall hits")
    print("═" * 62)
    epsilon = max(0.40, epsilon)   # partial reset: re-explore with wall
    env1w, env1r = make_env(wall=True, diff=0)
    epsilon, best_mean = run_stage(
        env1w, env1r, Q, N, eps1, epsilon, rng,
        "S1", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0, 0,
    )

    # ── Stage 2: Blinking box ─────────────────────────────────
    print("\n" + "═" * 62)
    print(f" STAGE 2 | Level 2 | BLINKING | {eps2} eps")
    print(" Learn: station-keep during blink, resume when visible")
    print("═" * 62)
    epsilon = max(0.30, epsilon)
    env2w, env2r = make_env(wall=True, diff=2)
    epsilon, best_mean = run_stage(
        env2w, env2r, Q, N, eps2, epsilon, rng,
        "S2", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1, 2,
    )

    # ── Stage 3: Full Level 3 ─────────────────────────────────
    print("\n" + "═" * 62)
    print(f" STAGE 3 | Level 3 | MOVING+BLINKING+WALL | {eps3} eps")
    print(" Learn: intercept moving box, handle all difficulty knobs")
    print("═" * 62)
    epsilon = max(0.25, epsilon)
    env3w, env3r = make_env(wall=True, diff=3, spd=box_speed)
    epsilon, best_mean = run_stage(
        env3w, env3r, Q, N, eps3, epsilon, rng,
        "S3", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1 + eps2, 3,
    )

    # ── Save & report ─────────────────────────────────────────
    save_Q(Q, N, final_path)
    log_csv(log_path, all_rwd)

    total_eps  = total_ep[0]
    total_succ = sum(1 for r in all_rwd if r > 500)
    last_100   = np.mean(all_rwd[-100:]) if len(all_rwd) >= 100 else np.mean(all_rwd)

    print(f"\n{'═'*62}")
    print(f" TRAINING COMPLETE")
    print(f"{'═'*62}")
    print(f"  Best mean (100-ep): {best_mean:.1f}")
    print(f"  Last 100 mean:      {last_100:.1f}")
    print(f"  Success rate:       {total_succ}/{total_eps} "
          f"({total_succ/total_eps*100:.1f}%)")
    print(f"  Q-table states:     {len(Q)}")
    print(f"  Best weights:       {os.path.basename(best_path)}")
    print(f"  Rewards log:        {log_path}")
    print(f"  Videos:             {VIDEO_DIR}/")

    # Final evaluation videos
    print("\n🎬 Recording final evaluation videos...")
    for diff, env_r, lbl in [(0, env0r, "static"), (3, env3r, "L3")]:
        for s in range(3):
            record_episode(env_r, Q, 99_000 + s * 10 + (3 if diff == 3 else 0),
                           diff, s * 137, max_steps)

    return all_rwd


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OBELIX Level 3 — Improved Q(λ) v2")
    p.add_argument("--eps0",      type=int,   default=3000,
                   help="Stage 0 episodes (no wall)")
    p.add_argument("--eps1",      type=int,   default=3000,
                   help="Stage 1 episodes (wall)")
    p.add_argument("--eps2",      type=int,   default=2000,
                   help="Stage 2 episodes (blinking)")
    p.add_argument("--eps3",      type=int,   default=6000,
                   help="Stage 3 episodes (Level 3 full)")
    p.add_argument("--max_steps", type=int,   default=1000)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--prefix",    type=str,   default="q_table_v2")
    p.add_argument("--box_speed", type=int,   default=2)
    args = p.parse_args()
    train(**vars(args))