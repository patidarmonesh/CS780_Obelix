"""
train_exp06plus.py  — OBELIX Exp06+ (Improved Dyna-Q Lambda)
=============================================================
Algorithm : Dyna-Q(λ) with replacing traces, LRU model

WHAT IS FIXED vs EXP06
══════════════════════
FIX 1 · WALL_LIKE THRESHOLD (Feature 6)
  exp06: wall_like = total >= 8 only
  Problem: Vertical wall at x=250 approached HEAD-ON fires only forward
  sonars (obs[4:12]). At slight angles, total = 5-7, NOT flagged.
  Fix A: Lower threshold to 7 AND add angled-approach patterns:
           (left >= 2 AND front >= 3)  →  approaching wall from angle
           (right >= 2 AND front >= 3) →  same from other side
           (left >= 2 AND right >= 2)  →  alongside wall, both sides fire
  Fix B: Add wall_timer_bucket as state feature 12 (NEW). Counts how
  many consecutive steps the wall was firing. Q-table learns different
  policies for "seen wall for 5+ steps" vs "just appeared now" (box).

FIX 2 · FW_ALWAYS REWARD CAUSES WALL APPROACH
  exp06: FW_ALWAYS=0.5 fires on EVERY FW step with obs[17]==0
  Problem: Even when total >= 4 (near wall), FW gets +0.5.
           Q-table converges to Q(FW|near_wall) ≈ Q(turn|near_wall).
  Fix: FW_ALWAYS only fires when wall_timer == 0 AND total == 0.
  This rewards exploration-style forward movement, NOT wall approach.

FIX 3 · BOUNDARY ESCAPE DIRECTION
  exp06: EscapeController turns = R45 when lw==rw==0 (boundary stuck)
  because `turn = R45 if lw >= rw` and 0 >= 0 → always R45.
  Problem: corner scenario → always same direction → re-hits corner.
  Fix: When total==0 (arena boundary), randomly pick L45 or R45 for
  diversity, AND add a "double-escape" cooldown so it walks further.

FIX 4 · EPSILON MANAGEMENT AT STAGE TRANSITIONS
  exp06: S1 resets epsilon to max(0.40, current) only
  Fix: S1 resets to max(0.55, current) — wall is a completely new
  obstacle that requires aggressive re-exploration to learn avoidance.

FIX 5 · MORE TRAINING BUDGET (fits in ~4-6 hours CPU)
  exp06: 19,000 total episodes
  exp06+: 33,000 total episodes  (S0=6k, S1=8k, S2=5k, S3=14k)
  Tabular Q is so fast (~0.3-0.5ms/step) that more episodes = free.

NEW STATE FEATURE (13 components → ~1200 practical unique states)
═══════════════════════════════════════════════════════════════════
0  enable_push       : attached to box? (2)
1  stuck             : wall/boundary collision flag (2)
2  ir                : IR sensor active (2)
3  fwd_near          : any forward near sonar (2)
4  fwd_far           : any forward far sonar (2)
5  direction         : box left/center/right (3)
6  wall_like         : improved instant discriminator (2)
7  is_blind          : was visible, now gone (blink) (2)
8  vel_hint          : centroid drift for moving box (3)
9  fast_grow         : sudden +4 sensor bits = wall warning (2)
10 was_close         : IR seen in last 15 steps (2)
11 moving_blind      : blind + recently pursuing box (2)
12 wall_timer_bucket : temporal wall history 0/1/2 (NEW, 3 values)

Theoretical: 2^11 × 3^3 = 27,648 | Practical: ~1,200

CURRICULUM
══════════
S0: diff=0, no wall  → 6,000 eps  (learn find + push)
S1: diff=0, wall ON  → 8,000 eps  (learn wall avoidance, doubled!)
S2: diff=2, wall ON  → 5,000 eps  (learn blink handling)
S3: diff=3, wall ON  → 14,000 eps (full Level 3)
"""

import sys, os, pickle, csv, random, time
import numpy as np
from collections import defaultdict, deque, OrderedDict
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [
    _HERE,
    os.path.join(_HERE, "env"),
    os.path.join(_HERE, "..", "env"),
    os.path.join(_HERE, "..", "..", "env"),
    os.path.join(_HERE, "..", "..", "..", "env"),
]:
    if os.path.exists(os.path.join(_c, "obelix.py")):
        sys.path.insert(0, _c)
        break

from obelix import OBELIX  # noqa

# ══════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════
ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5

# Q(λ) core  (unchanged from exp06 — already well-tuned)
LAMBDA    = 0.70
ALPHA_0   = 0.20
ALPHA_C   = 2.0
GAMMA     = 0.995
Q_INIT    = 3.0
TRACE_MIN = 1e-3

# Exploration
EPS_START = 1.0
EPS_END   = 0.04
EPS_DECAY = 0.9992      # ~17k eps to reach 0.04; fine for 33k total

# Behavioral controllers
BLINK_MEM   = 30
VEL_WIN     = 5
ESC_TURNS   = 4         # 4 × 45° = 180° true reversal
ESC_FW      = 6         # 30px clearance
POST_COOL   = 10        # FIX: +2 more cooldown steps vs exp06
PUSH_RECOV  = 6
PURSUIT_TTL = 18

# Dyna-Q
N_PLAN    = 30          # FIX: increased from 20 → 30 (free planning)
MODEL_CAP = 4000        # FIX: increased from 3000 → 4000

# Reward shaping
#   FW_ALWAYS is now CONDITIONAL (see RewardWrapper) — not always-on
FW_ALWAYS = 0.6         # slightly higher reward for safe exploration
SENSOR_W  = np.array([
    0.3, 0.3, 0.3, 0.3,   # left sonar far/near ×2
    0.5, 0.9, 0.5, 0.9,   # fwd far, fwd near × 4
    0.5, 0.9, 0.5, 0.9,
    0.3, 0.3, 0.3, 0.3,   # right sonar far/near ×2
    2.0,                   # IR: box right here
], dtype=np.float32)

# Temporal wall detection threshold for bucket-2
WALL_TIMER_LONG = 5     # >= 5 consecutive wall-like steps = "definitely wall"

# Output
VIDEO_EVERY = 500
VIDEO_FPS   = 15
VIDEO_DIR   = os.path.join(_HERE, "videos_exp06plus")


# ══════════════════════════════════════════════════════════════
# Dense Reward Wrapper  (training only)
# ══════════════════════════════════════════════════════════════
class RewardWrapper:
    """
    Key changes vs exp06:
    1. Tracks wall_timer (for conditional FW_ALWAYS and state feature 12)
    2. FW_ALWAYS only fires when wall_timer == 0 AND total == 0
       → rewards exploration but NOT wall-approach movement
    3. wall_timer is accessible as self.wall_timer for state construction
    """

    def __init__(self, raw_env: OBELIX):
        self._env       = raw_env
        self.enable_push = False
        self.wall_timer  = 0    # NEW: consecutive wall-like steps

    def reset(self, seed=None):
        self.enable_push = False
        self.wall_timer  = 0    # reset on episode boundary
        return self._env.reset(seed=seed)

    def step(self, action: str, render: bool = False):
        obs, reward, done = self._env.step(action, render=render)
        if reward > 50:
            self.enable_push = True

        # ── Update wall_timer (used for FW_ALWAYS and state feature 12) ──
        left  = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
        right = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
        fwd   = sum(int(obs[i]) for i in range(4, 12))
        total = left + right + fwd

        wall_now = (
            total >= 7
            or (left >= 2 and right >= 2)
            or (left >= 2 and fwd >= 3)
            or (right >= 2 and fwd >= 3)
        )
        if wall_now:
            self.wall_timer = min(self.wall_timer + 1, 20)
        else:
            self.wall_timer = max(self.wall_timer - 1, 0)

        # ── Dense sensor bonus (same as exp06) ──
        dense = float(np.dot(obs[:17].astype(np.float32), SENSOR_W))

        # ── FW_ALWAYS: only when CLEAR (wall_timer==0 AND no sensors) ──
        # FIX: exp06 fired this even near walls → trained wall-approach.
        if action == "FW" and obs[17] == 0 and self.wall_timer == 0 and total == 0:
            dense += FW_ALWAYS

        return obs, reward + dense, done

    def __getattr__(self, name):
        return getattr(self._env, name)


# ══════════════════════════════════════════════════════════════
# State Construction  (MUST match agent_exp06plus.py exactly)
# ══════════════════════════════════════════════════════════════
def _wall_like_instant(obs: np.ndarray) -> int:
    """
    Improved single-step wall discriminator.
    FIX: exp06 only used total >= 8. Added angled-approach patterns
    so the vertical wall is caught even at 5-7 total bits.
    """
    left  = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd   = sum(int(obs[i]) for i in range(4, 12))
    total = left + right + fwd

    if total >= 7:                           # very dense = wall
        return 1
    if left >= 2 and right >= 2:             # both sides = wide obstacle
        return 1
    if left >= 2 and fwd >= 3:               # angled from left
        return 1
    if right >= 2 and fwd >= 3:              # angled from right
        return 1
    return 0


def make_state(
    obs:           np.ndarray,
    history:       deque,
    enable_push:   bool,
    last_dir:      int,
    pursuit_steps: int,
    wall_timer:    int,       # NEW parameter (pass wrapped_env.wall_timer)
) -> tuple:
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8] or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9] or obs[11])

    left_s = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right_s = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s   = sum(int(obs[i]) for i in range(4, 12))
    total   = left_s + right_s + fwd_s

    # Direction
    if total == 0:
        direction = 1
    elif left_s > right_s + 1:
        direction = 0   # left
    elif right_s > left_s + 1:
        direction = 2   # right
    else:
        direction = 1   # centered

    # Improved wall discriminator
    wall_like = _wall_like_instant(obs)

    hist = list(history)
    any_now = bool(np.any(obs[:17]))
    recent  = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind = int(not any_now and recent)

    was_close = int(any(bool(h[16] == 1) for h in hist[-15:]) if hist else False)

    vel_hint = 1
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6]) + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0:
                centroids.append((r - l) / t)
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift > 0.3:  vel_hint = 2
            elif drift < -0.3: vel_hint = 0

    fast_grow = 0
    if len(hist) >= 2:
        prev = sum(int(x) for x in hist[-2][:16])
        fast_grow = int((total - prev) >= 4)

    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)

    # NEW: wall_timer_bucket — temporal wall history
    # 0 = no recent wall contact (fully clear)
    # 1 = brief wall contact (1-4 steps, might be box at close range)
    # 2 = sustained wall contact (5+ steps = definitely wall)
    if wall_timer == 0:
        wall_timer_bucket = 0
    elif wall_timer < WALL_TIMER_LONG:
        wall_timer_bucket = 1
    else:
        wall_timer_bucket = 2

    return (
        int(enable_push),   # 0
        stuck,              # 1
        ir,                 # 2
        fwd_near,           # 3
        fwd_far,            # 4
        direction,          # 5
        wall_like,          # 6  (improved)
        is_blind,           # 7
        vel_hint,           # 8
        fast_grow,          # 9
        was_close,          # 10
        moving_blind,       # 11
        wall_timer_bucket,  # 12 (NEW)
    )


# ══════════════════════════════════════════════════════════════
# Behavioral Controllers
# ══════════════════════════════════════════════════════════════
class EscapeController:
    """
    FIX vs exp06:
    - When boundary stuck (total==0): pick a random direction (50/50 L/R)
      instead of always defaulting to R45. Prevents corner trap where
      same turn always re-hits the same boundary corner.
    - POST_COOL increased from 8 → 10 (extra margin from wall).
    """
    SEQ_LEN = ESC_TURNS + ESC_FW  # 10 steps

    def __init__(self):
        self.active = False
        self._step  = 0
        self._turn  = 4  # 4=R45, 0=L45

    def trigger(self, obs: np.ndarray, rng=None):
        total = sum(int(obs[i]) for i in range(16))

        if total == 0:
            # ═══ BOUNDARY / CORNER STUCK (Method 4's failure point) ═══
            # Arena boundary is NOT in combined_object_frame → zero sensors.
            # Randomly pick L or R for diversity — breaks corner trap.
            if rng is not None:
                self._turn = int(rng.integers(0, 2)) * 4  # 0=L45, 4=R45
            else:
                self._turn = 0 if random.random() < 0.5 else 4
        else:
            # WALL / OBSTACLE STUCK: turn away from wall side
            lw = sum(int(obs[i]) for i in range(0, 4))
            rw = sum(int(obs[i]) for i in range(12, 16))
            self._turn = 4 if lw >= rw else 0

        self._step  = 0
        self.active = True

    def next_action(self) -> int:
        i = self._step
        self._step += 1
        if self._step >= self.SEQ_LEN:
            self.active = False
        return self._turn if i < ESC_TURNS else 2


class PushRecovery:
    """Unchanged from exp06 — already correct behavior."""

    def __init__(self):
        self.active = False
        self._step  = 0
        self._turn  = 1

    def trigger(self, obs: np.ndarray):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 1 if lw <= rw else 3
        self._step = 0
        self.active = True

    def next_action(self) -> int:
        i = self._step
        self._step += 1
        if self._step >= PUSH_RECOV:
            self.active = False
        return self._turn if i % 2 == 0 else 2


def intercept_action(vel_hint: int, step: int) -> int:
    if vel_hint == 2: return 3 if step % 4 == 0 else 2
    if vel_hint == 0: return 1 if step % 4 == 0 else 2
    return 2


def station_action(step: int) -> int:
    pattern = [1, 1, 3, 3, 3, 3, 1, 1]
    return pattern[step % 8]


# ══════════════════════════════════════════════════════════════
# Q-learning Helpers
# ══════════════════════════════════════════════════════════════
def get_alpha(n: int) -> float:
    return ALPHA_0 / (1.0 + ALPHA_C * n)


def eps_greedy(q_vals: np.ndarray, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


class LRUModel:
    def __init__(self, cap: int = MODEL_CAP):
        self._store: OrderedDict = OrderedDict()
        self._cap = cap

    def update(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._cap:
            self._store.popitem(last=False)

    def sample(self, k: int, rng) -> list:
        if not self._store:
            return []
        keys   = list(self._store.keys())
        chosen = rng.choice(len(keys), size=min(k, len(keys)), replace=False)
        return [(keys[i], self._store[keys[i]]) for i in chosen]

    def __len__(self):
        return len(self._store)


def save_Q(Q: dict, N: dict, path: str):
    data = {
        "Q": {str(k): v.tolist() for k, v in Q.items()},
        "N": {str(k): v.tolist() for k, v in N.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  💾 {len(Q)} states → {os.path.basename(path)}")


def log_csv(path: str, rewards: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i + 1, f"{r:.2f}"])


# ══════════════════════════════════════════════════════════════
# Core Training Stage
# ══════════════════════════════════════════════════════════════
def run_stage(
    wrapped_env, raw_env: OBELIX,
    Q: dict, N: dict,
    model: LRUModel,
    n_eps: int, epsilon: float, rng,
    name: str, max_steps: int,
    best_path: str, best_mean_ref: float,
    all_rwd: list, total_ep: list,
    vid_offset: int, difficulty: int,
    n_plan: int = N_PLAN,
    log_every: int = 200,
) -> tuple:
    stage_rwd = []
    best_mean = best_mean_ref
    esc      = EscapeController()
    push_rec = PushRecovery()

    for ep in range(n_eps):
        ep_seed     = int(rng.integers(0, 300_000))
        obs         = wrapped_env.reset(seed=ep_seed)
        history     = deque(maxlen=BLINK_MEM)
        history.append(obs.copy())
        enable_push = False
        cooldown    = 0
        blink_step  = 0
        last_dir    = 2
        pursuit_st  = 999
        total       = 0.0
        traces: dict = {}

        for step in range(max_steps):
            stuck   = obs[17] == 1
            ir_on   = obs[16] == 1
            any_vis = bool(np.any(obs[:17]))

            # Capture wall_timer BEFORE step (for current-state)
            cur_wt = wrapped_env.wall_timer

            state = make_state(obs, history, enable_push, last_dir, pursuit_st, cur_wt)
            vel   = state[8]
            mb    = state[11]

            # Track sensor contact for pursuit momentum
            if any(int(obs[i]) for i in range(16)):
                pursuit_st = 0
                if obs[12] or obs[13] or obs[14] or obs[15]: last_dir = 3
                elif obs[0] or obs[1] or obs[2] or obs[3]:   last_dir = 1
                else:                                          last_dir = 2
            else:
                pursuit_st = min(pursuit_st + 1, 999)

            # Controller triggers
            if enable_push and stuck and not push_rec.active:
                push_rec.trigger(obs)
                traces = {}
            elif not enable_push and stuck and not esc.active:
                esc.trigger(obs, rng)   # FIX: pass rng for random boundary escape
                cooldown = 0
                traces   = {}

            # Action selection
            use_q = False
            if push_rec.active:
                action_idx = push_rec.next_action()
            elif esc.active:
                action_idx = esc.next_action()
                if not esc.active:
                    cooldown = POST_COOL
            elif cooldown > 0:
                action_idx = 2; cooldown -= 1
            elif ir_on and not enable_push:
                action_idx = 2
            elif enable_push:
                action_idx = 2
                use_q      = True
            elif mb:
                action_idx = last_dir
            elif any_vis and vel != 1 and not enable_push:
                action_idx = intercept_action(vel, step)
                blink_step = 0
            elif state[7]:
                action_idx = station_action(blink_step)
                blink_step += 1
            else:
                blink_step = 0
                action_idx = eps_greedy(Q[state], epsilon, rng)
                use_q      = True

            # Environment step
            action   = ACTIONS[action_idx]
            next_obs, reward, done = wrapped_env.step(action, render=False)
            history.append(next_obs.copy())
            total += reward

            if not enable_push and reward > 50:
                enable_push = True
                traces      = {}

            # Q(λ) update with replacing traces
            if use_q and not esc.active and not push_rec.active and cooldown == 0:
                next_push  = wrapped_env.enable_push
                next_st    = (pursuit_st + 1 if not any(int(next_obs[i]) for i in range(16)) else 0)
                next_state = make_state(next_obs, history, next_push, last_dir, next_st,
                                        wrapped_env.wall_timer)   # uses NEW wall_timer
                best_next  = float(np.max(Q[next_state]))
                td_err     = reward + GAMMA * best_next - Q[state][action_idx]

                N[state][action_idx] += 1
                traces[(state, action_idx)] = 1.0  # replacing trace

                for (s, a), e_val in list(traces.items()):
                    Q[s][a]  += get_alpha(int(N[s][a])) * td_err * e_val
                    new_e     = e_val * GAMMA * LAMBDA
                    if new_e < TRACE_MIN:
                        del traces[(s, a)]
                    else:
                        traces[(s, a)] = new_e

                if next_push != enable_push:
                    traces = {}

                # Dyna-Q planning
                model.update((state, action_idx), (reward, next_state))
                for (ms, ma), (mr, mns) in model.sample(n_plan, rng):
                    mp_best   = float(np.max(Q[mns]))
                    Q[ms][ma] += get_alpha(int(N[ms][ma])) * (mr + GAMMA * mp_best - Q[ms][ma])

            obs = next_obs
            if done:
                break

        # End of episode
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        stage_rwd.append(total)
        all_rwd.append(total)
        total_ep[0] += 1

        if (ep + 1) % log_every == 0:
            m_rwd = np.mean(stage_rwd[-log_every:])
            prev  = (stage_rwd[-(2*log_every):-log_every]
                     if len(stage_rwd) > log_every else [m_rwd - 1])
            trend = "✅" if m_rwd > np.mean(prev) else "⏳"
            succ  = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            min_r = min(stage_rwd[-log_every:])
            print(
                f"[{name} Ep {ep+1:>5}/{n_eps}] "
                f"Mean:{m_rwd:>9.1f} | Min:{min_r:>9.0f} | "
                f"ε:{epsilon:.4f} | Succ/{log_every}:{succ:3d} | "
                f"States:{len(Q):5d} | Model:{len(model):4d} {trend}"
            )
            if m_rwd > best_mean:
                best_mean = m_rwd
                save_Q(Q, N, best_path)

        if raw_env is not None and (ep + 1) % VIDEO_EVERY == 0:
            record_episode(raw_env, Q, vid_offset + total_ep[0],
                           difficulty, int(rng.integers(0, 10_000)), max_steps)

    return epsilon, best_mean


# ══════════════════════════════════════════════════════════════
# Video Recording  (for visual inspection)
# ══════════════════════════════════════════════════════════════
def record_episode(raw_env: OBELIX, Q: dict, ep_num: int,
                   difficulty: int, seed: int, max_steps: int):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fname  = os.path.join(VIDEO_DIR, f"ep{ep_num:05d}_diff{difficulty}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    obs         = raw_env.reset(seed=seed)
    history     = deque(maxlen=BLINK_MEM)
    history.append(obs.copy())
    enable_push = False
    esc         = EscapeController()
    push_rec    = PushRecovery()
    cooldown    = 0
    blink_step  = 0
    last_dir    = 2
    pursuit_st  = 999
    wall_timer  = 0   # local for video (no wrapper)

    total   = 0.0
    rng_v   = np.random.default_rng(seed + 77_777)

    for step in range(max_steps):
        raw_env._update_frames(show=False)
        frame = cv2.flip(raw_env.frame.copy(), 0)
        mode  = ("PUSH" if enable_push else
                 "ESC"  if esc.active  else
                 "PREC" if push_rec.active else
                 "COOL" if cooldown > 0 else "FIND")
        cv2.putText(frame, f"S:{step:4d} R:{total:7.0f} [{mode}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        if writer is None:
            h, w   = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        # Update local wall_timer for video recording
        left   = sum(int(obs[i]) for i in range(0, 4))
        right  = sum(int(obs[i]) for i in range(12, 16))
        fwd    = sum(int(obs[i]) for i in range(4, 12))
        t_vid  = left + right + fwd
        wall_now = (t_vid >= 7 or (left >= 2 and right >= 2)
                    or (left >= 2 and fwd >= 3) or (right >= 2 and fwd >= 3))
        wall_timer = min(wall_timer + 1, 20) if wall_now else max(wall_timer - 1, 0)

        stuck  = obs[17] == 1
        ir_on  = obs[16] == 1
        any_vs = bool(np.any(obs[:17]))
        state  = make_state(obs, history, enable_push, last_dir, pursuit_st, wall_timer)
        vel    = state[8]
        mb     = state[11]

        if any(int(obs[i]) for i in range(16)):
            pursuit_st = 0
            if obs[12] or obs[13] or obs[14] or obs[15]: last_dir = 3
            elif obs[0] or obs[1] or obs[2] or obs[3]:   last_dir = 1
            else:                                          last_dir = 2
        else:
            pursuit_st = min(pursuit_st + 1, 999)

        if enable_push and stuck and not push_rec.active:
            push_rec.trigger(obs)
        elif not enable_push and stuck and not esc.active:
            esc.trigger(obs, rng_v)

        if push_rec.active:     action_idx = push_rec.next_action()
        elif esc.active:
            action_idx = esc.next_action()
            if not esc.active:  cooldown = POST_COOL
        elif cooldown > 0:      action_idx = 2; cooldown -= 1
        elif ir_on and not enable_push: action_idx = 2
        elif enable_push:       action_idx = 2
        elif mb:                action_idx = last_dir
        elif any_vs and vel != 1 and not enable_push:
            action_idx = intercept_action(vel, step); blink_step = 0
        elif state[7]:          action_idx = station_action(blink_step); blink_step += 1
        else:
            blink_step = 0
            action_idx = (int(np.argmax(Q[state])) if state in Q
                          else int(rng_v.integers(0, N_ACTIONS)))

        obs, reward, done = raw_env.step(ACTIONS[action_idx], render=False)
        history.append(obs.copy())
        total += reward
        if reward > 50:
            enable_push = True
        if done:
            break

    if writer:
        writer.release()
    print(f"  🎬 {os.path.basename(fname)} score={total:.0f}")


# ══════════════════════════════════════════════════════════════
# Main Training
# ══════════════════════════════════════════════════════════════
def train(
    eps0      = 6000,    # Stage 0: diff=0, no wall  (FIX: +2k vs exp06)
    eps1      = 8000,    # Stage 1: diff=0, wall ON   (FIX: +4k — most critical!)
    eps2      = 5000,    # Stage 2: diff=2, blinking  (FIX: +2k)
    eps3      = 14000,   # Stage 3: diff=3, full      (FIX: +6k)
    max_steps = 1000,
    seed      = 42,
    prefix    = "q_exp06plus",
    box_speed = 2,
    n_plan    = N_PLAN,
):
    best_path  = os.path.join(_HERE, f"{prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{prefix}_final.pkl")
    log_path   = os.path.join(_HERE, f"{prefix}_rewards.csv")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    Q     = defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
    N     = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))
    model = LRUModel(cap=MODEL_CAP)

    epsilon   = EPS_START
    all_rwd   = []
    best_mean = -np.inf
    total_ep  = [0]
    t0        = time.time()

    def make_env(wall: bool, diff: int, spd: int = 0):
        raw = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall, difficulty=diff, box_speed=spd, seed=seed,
        )
        return RewardWrapper(raw), raw

    # ── Stage 0: No wall ──────────────────────────────────────
    print("\n" + "═" * 65)
    print(f" STAGE 0 | diff=0 | NO WALL | {eps0} eps")
    print("═" * 65)
    w0, r0 = make_env(wall=False, diff=0)
    epsilon, best_mean = run_stage(
        w0, r0, Q, N, model, eps0, epsilon, rng,
        "S0", max_steps, best_path, best_mean,
        all_rwd, total_ep, 0, 0, n_plan,
    )

    # ── Stage 1: Wall ON ──────────────────────────────────────
    print("\n" + "═" * 65)
    print(f" STAGE 1 | diff=0 | WALL ON | {eps1} eps  [FIX: more eps + higher ε reset]")
    print("═" * 65)
    epsilon = max(0.55, epsilon)    # FIX: was 0.40 — wall needs more re-exploration
    w1, r1  = make_env(wall=True, diff=0)
    epsilon, best_mean = run_stage(
        w1, r1, Q, N, model, eps1, epsilon, rng,
        "S1", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0, 0, n_plan,
    )

    # ── Stage 2: Blinking ─────────────────────────────────────
    print("\n" + "═" * 65)
    print(f" STAGE 2 | diff=2 | BLINKING | {eps2} eps")
    print("═" * 65)
    epsilon = max(0.35, epsilon)
    w2, r2  = make_env(wall=True, diff=2)
    epsilon, best_mean = run_stage(
        w2, r2, Q, N, model, eps2, epsilon, rng,
        "S2", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1, 2, n_plan,
    )

    # ── Stage 3: Full Level 3 ─────────────────────────────────
    print("\n" + "═" * 65)
    print(f" STAGE 3 | diff=3 | MOVING+BLINKING+WALL | {eps3} eps")
    print("═" * 65)
    epsilon = max(0.30, epsilon)
    w3, r3  = make_env(wall=True, diff=3, spd=box_speed)
    epsilon, best_mean = run_stage(
        w3, r3, Q, N, model, eps3, epsilon, rng,
        "S3", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1 + eps2, 3, n_plan,
    )

    # ── Final save ────────────────────────────────────────────
    save_Q(Q, N, final_path)
    log_csv(log_path, all_rwd)

    elapsed    = (time.time() - t0) / 60
    succ_total = sum(1 for r in all_rwd if r > 500)
    last_200   = np.mean(all_rwd[-200:]) if len(all_rwd) >= 200 else np.mean(all_rwd)

    print(f"\n{'═' * 65}")
    print(f" TRAINING COMPLETE  ({elapsed:.1f} minutes)")
    print(f"{'═' * 65}")
    print(f"  Best 200-ep mean  : {best_mean:.1f}")
    print(f"  Last 200-ep mean  : {last_200:.1f}")
    print(f"  Success rate      : {succ_total}/{total_ep[0]} ({succ_total/total_ep[0]*100:.1f}%)")
    print(f"  Q-table states    : {len(Q)}")
    print(f"  Best weights      : {os.path.basename(best_path)}")
    print(f"  Rewards log       : {log_path}")

    print("\n🎬 Final evaluation videos...")
    for vid_diff, raw_env_v in [(0, r0), (3, r3)]:
        for s in range(3):
            record_episode(raw_env_v, Q, 99_000 + s + vid_diff * 10,
                           vid_diff, s * 137, max_steps)

    return all_rwd


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OBELIX Exp06+ — Improved Dyna-Q(λ)")
    p.add_argument("--eps0",       type=int,   default=6000)
    p.add_argument("--eps1",       type=int,   default=8000)
    p.add_argument("--eps2",       type=int,   default=5000)
    p.add_argument("--eps3",       type=int,   default=14000)
    p.add_argument("--max_steps",  type=int,   default=1000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--prefix",     type=str,   default="q_exp06plus")
    p.add_argument("--box_speed",  type=int,   default=2)
    p.add_argument("--n_plan",     type=int,   default=30)
    train(**vars(p.parse_args()))
