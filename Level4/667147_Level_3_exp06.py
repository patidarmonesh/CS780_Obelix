"""
exp06_train.py  —  OBELIX Level 3  —  Unified Best Approach
=============================================================
Algorithm : Dyna-Q(λ)  (tabular, replacing traces, LRU model)

NOTE ON GPU
───────────
Tabular Q does NOT benefit from GPU — state space is ~800 unique
states, NumPy is faster than CUDA overhead for this size.
GPU acceleration would only help for neural-net policies (DDQN).
For this approach, more episodes = better results.
Run multiple seeds in parallel on CPU cores instead.

─────────────────────────────────────────────────────────────
BUGS FIXED vs all previous experiments
─────────────────────────────────────────────────────────────

From EXP01 (Dyna-Q baseline):
  ✅ Escape [L45×4,FW×2]: only 2 FW steps, robot re-hits wall
     → New: 4 turns + 6 FW = true 180° + 30px clearance
  ✅ enable_push NOT in state: same obs, different optimal action
     → New: enable_push is state[0]
  ✅ Dyna-Q model accumulates stale early-exploration transitions
     → New: LRU eviction, model capped at MODEL_CAP entries

From EXP02 (Temporal Q-lambda):
  ✅ blind → station_action (L22/R22 oscillate): for MOVING box,
     this loses the box while it continues moving
     → New: moving_blind → maintain last pursuit direction (exp01 insight)
  ✅ No post-escape cooldown: robot immediately re-enters wall zone
     → New: POST_COOL FW steps after escape, Q-table silent
  ✅ push+stuck → escape: escape pulls box AWAY from boundary
     → New: push+stuck → PushRecovery (angle adjustment, no detach)
  ✅ spread_high threshold unreliable: box close also fires side sensors
     → New: wall_like uses total >= 8 bits (box far: 2-4 bits, wall: 8+)
  ✅ escape [L45,L45,FW,R45,R45,FW]: net 0° turn, 5px from wall
     → Same as exp01 fix above

From EXP04 (improved Q-lambda):
  ✅ Q_INIT = 0.0: per-step -1 makes Q(FW) < Q(turn) early on
     → New: Q_INIT = +3.0 (optimistic, FW gets tried first)
  ✅ FW dense bonus only when sensors active: no incentive to explore
     → New: +FW_ALWAYS for FW action even when sensors dark
  ✅ moving_blind feature in STATE but not in ACTION SELECTION
     → New: if moving_blind → last_dir action (actual pursuit momentum)
  ✅ station_action during blind (L3 box moves while robot waits)
     → Only for stationary box; L3 uses pursuit_direction instead

─────────────────────────────────────────────────────────────
STATE (12 components → ~800 practical unique states)
─────────────────────────────────────────────────────────────
  0  enable_push   : attached to box?                     (2)
  1  stuck         : wall/boundary collision flag         (2)
  2  ir            : IR sensor active                     (2)
  3  fwd_near      : any forward near sonar               (2)
  4  fwd_far       : any forward far sonar                (2)
  5  direction     : box left/center/right                (3)
  6  wall_like     : total sensor bits ≥ 8 = wall         (2)
  7  is_blind      : was visible, now gone (blink)        (2)
  8  vel_hint      : centroid drift for moving box        (3)
  9  fast_grow     : sudden +4 sensor bits = wall warning (2)
  10 was_close     : IR seen in last 15 steps             (2)
  11 moving_blind  : blind + recently pursuing box        (2)

Theoretical: 2^10 × 3^2 = 9216  |  Practical: ~800

─────────────────────────────────────────────────────────────
CURRICULUM (4 stages)
─────────────────────────────────────────────────────────────
  S0: diff=0, no wall  → learn find + push (4000 eps)
  S1: diff=0, wall ON  → add wall avoidance (4000 eps)
  S2: diff=2, wall ON  → add blink handling (3000 eps)
  S3: diff=3, wall ON  → full Level 3 (8000 eps)
"""

import sys, os, pickle, csv, random
import numpy as np
from collections import defaultdict, deque, OrderedDict
import cv2

# ── Path setup ────────────────────────────────────────────────
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

# Q(λ) core
LAMBDA     = 0.70    # reduced from 0.97: cleaner credit, no 100-step blame
ALPHA_0    = 0.20
ALPHA_C    = 2.0
GAMMA      = 0.995   # 0.995^200 ≈ 0.37 → meaningful horizon
Q_INIT     = 3.0     # CRITICAL FIX: optimistic → FW explored, not avoided
TRACE_MIN  = 1e-3

# Exploration
EPS_START  = 1.0
EPS_END    = 0.04
EPS_DECAY  = 0.9992  # ~17k total eps to reach 0.04

# Behavioral controllers
BLINK_MEM   = 30     # covers max blink-off (30 steps)
VEL_WIN     = 5      # centroid drift window
ESC_TURNS   = 4      # 4 × 45° = 180° true reversal
ESC_FW      = 6      # 6 × 5px = 30px clearance from wall
POST_COOL   = 8      # FW-only cooldown steps after escape
PUSH_RECOV  = 6      # angle-adjustment steps in push+stuck recovery
PURSUIT_TTL = 18     # steps of pursuit momentum after box disappears

# Dyna-Q
N_PLAN     = 20      # imaginary steps per real step
MODEL_CAP  = 3000    # LRU model: evict oldest beyond this

# Reward shaping
FW_ALWAYS  = 0.5     # FW bonus regardless of sensors (fixes rotation bias)
SENSOR_W   = np.array([
    0.3, 0.3, 0.3, 0.3,    # left sonar far/near ×2
    0.5, 0.9, 0.5, 0.9,    # fwd far, fwd near × 4
    0.5, 0.9, 0.5, 0.9,
    0.3, 0.3, 0.3, 0.3,    # right sonar far/near ×2
    2.0,                     # IR: box right here
], dtype=np.float32)

# Output
VIDEO_EVERY = 500
VIDEO_FPS   = 15
VIDEO_DIR   = os.path.join(_HERE, "videos_exp06")


# ══════════════════════════════════════════════════════════════
# Dense Reward Wrapper  (training only — eval uses raw env)
# ══════════════════════════════════════════════════════════════
class RewardWrapper:
    """
    Removes non-Markovian one-time sensor bonuses during training.
    Replaces with dense per-step signal so Q-table sees consistent values.

    Key fix vs exp04: FW_ALWAYS bonus fires even when sensors are dark.
    This prevents Q(turn) > Q(FW) in the early exploration phase.
    """

    def __init__(self, raw_env: OBELIX):
        self._env        = raw_env
        self.enable_push = False

    def reset(self, seed=None):
        self.enable_push = False
        return self._env.reset(seed=seed)

    def step(self, action: str, render: bool = False):
        obs, reward, done = self._env.step(action, render=render)
        if reward > 50:
            self.enable_push = True

        # Dense sensor bonus every step
        dense = float(np.dot(obs[:17].astype(np.float32), SENSOR_W))

        # FW always bonus: even when obs is all-zero (dark exploration)
        # THIS IS THE KEY FIX: exp04 only gave this when dense > 0
        if action == "FW" and obs[17] == 0:
            dense += FW_ALWAYS

        return obs, reward + dense, done

    def __getattr__(self, name):
        return getattr(self._env, name)


# ══════════════════════════════════════════════════════════════
# State Construction  (MUST match exp06_agent.py exactly)
# ══════════════════════════════════════════════════════════════
def make_state(
    obs: np.ndarray,
    history: deque,
    enable_push: bool,
    last_dir: int,
    pursuit_steps: int,
) -> tuple:
    """
    Combined state from best features of all experiments.
    last_dir      : last action that moved toward box (1=L22, 2=FW, 3=R22)
    pursuit_steps : steps since last sonar contact
    """
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    left_s  = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_s = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s   = sum(int(obs[i]) for i in range(4, 12))
    total   = left_s + right_s + fwd_s

    # Box direction in sensor field
    if total == 0:
        direction = 1
    elif left_s > right_s + 1:
        direction = 0   # box on left
    elif right_s > left_s + 1:
        direction = 2   # box on right
    else:
        direction = 1   # centered

    # Wall discriminator: ≥8 bits = wide object = wall
    # FIX from exp04: threshold was 6, but box at close range also fires 6+
    # At distance: box = 2-4 bits, wall = 8-16 bits
    wall_like = int(total >= 8)

    hist      = list(history)
    any_now   = bool(np.any(obs[:17]))
    recent    = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind  = int(not any_now and recent)

    # IR memory: was box directly ahead in last 15 steps?
    was_close = int(
        any(bool(h[16] == 1) for h in hist[-15:]) if hist else False
    )

    # Velocity hint: centroid drift = box moving (from exp02)
    vel_hint = 1   # 0=drifting-left, 1=center/unknown, 2=drifting-right
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
            if drift >  0.3: vel_hint = 2
            elif drift < -0.3: vel_hint = 0

    # Fast grow: sudden +4 sensor bits in one step = wall warning (exp02)
    fast_grow = 0
    if len(hist) >= 2:
        prev = sum(int(x) for x in hist[-2][:16])
        fast_grow = int((total - prev) >= 4)

    # Moving blind (exp01 key insight): box was recently in sensor range,
    # now gone (blink), AND robot was actively pursuing it.
    # → keep moving in last direction (not oscillate!)
    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)

    return (
        int(enable_push),   # 0
        stuck,              # 1
        ir,                 # 2
        fwd_near,           # 3
        fwd_far,            # 4
        direction,          # 5
        wall_like,          # 6
        is_blind,           # 7
        vel_hint,           # 8
        fast_grow,          # 9
        was_close,          # 10
        moving_blind,       # 11
    )


# ══════════════════════════════════════════════════════════════
# Behavioral Controllers
# ══════════════════════════════════════════════════════════════
class EscapeController:
    """
    Adaptive 180° turn + sustained forward movement away from wall.

    BUG in exp01/02: [L45,L45,FW,R45,R45,FW] = net 0° turn, 5px from wall.
    FIX: 4×45° = true 180° reversal, then 6×FW = 30px clearance.
    Direction chosen by which side has more wall sensors (turn AWAY).
    """
    SEQ_LEN = ESC_TURNS + ESC_FW   # 10 steps total

    def __init__(self):
        self.active = False
        self._step  = 0
        self._turn  = 4   # 4=R45, 0=L45

    def trigger(self, obs: np.ndarray):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn  = 4 if lw >= rw else 0   # turn away from wall side
        self._step  = 0
        self.active = True

    def next_action(self) -> int:
        i = self._step
        self._step += 1
        if self._step >= self.SEQ_LEN:
            self.active = False
        return self._turn if i < ESC_TURNS else 2   # turn phase then FW phase


class PushRecovery:
    """
    KEY FIX (new in exp06): When push+stuck, do NOT trigger EscapeController.

    Why escape is wrong in push mode:
      Escape turns robot 180° → robot drags attached box AWAY from boundary.
      Result (exp02_video5): robot oscillates at boundary, making no progress.

    Correct behavior: small angle adjustment to find better push direction.
    Strategy: alternate L22/FW or R22/FW for PUSH_RECOV steps.
    Chooses turn direction toward fewer side-sensor obstacles.
    """
    def __init__(self):
        self.active = False
        self._step  = 0
        self._turn  = 1   # 1=L22, 3=R22

    def trigger(self, obs: np.ndarray):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn  = 1 if lw <= rw else 3   # turn toward less obstruction
        self._step  = 0
        self.active = True

    def next_action(self) -> int:
        i = self._step
        self._step += 1
        if self._step >= PUSH_RECOV:
            self.active = False
        # Pattern: turn, FW, turn, FW... small corrections
        return self._turn if i % 2 == 0 else 2


def intercept_action(vel_hint: int, step: int) -> int:
    """Turn slightly toward where box WILL BE (predictive, from exp02)."""
    if vel_hint == 2: return 3 if step % 4 == 0 else 2   # drifting right
    if vel_hint == 0: return 1 if step % 4 == 0 else 2   # drifting left
    return 2                                               # centered → FW


def station_action(step: int) -> int:
    """
    Gentle arc search for STATIONARY box (diff=0/2 only).
    NOT used for diff=3 moving box — use pursuit_direction instead.
    Wide oscillation: L22×2, R22×4, L22×2 (covers 90° arc).
    """
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
    """
    Dyna-Q world model with LRU eviction.

    FIX vs exp01: exp01 model grew unbounded and accumulated stale
    early-exploration transitions. Planning over those propagated wrong
    Q-values forever (Q(FW) never recovered).

    LRU eviction keeps only the MODEL_CAP most-recently-seen transitions.
    Planning thus uses fresh, post-exploration experience.
    """
    def __init__(self, cap: int = MODEL_CAP):
        self._store: OrderedDict = OrderedDict()
        self._cap   = cap

    def update(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._cap:
            self._store.popitem(last=False)   # evict oldest

    def sample(self, k: int, rng) -> list:
        if not self._store:
            return []
        keys = list(self._store.keys())
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
# Video Recording
# ══════════════════════════════════════════════════════════════
def record_episode(
    raw_env: OBELIX, Q: dict,
    ep_num: int, difficulty: int,
    seed: int, max_steps: int,
):
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
    last_dir    = 2   # last direction that detected box
    pursuit_st  = 999

    total = 0.0
    rng_v = np.random.default_rng(seed + 77_777)

    for step in range(max_steps):
        raw_env._update_frames(show=False)
        frame = cv2.flip(raw_env.frame.copy(), 0)
        if enable_push:  mode = "PUSH"
        elif esc.active: mode = "ESC"
        elif push_rec.active: mode = "PREC"
        elif cooldown > 0: mode = "COOL"
        else:            mode = "FIND"
        cv2.putText(
            frame,
            f"S:{step:4d} R:{total:7.0f} [{mode}] ep={pursuit_st}",
            (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1,
        )
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        stuck   = obs[17] == 1
        ir_on   = obs[16] == 1
        any_vis = bool(np.any(obs[:17]))
        state   = make_state(obs, history, enable_push, last_dir, pursuit_st)
        vel     = state[8]
        mb      = state[11]  # moving_blind

        # Track sensor contact for pursuit momentum
        if any(int(obs[i]) for i in range(16)):
            pursuit_st = 0
            if obs[12] or obs[13] or obs[14] or obs[15]:  last_dir = 3
            elif obs[0]  or obs[1]  or obs[2]  or obs[3]: last_dir = 1
            else:                                          last_dir = 2
        else:
            pursuit_st = min(pursuit_st + 1, 999)

        # Controllers
        if enable_push and stuck and not push_rec.active:
            push_rec.trigger(obs)
        elif not enable_push and stuck and not esc.active:
            esc.trigger(obs)
            cooldown = 0

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
        elif mb:
            action_idx = last_dir   # pursuit momentum: keep direction
        elif any_vis and vel != 1 and not enable_push:
            action_idx = intercept_action(vel, step); blink_step = 0
        elif state[7]:   # is_blind, not moving_blind
            action_idx = station_action(blink_step); blink_step += 1
        else:
            blink_step = 0
            if state in Q:
                action_idx = int(np.argmax(Q[state]))
            else:
                action_idx = int(rng_v.integers(0, N_ACTIONS))

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
    model: LRUModel,
    n_eps: int, epsilon: float, rng,
    name: str, max_steps: int,
    best_path: str, best_mean_ref: float,
    all_rwd: list, total_ep: list,
    vid_offset: int, difficulty: int,
    n_plan: int = N_PLAN,
    log_every: int = 200,
) -> tuple:
    """
    Dyna-Q(λ) with replacing traces.

    Two loops per real step:
      1. Real step: env interaction → Q(λ) update
      2. n_plan imaginary steps: sample model → Q update (no traces)
    """
    stage_rwd = []
    best_mean = best_mean_ref
    esc       = EscapeController()
    push_rec  = PushRecovery()

    for ep in range(n_eps):
        ep_seed     = int(rng.integers(0, 300_000))
        obs         = wrapped_env.reset(seed=ep_seed)
        history     = deque(maxlen=BLINK_MEM)
        history.append(obs.copy())
        enable_push = False
        cooldown    = 0
        blink_step  = 0
        last_dir    = 2      # last direction robot turned toward box
        pursuit_st  = 999    # steps since last sensor contact
        total       = 0.0
        traces: dict = {}    # (state, action_idx) → eligibility

        for step in range(max_steps):
            stuck   = obs[17] == 1
            ir_on   = obs[16] == 1
            any_vis = bool(np.any(obs[:17]))
            state   = make_state(obs, history, enable_push, last_dir, pursuit_st)
            vel     = state[8]    # vel_hint
            mb      = state[11]   # moving_blind

            # ── Track sensor contact for pursuit momentum ──────
            if any(int(obs[i]) for i in range(16)):
                pursuit_st = 0
                if obs[12] or obs[13] or obs[14] or obs[15]:
                    last_dir = 3   # box on right → remember R22
                elif obs[0]  or obs[1]  or obs[2]  or obs[3]:
                    last_dir = 1   # box on left  → remember L22
                else:
                    last_dir = 2   # box forward  → remember FW
            else:
                pursuit_st = min(pursuit_st + 1, 999)

            # ── Controller triggers ────────────────────────────
            if enable_push and stuck and not push_rec.active:
                # KEY FIX: push+stuck = push recovery NOT escape
                push_rec.trigger(obs)
                traces = {}   # reset: new mini-phase

            elif not enable_push and stuck and not esc.active:
                esc.trigger(obs)
                cooldown = 0
                traces   = {}   # reset: escape = fresh slate

            # ── Action selection ───────────────────────────────
            use_q = False

            if push_rec.active:
                action_idx = push_rec.next_action()

            elif esc.active:
                action_idx = esc.next_action()
                if not esc.active:
                    cooldown = POST_COOL

            elif cooldown > 0:
                action_idx = 2   # FW cooldown: move away from wall
                cooldown  -= 1

            elif ir_on and not enable_push:
                # IR fires: box is RIGHT THERE → FW to attach
                action_idx = 2

            elif enable_push:
                # Attached: always FW (push_rec handles stuck case above)
                action_idx = 2
                use_q      = True   # learn push-phase Q-values

            elif mb:
                # KEY FIX (exp01 insight): moving_blind → PURSUE, not oscillate
                # Box blinked but was moving → keep going in last direction
                action_idx = last_dir

            elif any_vis and vel != 1 and not enable_push:
                # Box visible, drifting → predictive intercept
                action_idx = intercept_action(vel, step)
                blink_step = 0

            elif state[7]:   # is_blind but NOT moving_blind (stationary box)
                # Box blinked, was stationary → gentle arc search
                action_idx = station_action(blink_step)
                blink_step += 1

            else:
                # ── Q-table (exploration or exploitation) ──────
                blink_step = 0
                action_idx = eps_greedy(Q[state], epsilon, rng)
                use_q      = True

            # ── Environment step ───────────────────────────────
            action = ACTIONS[action_idx]
            next_obs, reward, done = wrapped_env.step(action, render=False)
            history.append(next_obs.copy())
            total += reward

            if not enable_push and reward > 50:
                enable_push = True
                traces = {}   # new phase = clear eligibility

            # ── Q(λ) update with replacing traces ─────────────
            if use_q and not esc.active and not push_rec.active and cooldown == 0:
                next_push  = wrapped_env.enable_push
                next_st    = pursuit_st + 1 if not any(int(next_obs[i]) for i in range(16)) else 0
                next_ld    = last_dir   # will be updated next iteration
                next_state = make_state(next_obs, history, next_push, next_ld, next_st)
                best_next  = float(np.max(Q[next_state]))
                td_err     = reward + GAMMA * best_next - Q[state][action_idx]

                N[state][action_idx] += 1
                traces[(state, action_idx)] = 1.0   # REPLACING trace (not accumulating)

                for (s, a), e_val in list(traces.items()):
                    Q[s][a]     += get_alpha(int(N[s][a])) * td_err * e_val
                    new_e        = e_val * GAMMA * LAMBDA
                    if new_e < TRACE_MIN:
                        del traces[(s, a)]
                    else:
                        traces[(s, a)] = new_e

                # Phase transition: clear traces
                if next_push != enable_push:
                    traces = {}

                # ── Dyna-Q planning ────────────────────────────
                model.update((state, action_idx), (reward, next_state))
                for (ms, ma), (mr, mns) in model.sample(n_plan, rng):
                    mp_best   = float(np.max(Q[mns]))
                    Q[ms][ma] += get_alpha(int(N[ms][ma])) * (
                        mr + GAMMA * mp_best - Q[ms][ma]
                    )

            obs = next_obs
            if done:
                break

        # ── End of episode ─────────────────────────────────────
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        stage_rwd.append(total)
        all_rwd.append(total)
        total_ep[0] += 1

        if (ep + 1) % log_every == 0:
            m_rwd = np.mean(stage_rwd[-log_every:])
            prev  = (stage_rwd[-(2 * log_every):-log_every]
                     if len(stage_rwd) > log_every else [m_rwd - 1])
            trend = "✅" if m_rwd > np.mean(prev) else "⏳"
            succ  = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            min_r = min(stage_rwd[-log_every:])
            print(
                f"[{name} Ep {ep+1:>5}/{n_eps}] "
                f"Mean: {m_rwd:>9.1f} | Min: {min_r:>9.0f} | "
                f"ε:{epsilon:.4f} | Succ/{log_every}:{succ:3d} | "
                f"States:{len(Q):5d} | Model:{len(model):4d}  {trend}"
            )
            if m_rwd > best_mean:
                best_mean = m_rwd
                save_Q(Q, N, best_path)

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
# Main Training
# ══════════════════════════════════════════════════════════════
def train(
    eps0      = 4000,   # Stage 0: diff=0, no wall
    eps1      = 4000,   # Stage 1: diff=0, wall ON
    eps2      = 3000,   # Stage 2: diff=2, blinking
    eps3      = 8000,   # Stage 3: diff=3, full Level 3
    max_steps = 1000,
    seed      = 42,
    prefix    = "q_exp06",
    box_speed = 2,
    n_plan    = N_PLAN,
):
    best_path  = os.path.join(_HERE, f"{prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{prefix}_final.pkl")
    log_path   = os.path.join(_HERE, f"{prefix}_rewards.csv")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Single unified Q-table (not split by module — enable_push in state handles it)
    Q = defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
    N = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))
    model = LRUModel(cap=MODEL_CAP)

    epsilon   = EPS_START
    all_rwd   = []
    best_mean = -np.inf
    total_ep  = [0]

    def make_env(wall: bool, diff: int, spd: int = 0):
        raw = OBELIX(
            scaling_factor=5, arena_size=500,
            max_steps=max_steps, wall_obstacles=wall,
            difficulty=diff, box_speed=spd, seed=seed,
        )
        return RewardWrapper(raw), raw

    # ── Stage 0: No wall, static box — learn basic find + push ──
    print("\n" + "═" * 65)
    print(f" STAGE 0 | diff=0 | NO WALL | {eps0} eps")
    print(" Goal: locate box, approach, attach, push to boundary")
    print("═" * 65)
    w0, r0 = make_env(wall=False, diff=0)
    epsilon, best_mean = run_stage(
        w0, r0, Q, N, model, eps0, epsilon, rng,
        "S0", max_steps, best_path, best_mean,
        all_rwd, total_ep, 0, 0, n_plan,
    )

    # ── Stage 1: Wall ON — learn avoidance + gap navigation ──────
    print("\n" + "═" * 65)
    print(f" STAGE 1 | diff=0 | WALL ON | {eps1} eps")
    print(" Goal: avoid -200 hits, navigate through wall gap")
    print("═" * 65)
    epsilon = max(0.40, epsilon)   # partial reset: re-explore with wall
    w1, r1 = make_env(wall=True, diff=0)
    epsilon, best_mean = run_stage(
        w1, r1, Q, N, model, eps1, epsilon, rng,
        "S1", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0, 0, n_plan,
    )

    # ── Stage 2: Blinking box — learn blink handling ──────────────
    print("\n" + "═" * 65)
    print(f" STAGE 2 | diff=2 | BLINKING | {eps2} eps")
    print(" Goal: station-keep during blink, resume when visible")
    print("═" * 65)
    epsilon = max(0.30, epsilon)
    w2, r2 = make_env(wall=True, diff=2)
    epsilon, best_mean = run_stage(
        w2, r2, Q, N, model, eps2, epsilon, rng,
        "S2", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1, 2, n_plan,
    )

    # ── Stage 3: Full Level 3 — moving + blinking + wall ──────────
    print("\n" + "═" * 65)
    print(f" STAGE 3 | diff=3 | MOVING+BLINKING+WALL | {eps3} eps")
    print(" Goal: intercept moving box, handle all difficulty knobs")
    print("═" * 65)
    epsilon = max(0.25, epsilon)
    w3, r3 = make_env(wall=True, diff=3, spd=box_speed)
    epsilon, best_mean = run_stage(
        w3, r3, Q, N, model, eps3, epsilon, rng,
        "S3", max_steps, best_path, best_mean,
        all_rwd, total_ep, eps0 + eps1 + eps2, 3, n_plan,
    )

    # ── Final save ─────────────────────────────────────────────────
    save_Q(Q, N, final_path)
    log_csv(log_path, all_rwd)

    total_eps  = total_ep[0]
    succ_total = sum(1 for r in all_rwd if r > 500)
    last_200   = np.mean(all_rwd[-200:]) if len(all_rwd) >= 200 else np.mean(all_rwd)

    print(f"\n{'═' * 65}")
    print(f" TRAINING COMPLETE")
    print(f"{'═' * 65}")
    print(f"  Best 200-ep mean : {best_mean:.1f}")
    print(f"  Last 200-ep mean : {last_200:.1f}")
    print(f"  Success rate     : {succ_total}/{total_eps} ({succ_total/total_eps*100:.1f}%)")
    print(f"  Q-table states   : {len(Q)}")
    print(f"  Model transitions: {len(model)}")
    print(f"  Best weights     : {os.path.basename(best_path)}")
    print(f"  Rewards log      : {log_path}")
    print(f"  Videos           : {VIDEO_DIR}/")

    print("\n🎬 Final evaluation videos...")
    for diff, raw, vid_diff in [(0, r0, 0), (3, r3, 3)]:
        for s in range(3):
            record_episode(raw, Q, 99_000 + s + diff * 10,
                           vid_diff, s * 137, max_steps)

    return all_rwd


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OBELIX EXP06 — Dyna-Q(λ) Unified")
    p.add_argument("--eps0",      type=int, default=4000)
    p.add_argument("--eps1",      type=int, default=4000)
    p.add_argument("--eps2",      type=int, default=3000)
    p.add_argument("--eps3",      type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--prefix",    type=str, default="q_exp06")
    p.add_argument("--box_speed", type=int, default=2)
    p.add_argument("--n_plan",    type=int, default=20,
                   help="Dyna-Q planning steps per real step")
    args = p.parse_args()
    train(**vars(args))