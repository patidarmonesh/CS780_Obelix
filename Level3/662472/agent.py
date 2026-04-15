"""
agent_v2.py — OBELIX Level 3 Submission  (exp02 / v2)
=======================================================
Load-once Q-table, episode-stateful policy.

Submit:
    zip submission.zip agent_v2.py q_table_v2_best.pkl
    Upload to Codabench.

Requirements:
    - No GPU needed — pure NumPy
    - make_state() MUST match train_v2.py exactly
    - Global state auto-resets on episode boundary
"""

import os
import ast
import pickle
import numpy as np
from collections import deque
from typing import Sequence

# ── Constants ─────────────────────────────────────────────────
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS     = 5
WEIGHTS_FILE  = "q_table_v2_best.pkl"   # change if using different prefix

BLINK_MEM     = 25
VEL_WIN       = 5
ESC_SEQ_LEN   = 10
POST_ESC_COOL = 8
MAX_EP_STEPS  = 1000     # Codabench eval max steps


# ══════════════════════════════════════════════════════════════
# Q-table (loaded once, then reused)
# ══════════════════════════════════════════════════════════════
_Q: dict = None   # type: ignore


def _load():
    """Load Q-table from pkl once. Prints summary on success."""
    global _Q
    if _Q is not None:
        return

    # Look for weights file next to this script
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE),
        WEIGHTS_FILE,  # cwd fallback
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)

    if path is None:
        print(f"[agent_v2] WARNING: {WEIGHTS_FILE} not found. "
              "Running rule-only fallback policy.")
        _Q = {}
        return

    with open(path, "rb") as f:
        raw = pickle.load(f)

    _Q = {
        ast.literal_eval(k): np.array(v)
        for k, v in raw["Q"].items()
    }
    print(f"[agent_v2] Loaded {len(_Q)} states from {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════
# Episode State  (global, auto-resets at episode boundary)
# ══════════════════════════════════════════════════════════════
_history         : deque = None    # type: ignore
_enable_push     = False    # attached to box?
_esc_active      = False    # currently executing escape?
_esc_step        = 0        # step within current escape sequence
_esc_turn        = 4        # action idx for turn phase (0=L45, 4=R45)
_cooldown        = 0        # post-escape FW cooldown counter
_blink_step      = 0        # station-keep oscillation step
_ep_step         = 0        # steps since episode start
_prev_obs_sum    = -1       # for episode boundary detection
_fwd_streak      = 0        # consecutive fwd-near-active steps (attach detector)
_last_action_idx = 2        # last action taken (for attach heuristic)


def _reset():
    """Full episode state reset."""
    global _history, _enable_push, _esc_active, _esc_step, _esc_turn
    global _cooldown, _blink_step, _ep_step, _prev_obs_sum
    global _fwd_streak, _last_action_idx

    _history         = deque(maxlen=BLINK_MEM)
    _enable_push     = False
    _esc_active      = False
    _esc_step        = 0
    _esc_turn        = 4
    _cooldown        = 0
    _blink_step      = 0
    _ep_step         = 0
    _prev_obs_sum    = -1
    _fwd_streak      = 0
    _last_action_idx = 2


def _check_reset(obs: np.ndarray) -> bool:
    """
    Heuristic episode boundary detection.
    Codabench doesn't pass done flag to policy().
    Detects: step limit reached OR sudden state jump from push→zero.
    """
    global _ep_step, _prev_obs_sum, _enable_push

    _ep_step += 1
    curr_sum = int(np.sum(obs))

    boundary = False
    if _ep_step >= MAX_EP_STEPS:
        boundary = True
    elif (_ep_step > 10
          and _prev_obs_sum > 3
          and curr_sum == 0
          and _enable_push):
        # Was in push state (box always visible), now all-zero → new episode
        boundary = True

    _prev_obs_sum = curr_sum
    return boundary


# ══════════════════════════════════════════════════════════════
# State Construction  (MUST match train_v2.py exactly)
# ══════════════════════════════════════════════════════════════
def make_state(obs: np.ndarray, history: deque, enable_push: bool) -> tuple:
    """
    Temporal state for Q-table. Identical to train_v2.py.
    Any mismatch = Q-table keys never hit = random fallback policy.
    """
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    left_s  = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_s = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s   = sum(int(obs[i]) for i in range(4, 12))
    total   = left_s + right_s + fwd_s

    if total == 0:
        direction = 1
    elif left_s > right_s + 1:
        direction = 0
    elif right_s > left_s + 1:
        direction = 2
    else:
        direction = 1

    wall_like = int(total >= 6)

    hist     = list(history)
    any_now  = bool(np.any(obs[:17]))
    recent   = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind = int(not any_now and recent)

    vel_hint = 1
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6])  + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0:
                centroids.append((r - l) / t)
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift >  0.3:
                vel_hint = 2
            elif drift < -0.3:
                vel_hint = 0

    return (int(enable_push), stuck, ir, fwd_near, fwd_far,
            direction, wall_like, is_blind, vel_hint)


# ══════════════════════════════════════════════════════════════
# Deterministic Behaviors (identical to train_v2.py)
# ══════════════════════════════════════════════════════════════
def _trigger_escape(obs: np.ndarray):
    """Adaptive 180° escape: turn AWAY from wall sensors."""
    global _esc_active, _esc_step, _esc_turn, _cooldown

    left_w  = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_w = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    _esc_turn   = 4 if left_w >= right_w else 0   # R45 if wall-left, L45 if wall-right
    _esc_step   = 0
    _esc_active = True
    _cooldown   = 0


def _escape_next() -> int:
    """Returns next action index in escape sequence."""
    global _esc_step, _esc_active, _cooldown

    i = _esc_step
    _esc_step += 1
    if _esc_step >= ESC_SEQ_LEN:
        _esc_active = False
        _cooldown   = POST_ESC_COOL

    return _esc_turn if i < 4 else 2   # turn×4 then FW×6


def _station_action(step: int) -> int:
    """Oscillation during blink blackout."""
    pattern = [1, 1, 3, 3, 3, 3, 1, 1]
    return pattern[step % len(pattern)]


def _intercept_action(vel_hint: int, step: int) -> int:
    """Predictive intercept for moving box."""
    if vel_hint == 2:
        return 3 if step % 4 == 0 else 2
    elif vel_hint == 0:
        return 1 if step % 4 == 0 else 2
    return 2


# ══════════════════════════════════════════════════════════════
# Attach Detection  (no reward available at inference time)
# ══════════════════════════════════════════════════════════════
def _update_push_state(obs: np.ndarray, last_action: int, stuck: bool):
    """
    Infer attachment from sensor patterns.
    Once attached, box is always right in front → fwd_near stays active.

    Two signals:
      1. fwd_near streak: box close for 3+ consecutive FW steps = attached
      2. IR fired right after FW action = just crossed attachment threshold
    """
    global _enable_push, _fwd_streak

    if _enable_push:
        return   # already attached, don't change

    fwd_near = int(obs[4] or obs[6] or obs[8] or obs[10])

    if fwd_near and not stuck:
        _fwd_streak += 1
        # Also accelerate if IR fired after FW (very strong signal)
        if last_action == 2 and obs[16] == 1:
            _fwd_streak += 2
    elif not stuck:
        _fwd_streak = max(0, _fwd_streak - 1)   # gradual decay
    else:
        _fwd_streak = 0   # stuck = not successfully pushing

    if _fwd_streak >= 4:
        _enable_push = True


# ══════════════════════════════════════════════════════════════
# Policy  (called by evaluator at every step)
# ══════════════════════════════════════════════════════════════
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Main entry point called by Codabench evaluator.

    Args:
        obs : np.ndarray shape (18,), binary 0/1
        rng : np.random.Generator (from evaluator, used for exploration)

    Returns:
        One of: 'L45', 'L22', 'FW', 'R22', 'R45'

    Episode state (history, push flag, escape counters) maintained
    globally. Auto-resets when episode boundary detected.
    """
    global _history, _enable_push, _esc_active, _cooldown
    global _blink_step, _last_action_idx

    # ── Load weights on first call ────────────────────────────
    _load()

    # ── Episode boundary check ────────────────────────────────
    if _history is None or _check_reset(obs):
        _reset()

    # ── Update attachment state ───────────────────────────────
    stuck   = obs[17] == 1
    _update_push_state(obs, _last_action_idx, stuck)
    _history.append(obs.copy())

    ir_on   = obs[16] == 1
    any_vis = bool(np.any(obs[:17]))
    recent  = any(bool(np.any(h[:17])) for h in _history)
    blind   = not any_vis and recent
    state   = make_state(obs, _history, _enable_push)
    vel     = state[8]   # vel_hint: 0=left, 1=center, 2=right

    # ══ Priority 1: Wall stuck → trigger escape ════════════════
    if stuck and not _esc_active:
        _trigger_escape(obs)
        _fwd_streak = 0   # can't be attaching if stuck

    if _esc_active:
        action_idx       = _escape_next()
        _last_action_idx = action_idx
        return ACTIONS[action_idx]

    # ══ Priority 2: Post-escape cooldown (move away from wall) ═
    if _cooldown > 0:
        _cooldown       -= 1
        _last_action_idx = 2
        return "FW"

    # ══ Priority 3: IR sensor fires → box right here → attach ══
    if ir_on and not _enable_push:
        _last_action_idx = 2
        return "FW"

    # ══ Priority 4: Attached → push to boundary ════════════════
    if _enable_push:
        # Push state: always forward (escape handles stuck case)
        _last_action_idx = 2
        return "FW"

    # ══ Priority 5: Blink blackout → station keep ══════════════
    if blind and recent:
        action_idx       = _station_action(_blink_step)
        _blink_step     += 1
        _last_action_idx = action_idx
        return ACTIONS[action_idx]

    _blink_step = 0

    # ══ Priority 6: Moving box detected → intercept ════════════
    if any_vis and vel != 1 and not _enable_push:
        action_idx       = _intercept_action(vel, _ep_step)
        _last_action_idx = action_idx
        return ACTIONS[action_idx]

    # ══ Priority 7: Q-table ════════════════════════════════════
    if state in _Q:
        action_idx       = int(np.argmax(_Q[state]))
        _last_action_idx = action_idx
        return ACTIONS[action_idx]

    # ══ Fallback: reactive heuristic (if Q-table empty/cold) ═══
    # Forward sensors: box in front → go straight
    if obs[4] or obs[6] or obs[8] or obs[10]:
        _last_action_idx = 2
        return "FW"
    if obs[5] or obs[7] or obs[9] or obs[11]:
        _last_action_idx = 2
        return "FW"

    # Side sensors: turn toward box
    left_s  = sum(int(obs[i]) for i in range(0, 4))
    right_s = sum(int(obs[i]) for i in range(12, 16))
    if left_s > right_s:
        _last_action_idx = 1
        return "L22"
    if right_s > left_s:
        _last_action_idx = 3
        return "R22"

    # Systematic arc search: broad sweep to find distant box
    # Period 60: go forward 50 steps then turn 22° (covers full arc eventually)
    if _ep_step % 60 == 0:
        _last_action_idx = 1
        return "L22"
    if _ep_step % 60 == 30:
        _last_action_idx = 3
        return "R22"

    _last_action_idx = 2
    return "FW"