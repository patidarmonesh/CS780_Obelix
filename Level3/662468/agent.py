"""
agent.py — experiments/Level_3/exp01
======================================
Submission file — works locally AND on Codabench.

Zip contents:
    agent.py
    q_table_best.pkl

Submit:
    cd experiments/Level_3/exp01
    zip submission.zip agent.py q_table_best.pkl
    Upload to Codabench.

Notes:
    - No GPU required — pure NumPy
    - Temporal state maintained via global deque
    - Episode boundary auto-detected (step counter reset)
    - Hardcoded overrides: escape + IR attach + station keep
"""

import os
import pickle
import ast
import numpy as np
from collections import defaultdict, deque
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS    = 5
WEIGHTS_FILE = "q_table_best.pkl"

BLINK_MEMORY = 30
VEL_WINDOW   = 6
MAX_EP_STEPS = 1000   # Codabench eval max steps


# ─────────────────────────────────────────────────────────────
# Global episode state
# ─────────────────────────────────────────────────────────────
_Q           = None        # loaded Q-tables
_history     = None        # obs deque
_enable_push = False       # attached to box?
_esc_step    = 0           # escape sequence step
_blink_step  = 0           # station-keep step
_step_in_ep  = 0           # steps since episode start
_prev_sum    = -1          # for episode boundary detection


def _load():
    global _Q
    if _Q is not None:
        return

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE
    )

    _Q = {m: {} for m in ("finder", "pusher", "escape")}

    if not os.path.exists(path):
        # No weights — hardcoded rules still work
        print(f"[agent] WARNING: {WEIGHTS_FILE} not found. "
              f"Running with rules only.")
        return

    with open(path, "rb") as f:
        raw = pickle.load(f)

    for m in ("finder", "pusher", "escape"):
        key = f"Q_{m}"
        if key in raw:
            _Q[m] = {
                ast.literal_eval(k): np.array(v)
                for k, v in raw[key].items()
            }

    print(f"[agent] Loaded | "
          f"F:{len(_Q['finder'])} P:{len(_Q['pusher'])} "
          f"E:{len(_Q['escape'])}")


def _reset_episode():
    global _history, _enable_push, _esc_step, _blink_step, _step_in_ep
    _history     = deque(maxlen=BLINK_MEMORY)
    _enable_push = False
    _esc_step    = 0
    _blink_step  = 0
    _step_in_ep  = 0


def _detect_boundary(obs):
    """
    Detect episode reset heuristically.
    Codabench doesn't pass done flag to policy().
    """
    global _step_in_ep, _prev_sum
    _step_in_ep += 1
    curr_sum     = int(np.sum(obs))
    boundary     = False

    if _step_in_ep >= MAX_EP_STEPS:
        boundary = True
    elif (_step_in_ep > 10
          and _prev_sum >= 3
          and curr_sum == 0
          and _enable_push):
        # Was in push state, now all zeros → likely new episode
        boundary = True

    _prev_sum = curr_sum
    return boundary


# ─────────────────────────────────────────────────────────────
# State construction — MUST match train.py exactly
# ─────────────────────────────────────────────────────────────
def sensor_centroid(obs):
    left_fwd  = int(obs[4]) + int(obs[5]) + int(obs[6])  + int(obs[7])
    right_fwd = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    total     = left_fwd + right_fwd
    if total == 0:
        return None
    if left_fwd > right_fwd + 1:
        return -1
    if right_fwd > left_fwd + 1:
        return +1
    return 0


def make_state(obs, history):
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])

    count_fwd  = sum(int(obs[i]) for i in range(4, 12))
    count_side = (sum(int(obs[i]) for i in range(0, 4)) +
                  sum(int(obs[i]) for i in range(12, 16)))
    spread_high = int(count_side > 2 and count_fwd >= 2)
    sides_any   = int(count_side > 0)

    hist_list  = list(history)
    n          = len(hist_list)
    recent_saw = int(any(bool(np.any(h[:17])) for h in hist_list))
    was_close  = int(any(bool(h[16] == 1)     for h in hist_list))
    any_now    = bool(np.any(obs[:17]))
    is_blind   = int(not any_now and recent_saw)

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
            if drift > 0.4:
                vel_hint = +1
            elif drift < -0.4:
                vel_hint = -1

    fast_grow = 0
    if n >= 2:
        prev_count = sum(int(x) for x in hist_list[-2][:16])
        curr_count = sum(int(x) for x in obs[:16])
        fast_grow  = int((curr_count - prev_count) >= 4)

    return (stuck, ir, fwd_near, fwd_far,
            sides_any, spread_high, is_blind, was_close,
            vel_hint + 1, fast_grow)


# ─────────────────────────────────────────────────────────────
# Deterministic behaviors
# ─────────────────────────────────────────────────────────────
_ESCAPE_SEQ = [0, 0, 2, 4, 4, 2]

def escape_action(step):
    return _ESCAPE_SEQ[step % len(_ESCAPE_SEQ)]

def station_action(step):
    return 1 if step % 2 == 0 else 3

def intercept_action(vel_hint, step):
    if vel_hint == +1:
        return 3 if step % 3 == 0 else 2
    elif vel_hint == -1:
        return 1 if step % 3 == 0 else 2
    return 2


# ─────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Called by evaluator at every step.

    Episode state (history, push flag, counters) maintained
    globally — auto-resets on episode boundary.

    Args:
        obs : np.ndarray shape (18,), binary 0/1
        rng : np.random.Generator (from evaluator)

    Returns:
        action string: one of 'L45','L22','FW','R22','R45'
    """
    global _history, _enable_push, _esc_step, _blink_step

    _load()

    # Episode boundary check
    if _history is None or _detect_boundary(obs):
        _reset_episode()

    _history.append(obs.copy())

    stuck   = obs[17] == 1
    ir_on   = obs[16] == 1 and not stuck
    any_vis = bool(np.any(obs[:17]))
    recent  = any(bool(np.any(h[:17])) for h in _history)
    blind   = not any_vis and recent
    state   = make_state(obs, _history)
    vel     = state[8] - 1   # -1/0/+1

    module = "escape" if stuck else ("pusher" if _enable_push else "finder")

    # ── Priority 1: Wall stuck → escape ──────────────────────
    if stuck:
        action_idx  = escape_action(_esc_step)
        _esc_step  += 1
        _blink_step = 0
        return ACTIONS[action_idx]

    _esc_step = 0

    # ── Priority 2: IR + not stuck → BOX → attach ────────────
    if ir_on and not _enable_push:
        _blink_step = 0
        return "FW"

    # ── Priority 3: Attached → push to boundary ──────────────
    if _enable_push:
        _blink_step = 0
        # Check if just got reward > 50 (attach signal)
        return "FW"

    # ── Priority 4: Blind (blink) → station keep ─────────────
    if blind and recent:
        a = station_action(_blink_step)
        _blink_step += 1
        return ACTIONS[a]

    _blink_step = 0

    # ── Priority 5: Box moving → intercept ───────────────────
    if any_vis and vel != 0:
        return ACTIONS[intercept_action(vel, _step_in_ep)]

    # ── Priority 6: Q-table ───────────────────────────────────
    Q_mod = _Q.get(module, {})
    if state in Q_mod:
        return ACTIONS[int(np.argmax(Q_mod[state]))]

    # ── Fallback: reactive heuristic ─────────────────────────
    if obs[4] or obs[6] or obs[8] or obs[10]:  return "FW"
    if obs[5] or obs[7] or obs[9] or obs[11]:  return "FW"
    left  = sum(int(obs[i]) for i in range(0, 4))
    right = sum(int(obs[i]) for i in range(12, 16))
    if left > right:   return "L22"
    if right > left:   return "R22"

    # Systematic arc search
    if _step_in_ep % 40 == 0:
        return "L22"
    return "FW"