"""
OBELIX Level-3 Agent
=====================
Matches train_l3.py exactly:
  - 10-bit state (adds moving_blind)
  - No IR probe (for moving box, probe loses contact)
  - Pursuit momentum: keep moving 15 steps after box blinks
  - Fast attach: IR=1 → immediately go to push mode
  - Clean episode reset via ep_steps counter
Submit: agent.py + q_table_l3.pkl
"""

import os, pickle
import numpy as np
from collections import deque

ACTIONS      = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS    = 5
Q_INIT       = 5.0
BLINK_MEM    = 30
_ESCAPE_SEQ  = [0, 0, 0, 0, 2, 2]   # L45×4=180°, FW×2

_Q           = None
_history     = deque(maxlen=BLINK_MEM)
_enable_push = False
_ir_streak   = 0
_esc_step    = 0
_ep_steps    = 0
_last_dir    = 2
_pursuit_steps = 0    # steps since last sonar sighting


def _load():
    global _Q
    if _Q is not None: return
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "q_table_l3.pkl"), "rb") as f:
        data = pickle.load(f)
    _Q = {
        "finder": {eval(k): np.array(v) for k, v in data["Q_finder"].items()},
        "pusher": {eval(k): np.array(v) for k, v in data["Q_pusher"].items()},
        "escape": {eval(k): np.array(v) for k, v in data.get("Q_escape", {}).items()},
    }


def _make_state(obs, recent_saw, was_close, is_blind, moving_blind):
    return (
        int(np.any(obs[[5, 7, 9, 11]])),
        int(np.any(obs[[4, 6, 8, 10]])),
        int(np.any(obs[0:4])),
        int(np.any(obs[12:16])),
        int(obs[16]), int(obs[17]),
        int(recent_saw), int(was_close), int(is_blind),
        int(moving_blind),
    )


def _get_q(module, state):
    if state in _Q[module]: return _Q[module][state]
    d = np.full(N_ACTIONS, Q_INIT, dtype=np.float64)
    d[2] = Q_INIT + 5.0
    return d


def _reset():
    global _history, _enable_push, _ir_streak, _esc_step
    global _ep_steps, _last_dir, _pursuit_steps
    _history.clear()
    _enable_push   = False
    _ir_streak     = 0
    _esc_step      = 0
    _ep_steps      = 0
    _last_dir      = 2
    _pursuit_steps = 0


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _enable_push, _ir_streak, _esc_step, _ep_steps
    global _last_dir, _pursuit_steps

    _load()

    # ── Episode boundary detection ────────────────────────────────────────────
    _ep_steps += 1
    if _ep_steps > 1000:
        _reset(); _ep_steps = 1
    # Reset if push mode but IR gone for too long (new episode started silently)
    if _enable_push and _ep_steps > 50:
        if not any(bool(h[16]) for h in _history) and not obs[16]:
            _reset(); _ep_steps = 1

    # ── IR streak → attachment ────────────────────────────────────────────────
    if obs[16] and not obs[17]:
        _ir_streak += 1
    else:
        _ir_streak = 0
    if not _enable_push and _ir_streak >= 2:   # Level 3: faster attach (2 not 3)
        _enable_push = True

    # ── Last known box direction + pursuit counter ────────────────────────────
    if   any(obs[0:4]):    _last_dir = 1; _pursuit_steps = 0
    elif any(obs[4:12]):   _last_dir = 2; _pursuit_steps = 0
    elif any(obs[12:16]):  _last_dir = 3; _pursuit_steps = 0
    else:
        _pursuit_steps += 1

    _history.append(obs.copy())

    any_sens     = bool(np.any(obs[:17]))
    recent_saw   = any(bool(np.any(h[:17])) for h in _history)
    was_close    = any(bool(h[16] == 1)     for h in _history)
    is_blind     = (not any_sens) and recent_saw
    moving_blind = int(is_blind and _pursuit_steps <= 15)
    stuck        = bool(obs[17])

    # ── Priority 1: ESCAPE ────────────────────────────────────────────────────
    if stuck:
        a = ACTIONS[_ESCAPE_SEQ[_esc_step % len(_ESCAPE_SEQ)]]
        _esc_step += 1
        return a
    _esc_step = 0

    # ── Priority 2: PURSUIT (blind but recently saw box) ─────────────────────
    # Level 3 key: box is moving, so keep going in last known direction
    # Don't oscillate, don't probe — just pursue
    if not _enable_push and is_blind and was_close:
        return ACTIONS[_last_dir]   # directional pursuit, not L22/R22 oscillation

    # ── Priority 3: Q-table ───────────────────────────────────────────────────
    module = "pusher" if _enable_push else "finder"
    state  = _make_state(obs, recent_saw, was_close, is_blind, moving_blind)
    return ACTIONS[int(np.argmax(_get_q(module, state)))]