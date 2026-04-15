import os
import pickle
import numpy as np
from collections import deque


ACTIONS      = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS    = len(ACTIONS)
Q_INIT       = 30.0
BLINK_MEMORY = 20


_Q           = None
_history     = deque(maxlen=BLINK_MEMORY)
_enable_push = False
_ir_consec   = 0
_esc_step    = 0
_blk_step    = 0
_ep_steps    = 0



def _load():
    global _Q
    if _Q is not None:
        return
    base = os.path.dirname(__file__)
    path = os.path.join(base, "q_table_l2.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    _Q = {
        "finder": {eval(k): np.array(v) for k, v in data["Q_finder"].items()},
        "pusher": {eval(k): np.array(v) for k, v in data["Q_pusher"].items()},
        "escape": {eval(k): np.array(v) for k, v in data.get("Q_escape", {}).items()},
    }



def _make_state(obs, recent_saw, was_close, is_blind):
    return tuple(obs.astype(int).tolist()) + (int(recent_saw), int(was_close), int(is_blind))



def _get_q(module, state):
    if state in _Q[module]:
        return _Q[module][state]
    return np.full(N_ACTIONS, Q_INIT)



_ESCAPE_SEQ = [0, 0, 2, 0, 0, 2]


def _escape_action(step):
    return ACTIONS[_ESCAPE_SEQ[step % len(_ESCAPE_SEQ)]]


def _station_action(step):
    return ACTIONS[1] if (step % 2 == 0) else ACTIONS[3]



def _soft_reset():
    global _history, _enable_push, _ir_consec, _esc_step, _blk_step, _ep_steps
    _history.clear()
    _enable_push = False
    _ir_consec   = 0
    _esc_step    = 0
    _blk_step    = 0
    _ep_steps    = 0



def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _enable_push, _ir_consec, _esc_step, _blk_step, _ep_steps

    _load()

    _ep_steps += 1
    if _ep_steps > 1000:
        _soft_reset()
        _ep_steps = 1

    if obs[16] == 1:
        _ir_consec += 1
    else:
        _ir_consec = 0

    if not _enable_push and _ir_consec >= 3:
        _enable_push = True

    if _enable_push and obs[16] == 0 and _ir_consec == 0:
        _soft_reset()
        _ep_steps = 1

    _history.append(obs.copy())

    any_sens   = bool(np.any(obs[:17]))
    recent_saw = any(bool(np.any(h[:17])) for h in _history)
    was_close  = any(bool(h[16] == 1)     for h in _history)
    is_blind   = (not any_sens) and recent_saw

    stuck  = bool(obs[17] == 1)
    module = "escape" if stuck else ("pusher" if _enable_push else "finder")

    if stuck:
        action = _escape_action(_esc_step)
        _esc_step += 1
        _blk_step  = 0
        return action
    else:
        _esc_step = 0

    if not _enable_push and is_blind and was_close:
        action = _station_action(_blk_step)
        _blk_step += 1
        return action
    else:
        _blk_step = 0

    state  = _make_state(obs, recent_saw, was_close, is_blind)
    q_vals = _get_q(module, state)
    return ACTIONS[int(np.argmax(q_vals))]
