import os, ast, pickle
import numpy as np
from collections import deque
from typing import Sequence


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS    = 5
WEIGHTS_FILE = "q_exp06_best.pkl"


BLINK_MEM   = 30
VEL_WIN     = 5
ESC_TURNS   = 4
ESC_FW      = 6
POST_COOL   = 8
PUSH_RECOV  = 6
PURSUIT_TTL = 18
MAX_EP_ST   = 1000


_Q: dict = None


def _load():
    global _Q
    if _Q is not None: return
    for path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE),
        WEIGHTS_FILE,
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = pickle.load(f)
            _Q = {ast.literal_eval(k): np.array(v) for k, v in raw["Q"].items()}
            print(f"[exp06_agent] Loaded {len(_Q)} states")
            return
    print(f"[exp06_agent] WARNING: {WEIGHTS_FILE} not found — rule-only fallback")
    _Q = {}


_history     = None
_enable_push = False
_esc_active  = False
_esc_step    = 0
_esc_turn    = 4
_prec_active = False
_prec_step   = 0
_prec_turn   = 1
_cooldown    = 0
_blink_step  = 0
_ep_step     = 0
_prev_sum    = -1
_last_dir    = 2
_pursuit_st  = 999


def _reset():
    global _history, _enable_push, _esc_active, _esc_step, _esc_turn
    global _prec_active, _prec_step, _prec_turn
    global _cooldown, _blink_step, _ep_step, _prev_sum, _last_dir, _pursuit_st
    _history     = deque(maxlen=BLINK_MEM)
    _enable_push = False
    _esc_active  = False
    _esc_step    = 0
    _esc_turn    = 4
    _prec_active = False
    _prec_step   = 0
    _prec_turn   = 1
    _cooldown    = 0
    _blink_step  = 0
    _ep_step     = 0
    _prev_sum    = -1
    _last_dir    = 2
    _pursuit_st  = 999


def _check_reset(obs: np.ndarray) -> bool:
    global _ep_step, _prev_sum, _enable_push
    _ep_step += 1
    cs = int(np.sum(obs))
    boundary = (_ep_step >= MAX_EP_ST or
                (_ep_step > 10 and _prev_sum > 3 and cs == 0 and _enable_push))
    _prev_sum = cs
    return boundary


def make_state(obs, history, enable_push, last_dir, pursuit_steps):
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])
    left_s   = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_s  = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s    = sum(int(obs[i]) for i in range(4, 12))
    total    = left_s + right_s + fwd_s
    if total == 0:              direction = 1
    elif left_s > right_s + 1: direction = 0
    elif right_s > left_s + 1: direction = 2
    else:                       direction = 1
    wall_like = int(total >= 8)
    hist      = list(history)
    any_now   = bool(np.any(obs[:17]))
    recent    = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind  = int(not any_now and recent)
    was_close = int(any(bool(h[16] == 1) for h in hist[-15:]) if hist else False)
    vel_hint  = 1
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6])  + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0: centroids.append((r - l) / t)
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift >  0.3: vel_hint = 2
            elif drift < -0.3: vel_hint = 0
    fast_grow = 0
    if len(hist) >= 2:
        prev = sum(int(x) for x in hist[-2][:16])
        fast_grow = int((total - prev) >= 4)
    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)
    return (int(enable_push), stuck, ir, fwd_near, fwd_far,
            direction, wall_like, is_blind, vel_hint,
            fast_grow, was_close, moving_blind)


def _esc_trigger(obs):
    global _esc_active, _esc_step, _esc_turn, _cooldown
    lw = sum(int(obs[i]) for i in range(0, 4))
    rw = sum(int(obs[i]) for i in range(12, 16))
    _esc_turn   = 4 if lw >= rw else 0
    _esc_step   = 0
    _esc_active = True
    _cooldown   = 0


def _esc_next() -> int:
    global _esc_step, _esc_active, _cooldown
    i = _esc_step; _esc_step += 1
    if _esc_step >= ESC_TURNS + ESC_FW:
        _esc_active = False; _cooldown = POST_COOL
    return _esc_turn if i < ESC_TURNS else 2


def _prec_trigger(obs):
    global _prec_active, _prec_step, _prec_turn
    lw = sum(int(obs[i]) for i in range(0, 4))
    rw = sum(int(obs[i]) for i in range(12, 16))
    _prec_turn   = 1 if lw <= rw else 3
    _prec_step   = 0
    _prec_active = True


def _prec_next() -> int:
    global _prec_step, _prec_active
    i = _prec_step; _prec_step += 1
    if _prec_step >= PUSH_RECOV:
        _prec_active = False
    return _prec_turn if i % 2 == 0 else 2


def _intercept(vel_hint, step):
    if vel_hint == 2: return 3 if step % 4 == 0 else 2
    if vel_hint == 0: return 1 if step % 4 == 0 else 2
    return 2


def _station(step):
    return [1, 1, 3, 3, 3, 3, 1, 1][step % 8]


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _history, _enable_push, _esc_active, _prec_active
    global _cooldown, _blink_step, _last_dir, _pursuit_st

    _load()
    if _history is None or _check_reset(obs):
        _reset()

    any_sensor = any(int(obs[i]) for i in range(16))
    if any_sensor:
        _pursuit_st = 0
        if obs[12] or obs[13] or obs[14] or obs[15]: _last_dir = 3
        elif obs[0]  or obs[1]  or obs[2]  or obs[3]: _last_dir = 1
        else: _last_dir = 2
    else:
        _pursuit_st = min(_pursuit_st + 1, 999)

    _history.append(obs.copy())
    stuck   = obs[17] == 1
    ir_on   = obs[16] == 1
    any_vis = bool(np.any(obs[:17]))
    state   = make_state(obs, _history, _enable_push, _last_dir, _pursuit_st)
    vel     = state[8]
    mb      = state[11]

    if _enable_push and stuck and not _prec_active:
        _prec_trigger(obs)
    elif not _enable_push and stuck and not _esc_active:
        _esc_trigger(obs)

    if _prec_active:
        return ACTIONS[_prec_next()]

    if _esc_active:
        return ACTIONS[_esc_next()]

    if _cooldown > 0:
        _cooldown -= 1
        return "FW"

    if ir_on and not _enable_push:
        return "FW"

    if _enable_push:
        return "FW"

    if mb:
        return ACTIONS[_last_dir]

    if any_vis and vel != 1:
        return ACTIONS[_intercept(vel, _ep_step)]

    if state[7]:
        a = _station(_blink_step); _blink_step += 1
        return ACTIONS[a]

    _blink_step = 0

    if state in _Q:
        return ACTIONS[int(np.argmax(_Q[state]))]

    if obs[4] or obs[6] or obs[8] or obs[10]: return "FW"
    if obs[5] or obs[7] or obs[9] or obs[11]: return "FW"
    left_s  = sum(int(obs[i]) for i in range(0, 4))
    right_s = sum(int(obs[i]) for i in range(12, 16))
    if left_s > right_s: return "L22"
    if right_s > left_s: return "R22"
    if _ep_step % 70 == 0:  return "L22"
    if _ep_step % 70 == 35: return "R22"
    return "FW"