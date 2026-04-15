import os, pickle, ast
import numpy as np
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
WEIGHTS_FILE = "q_table_best.pkl"
BLINK_MEM = 30
VEL_WINDOW = 6
MAX_EP_STEPS = 1000
PUSH_STUCK_LIMIT = 15
COOLDOWN_STEPS = 8
_ESCAPE_SEQ = [0, 0, 2, 2]

_Q = None
_history = None
_enable_push = False
_esc_step = 0
_cooldown = 0
_push_stuck = 0
_last_dir = 2
_pursuit_step = 0
_step_in_ep = 0
_prev_sum = -1

def _load():
    global _Q
    if _Q is not None:
        return
    _Q = {m: {} for m in ("finder", "pusher", "escape")}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE)
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        raw = pickle.load(f)
    for m in ("finder", "pusher", "escape"):
        key = f"Q_{m}"
        if key in raw:
            _Q[m] = {ast.literal_eval(k): np.array(v) for k, v in raw[key].items()}

def _reset():
    global _history, _enable_push, _esc_step, _cooldown, _push_stuck, _last_dir, _pursuit_step, _step_in_ep, _prev_sum
    _history = deque(maxlen=BLINK_MEM)
    _enable_push = False
    _esc_step = 0
    _cooldown = 0
    _push_stuck = 0
    _last_dir = 2
    _pursuit_step = 0
    _step_in_ep = 0
    _prev_sum = -1

def _detect_boundary(obs):
    global _step_in_ep, _prev_sum
    _step_in_ep += 1
    curr_sum = int(np.sum(obs))
    boundary = (_step_in_ep >= MAX_EP_STEPS or (_step_in_ep > 10 and _prev_sum >= 3 and curr_sum == 0 and _enable_push))
    _prev_sum = curr_sum
    return boundary

def sensor_centroid(obs):
    lf = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
    rf = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    total = lf + rf
    if total == 0:
        return None
    if lf > rf + 1:
        return -1
    if rf > lf + 1:
        return +1
    return 0

def make_state(obs, history, enable_push, moving_blind):
    stuck = int(obs[17])
    ir = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8] or obs[10])
    fwd_far = int(obs[5] or obs[7] or obs[9] or obs[11])
    count_fwd = sum(int(obs[i]) for i in range(4, 12))
    count_side = sum(int(obs[i]) for i in range(0, 4)) + sum(int(obs[i]) for i in range(12, 16))
    sides_any = int(count_side > 0)
    spread_high = int(count_side > 2 and count_fwd >= 2)
    hist_list = list(history)
    n = len(hist_list)
    recent_saw = int(any(bool(np.any(h[:17])) for h in hist_list))
    was_close = int(any(bool(h[16] == 1) for h in hist_list))
    any_now = bool(np.any(obs[:17]))
    is_blind = int(not any_now and recent_saw)
    vel_hint = 0
    if n >= 2:
        centroids = []
        for h in hist_list[-VEL_WINDOW:]:
            c = sensor_centroid(h)
            if c is not None:
                centroids.append(c)
        if len(centroids) >= 3:
            half = len(centroids) // 2
            avg_first = np.mean(centroids[:half])
            avg_second = np.mean(centroids[half:])
            drift = avg_second - avg_first
            if drift > 0.4:
                vel_hint = 1
            elif drift < -0.4:
                vel_hint = -1
    fast_grow = 0
    if n >= 2:
        prev_cnt = sum(int(x) for x in hist_list[-2][:16])
        curr_cnt = sum(int(x) for x in obs[:16])
        fast_grow = int((curr_cnt - prev_cnt) >= 4)
    return (
        stuck,
        ir,
        fwd_near,
        fwd_far,
        sides_any,
        spread_high,
        fast_grow,
        is_blind,
        was_close,
        vel_hint + 1,
        int(enable_push),
        int(moving_blind),
    )

def escape_action(esc_step):
    return _ESCAPE_SEQ[esc_step % len(_ESCAPE_SEQ)]

def pursuit_action(last_dir, vel_hint, step):
    if vel_hint == 1:
        return 3 if step % 4 == 0 else last_dir
    if vel_hint == -1:
        return 1 if step % 4 == 0 else last_dir
    return last_dir

def explore_action(step):
    if step % 40 == 0:
        return 1
    probs = np.array([0.0, 0.10, 0.80, 0.10, 0.0])
    return int(np.random.choice(N_ACTIONS, p=probs))

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _history, _enable_push, _esc_step, _cooldown, _push_stuck, _last_dir, _pursuit_step

    _load()

    if _history is None or _detect_boundary(obs):
        _reset()

    _history.append(obs.copy())

    if any(obs[0:4]):
        _last_dir = 1
        _pursuit_step = 0
    elif any(obs[4:12]):
        _last_dir = 2
        _pursuit_step = 0
    elif any(obs[12:16]):
        _last_dir = 3
        _pursuit_step = 0
    else:
        _pursuit_step += 1

    stuck = obs[17] == 1
    ir_on = obs[16] == 1 and not stuck
    any_vis = bool(np.any(obs[:17]))
    recent = any(bool(np.any(h[:17])) for h in _history)
    blind = not any_vis and recent
    moving_blind = int(blind and _pursuit_step <= 20)
    _cooldown = max(0, _cooldown - 1)

    state = make_state(obs, _history, _enable_push, moving_blind)
    vel = state[9] - 1

    if _enable_push and stuck:
        _push_stuck += 1
        if _push_stuck >= PUSH_STUCK_LIMIT:
            _enable_push = False
            _push_stuck = 0
            _cooldown = COOLDOWN_STEPS
    elif not stuck:
        _push_stuck = 0

    if stuck and _enable_push:
        return "FW"

    if stuck and not _enable_push:
        a = escape_action(_esc_step)
        _esc_step += 1
        _cooldown = COOLDOWN_STEPS
        return ACTIONS[a]

    _esc_step = 0

    if ir_on and not _enable_push and _cooldown == 0:
        return "FW"

    if _enable_push and not stuck:
        return "FW"

    if blind and recent:
        return ACTIONS[pursuit_action(_last_dir, vel, _step_in_ep)]

    if any_vis and _cooldown == 0:
        module = "pusher" if _enable_push else "finder"
        q_mod = _Q.get(module, {})
        if state in q_mod:
            return ACTIONS[int(np.argmax(q_mod[state]))]
        if obs[4] or obs[6] or obs[8] or obs[10]:
            return "FW"
        if obs[5] or obs[7] or obs[9] or obs[11]:
            return "FW"
        left = sum(int(obs[i]) for i in range(0, 4))
        right = sum(int(obs[i]) for i in range(12, 16))
        if left > right:
            return "L22"
        if right > left:
            return "R22"
        return "FW"

    return ACTIONS[explore_action(_step_in_ep)]