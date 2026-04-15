import os
import pickle
import numpy as np
from typing import Sequence
from collections import deque


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5


BLINK_MEM   = 30
VEL_WIN     = 5
ESC_TURNS   = 4
ESC_FW      = 6
POST_COOL   = 8
PURSUIT_TTL = 18


def make_state(obs, history, enable_push, last_dir, pursuit_steps):
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])
    left_s   = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])
    right_s  = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    any_now  = ir or fwd_near or fwd_far or left_s > 0 or right_s > 0

    if ir:                             phase = 4
    elif fwd_near:                     phase = 3
    elif fwd_far:                      phase = 2
    elif left_s > 0 or right_s > 0:    phase = 1
    else:                              phase = 0

    fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
    fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    if fl + fr == 0:       direction = 1
    elif fl > fr + 1:      direction = 0
    elif fr > fl + 1:      direction = 2
    else:                  direction = 1

    if left_s > 0 and right_s > 0:   wall_side = 3
    elif left_s > 0:                  wall_side = 1
    elif right_s > 0:                 wall_side = 2
    else:                             wall_side = 0

    hist = list(history)
    recent = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind = int(not any_now and recent)
    was_close = int(any(bool(h[16] == 1) for h in hist[-15:]) if hist else False)

    vel_hint = 1
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6]) + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0: centroids.append((r - l) / t)
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift > 0.3:    vel_hint = 2
            elif drift < -0.3: vel_hint = 0

    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)
    return (phase, direction, int(enable_push), is_blind,
            vel_hint, wall_side, was_close, moving_blind)


class _EscapeCtrl:
    SEQ_LEN = ESC_TURNS + ESC_FW
    def __init__(self):
        self.active = False
        self._step = 0
        self._turn = 4
    def trigger(self, obs):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 4 if lw > rw else (0 if rw > lw else 4)
        self._step = 0
        self.active = True
    def next_action(self):
        i = self._step; self._step += 1
        if self._step >= self.SEQ_LEN: self.active = False
        return self._turn if i < ESC_TURNS else 2


def _intercept(vel_hint, step):
    if vel_hint == 2: return 3 if step % 3 == 0 else 2
    if vel_hint == 0: return 1 if step % 3 == 0 else 2
    return 2


def _station(step):
    return [1, 1, 3, 3, 3, 3, 1, 1][step % 8]


_Q = None
_loaded = False


def _load_q():
    global _Q, _loaded
    if _loaded: return
    _loaded = True
    _Q = {}
    d = os.path.dirname(os.path.abspath(__file__))
    for fname in ["q_final_best.pkl", "q_table_best.pkl", "q_exp06_best.pkl",
                   "q_final_final.pkl", "q_table_final.pkl"]:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if "Q" in data:
                    for k_str, v_list in data["Q"].items():
                        try: _Q[eval(k_str)] = np.array(v_list, dtype=np.float64)
                        except: pass
                elif "Q_finder" in data:
                    for mod_key in ["Q_finder", "Q_pusher"]:
                        if mod_key in data:
                            for k_str, v_list in data[mod_key].items():
                                try:
                                    key = eval(k_str)
                                    arr = np.array(v_list, dtype=np.float64)
                                    if key not in _Q: _Q[key] = arr
                                    else: _Q[key] = np.maximum(_Q[key], arr)
                                except: pass
                return
            except: pass


class _AgentState:
    def __init__(self):
        self.reset()
    def reset(self):
        self.history = deque(maxlen=BLINK_MEM)
        self.esc = _EscapeCtrl()
        self.cooldown = 0
        self.blink_step = 0
        self.last_dir = 2
        self.pursuit_st = 999
        self.step_count = 0
        self.prev_obs = None
        self.push_confirmed = False
        self.consecutive_fw_moving = 0
        self.last_pos = None
        self.ir_while_moving = 0


_S = _AgentState()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_q()
    s = _S

    new_ep = False
    if s.prev_obs is None:
        new_ep = True
    elif s.step_count >= 1005:
        new_ep = True
    elif s.step_count > 10:
        prev_sensor_count = int(np.sum(s.prev_obs[:17]))
        curr_sensor_count = int(np.sum(obs[:17]))
        prev_stuck = int(s.prev_obs[17])
        curr_stuck = int(obs[17])
        if prev_sensor_count >= 3 and curr_sensor_count == 0 and prev_stuck == 0 and curr_stuck == 0:
            new_ep = True
        bit_diff = int(np.sum(np.abs(obs[:17] - s.prev_obs[:17])))
        if bit_diff >= 6 and prev_stuck != curr_stuck:
            new_ep = True
    if new_ep:
        s.reset()

    s.step_count += 1
    s.history.append(obs.copy())
    s.prev_obs = obs.copy()

    stuck = bool(obs[17])
    ir_on = bool(obs[16])
    any_vis = bool(np.any(obs[:17]))

    front_detected = bool(obs[16] or any(obs[i] for i in range(4, 12)))
    if front_detected:
        s.pursuit_st = 0
        fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
        fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
        if fl > fr + 1:      s.last_dir = 1
        elif fr > fl + 1:    s.last_dir = 3
        else:                s.last_dir = 2
    else:
        s.pursuit_st = min(s.pursuit_st + 1, 999)

    state = make_state(obs, s.history, s.push_confirmed, s.last_dir, s.pursuit_st)
    vel = state[4]
    mb  = state[7]

    if stuck and not s.esc.active:
        s.esc.trigger(obs)
        s.cooldown = 0

    s.cooldown = max(0, s.cooldown - 1)

    if s.esc.active:
        a = s.esc.next_action()
        if not s.esc.active:
            s.cooldown = POST_COOL
        return ACTIONS[a]

    if s.cooldown > 0:
        return "FW"

    if stuck:
        s.esc.trigger(obs)
        return ACTIONS[s.esc.next_action()]

    if ir_on:
        return "FW"

    if mb:
        return ACTIONS[s.last_dir]

    front_vis = bool(obs[16] or obs[4] or obs[5] or obs[6] or obs[7]
                     or obs[8] or obs[9] or obs[10] or obs[11])

    if front_vis and vel != 1:
        s.blink_step = 0
        return ACTIONS[_intercept(vel, s.step_count)]

    if state[3]:
        a = _station(s.blink_step)
        s.blink_step += 1
        return ACTIONS[a]

    if front_vis:
        s.blink_step = 0
        if _Q and state in _Q:
            return ACTIONS[int(np.argmax(_Q[state]))]
        direction = state[1]
        if direction == 0:   return "L22"
        elif direction == 2: return "R22"
        else:                return "FW"

    s.blink_step = 0
    p = rng.random()
    if p < 0.025:    return "L45"
    elif p < 0.175:  return "L22"
    elif p < 0.825:  return "FW"
    elif p < 0.975:  return "R22"
    else:            return "R45"