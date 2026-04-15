import os, sys, pickle
import numpy as np
from collections import deque
from typing import Sequence


_HERE    = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(_HERE, "q_exp06plus_best.pkl")  # ← change if needed


_Q: dict = {}


def _load_q():
    global _Q
    if not os.path.exists(PKL_PATH):
        print(f"[agent_exp06plus] WARNING: Q-table not found at {PKL_PATH}")
        print("  Running in HEURISTIC-ONLY mode.")
        return
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    raw = data["Q"]
    _Q  = {eval(k): np.array(v) for k, v in raw.items()}
    print(f"[agent_exp06plus] Loaded Q-table: {len(_Q)} states from {os.path.basename(PKL_PATH)}")


_load_q()


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS   = 5
BLINK_MEM   = 30
VEL_WIN     = 5
PURSUIT_TTL = 18
ESC_TURNS   = 4
ESC_FW      = 6
POST_COOL   = 10
PUSH_RECOV  = 6
WALL_TIMER_LONG = 5


def _wall_like_instant(obs: np.ndarray) -> int:
    left  = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd   = sum(int(obs[i]) for i in range(4, 12))
    total = left + right + fwd
    if total >= 7:                return 1
    if left >= 2 and right >= 2: return 1
    if left >= 2 and fwd >= 3:   return 1
    if right >= 2 and fwd >= 3:  return 1
    return 0


def _make_state(
    obs:           np.ndarray,
    history:       deque,
    enable_push:   bool,
    last_dir:      int,
    pursuit_steps: int,
    wall_timer:    int,
) -> tuple:
    stuck    = int(obs[17])
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8] or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9] or obs[11])

    left_s = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right_s = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd_s   = sum(int(obs[i]) for i in range(4, 12))
    total   = left_s + right_s + fwd_s

    if total == 0:             direction = 1
    elif left_s > right_s + 1: direction = 0
    elif right_s > left_s + 1: direction = 2
    else:                      direction = 1

    wall_like = _wall_like_instant(obs)

    hist     = list(history)
    any_now  = bool(np.any(obs[:17]))
    recent   = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
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
            if drift > 0.3:    vel_hint = 2
            elif drift < -0.3: vel_hint = 0

    fast_grow = 0
    if len(hist) >= 2:
        prev = sum(int(x) for x in hist[-2][:16])
        fast_grow = int((total - prev) >= 4)

    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)

    if wall_timer == 0:                  wall_timer_bucket = 0
    elif wall_timer < WALL_TIMER_LONG:   wall_timer_bucket = 1
    else:                                wall_timer_bucket = 2

    return (
        int(enable_push), stuck, ir, fwd_near, fwd_far, direction,
        wall_like, is_blind, vel_hint, fast_grow, was_close,
        moving_blind, wall_timer_bucket,
    )


class _EscapeCtrl:
    SEQ_LEN = ESC_TURNS + ESC_FW
    def __init__(self):
        self.active = False; self._step = 0; self._turn = 4

    def trigger(self, obs: np.ndarray, rng):
        total = sum(int(obs[i]) for i in range(16))
        if total == 0:
            self._turn = int(rng.integers(0, 2)) * 4
        else:
            lw = sum(int(obs[i]) for i in range(0, 4))
            rw = sum(int(obs[i]) for i in range(12, 16))
            self._turn = 4 if lw >= rw else 0
        self._step = 0; self.active = True

    def next_action(self) -> int:
        i = self._step; self._step += 1
        if self._step >= self.SEQ_LEN: self.active = False
        return self._turn if i < ESC_TURNS else 2


class _PushRecov:
    def __init__(self):
        self.active = False; self._step = 0; self._turn = 1

    def trigger(self, obs: np.ndarray):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 1 if lw <= rw else 3
        self._step = 0; self.active = True

    def next_action(self) -> int:
        i = self._step; self._step += 1
        if self._step >= PUSH_RECOV: self.active = False
        return self._turn if i % 2 == 0 else 2


def _intercept(vel_hint: int, step: int) -> int:
    if vel_hint == 2: return 3 if step % 4 == 0 else 2
    if vel_hint == 0: return 1 if step % 4 == 0 else 2
    return 2


def _station(step: int) -> int:
    return [1, 1, 3, 3, 3, 3, 1, 1][step % 8]


class _AgentState:
    def __init__(self):
        self.history     = deque(maxlen=BLINK_MEM)
        self.enable_push = False
        self.wall_timer  = 0
        self.last_dir    = 2
        self.pursuit_st  = 999
        self.cooldown    = 0
        self.blink_step  = 0
        self.esc         = _EscapeCtrl()
        self.push_rec    = _PushRecov()
        self.step        = 0
        self.quiet_steps = 0


_ST = _AgentState()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    g = _ST
    g.step += 1
    g.history.append(obs.copy())

    left  = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    fwd   = sum(int(obs[i]) for i in range(4, 12))
    total = left + right + fwd

    wall_now = (total >= 7
                or (left >= 2 and right >= 2)
                or (left >= 2 and fwd >= 3)
                or (right >= 2 and fwd >= 3))
    if wall_now:
        g.wall_timer = min(g.wall_timer + 1, 20)
    else:
        g.wall_timer = max(g.wall_timer - 1, 0)

    any_active = bool(np.any(obs[:18]))
    if not any_active:
        g.quiet_steps += 1
    else:
        g.quiet_steps = 0

    if g.quiet_steps > 150 and g.enable_push:
        g.enable_push = False
        g.wall_timer  = 0
        g.pursuit_st  = 999
        g.quiet_steps = 0

    cur_wt = g.wall_timer
    state  = _make_state(obs, g.history, g.enable_push, g.last_dir,
                         g.pursuit_st, cur_wt)
    vel = state[8]
    mb  = state[11]

    stuck  = obs[17] == 1
    ir_on  = obs[16] == 1
    any_vs = bool(np.any(obs[:17]))

    if any(int(obs[i]) for i in range(16)):
        g.pursuit_st = 0
        if obs[12] or obs[13] or obs[14] or obs[15]: g.last_dir = 3
        elif obs[0] or obs[1] or obs[2] or obs[3]:   g.last_dir = 1
        else:                                          g.last_dir = 2
    else:
        g.pursuit_st = min(g.pursuit_st + 1, 999)

    if ir_on and not g.enable_push:
        g.enable_push = True

    if g.enable_push and stuck and not g.push_rec.active:
        g.push_rec.trigger(obs)
    elif not g.enable_push and stuck and not g.esc.active:
        g.esc.trigger(obs, rng)
        g.cooldown = 0

    if g.push_rec.active:
        action_idx = g.push_rec.next_action()

    elif g.esc.active:
        action_idx = g.esc.next_action()
        if not g.esc.active:
            g.cooldown = POST_COOL

    elif g.cooldown > 0:
        action_idx = 2; g.cooldown -= 1

    elif ir_on and not g.enable_push:
        action_idx = 2

    elif g.enable_push:
        action_idx = 2

    elif mb:
        action_idx = g.last_dir

    elif any_vs and vel != 1 and not g.enable_push:
        action_idx = _intercept(vel, g.step)
        g.blink_step = 0

    elif state[7]:
        action_idx = _station(g.blink_step)
        g.blink_step += 1

    else:
        g.blink_step = 0
        if state in _Q:
            action_idx = int(np.argmax(_Q[state]))
        else:
            if total > 0:
                action_idx = 1 if left > right + 1 else (3 if right > left + 1 else 2)
            else:
                phase = g.step % 62
                if phase < 55:
                    action_idx = 2
                else:
                    action_idx = 3 if (g.step // 62) % 2 == 0 else 1

    return ACTIONS[action_idx]