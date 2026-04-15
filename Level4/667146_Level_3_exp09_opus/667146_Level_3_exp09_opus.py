"""
agent.py — OBELIX exp09 (667146) Inference Agent
==================================================
This file is submitted to Codabench for evaluation.

It loads the trained Q-table and uses it to make decisions.
The agent also has hardcoded behaviors for escape, pursuit,
interception, and station keeping.

Key constraint: the policy() function only receives obs and rng.
It does NOT receive:
- reward (can't detect attachment via +100 bonus)
- done flag (can't detect episode boundaries)
- position (can't know where robot is)
- facing angle (can't know which direction robot faces)

So the agent must:
- Detect episode boundaries heuristically (sensor pattern changes)
- Infer push mode from IR streak patterns
- Maintain all temporal context in global variables

Submit as zip: agent.py + q_final_best.pkl (or similar .pkl file)
"""

import os
import pickle
import numpy as np
from typing import Sequence
from collections import deque


# ══════════════════════════════════════════════════════════════
# Constants (must match train.py)
# ══════════════════════════════════════════════════════════════

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5

# Controller parameters (identical to train.py)
BLINK_MEM   = 30    # observation history length
VEL_WIN     = 5     # velocity estimation window
ESC_TURNS   = 4     # escape: 4 × 45° = 180° turn
ESC_FW      = 6     # escape: 6 × 5px = 30px forward
POST_COOL   = 8     # post-escape cooldown (8 × FW)
PURSUIT_TTL = 18    # keep pursuing for 18 steps after losing contact


# ══════════════════════════════════════════════════════════════
# STATE CONSTRUCTION (IDENTICAL to train.py)
# ══════════════════════════════════════════════════════════════
# If this function differs from train.py even slightly,
# the Q-table lookups will fail (wrong state → wrong Q-values)

def make_state(obs, history, enable_push, last_dir, pursuit_steps):
    """
    Convert raw observation + temporal context into 8-feature state tuple.

    State features:
    0. phase (0-4):        distance ladder (nothing/side/far/near/IR)
    1. direction (0-2):    object left(0)/center(1)/right(2) of forward arc
    2. enable_push (0-1):  is box attached?
    3. is_blind (0-1):     sensors just went dark after recent activity?
    4. vel_hint (0-2):     box moving left(0)/static(1)/right(2)
    5. wall_side (0-3):    which sides have sensor activity?
    6. was_close (0-1):    was IR active in last 15 steps?
    7. moving_blind (0-1): blind + recently tracking → keep pursuing
    """

    # ── Basic sensor readings ─────────────────────────────────
    ir       = int(obs[16])                                          # IR sensor
    fwd_near = int(obs[4] or obs[6] or obs[8]  or obs[10])          # any forward NEAR sonar
    fwd_far  = int(obs[5] or obs[7] or obs[9]  or obs[11])          # any forward FAR sonar
    left_s   = int(obs[0]) + int(obs[1]) + int(obs[2])  + int(obs[3])     # left sensor sum (0-4)
    right_s  = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])  # right sensor sum (0-4)
    any_now  = ir or fwd_near or fwd_far or left_s > 0 or right_s > 0     # anything detected?

    # ── Phase: 5-level distance estimate ──────────────────────
    # The KEY innovation of exp09
    # Higher phases override lower ones (IR > near > far > sides > nothing)
    if ir:                             phase = 4   # within 20px (IR range)
    elif fwd_near:                     phase = 3   # within 90px (near sonar range)
    elif fwd_far:                      phase = 2   # within 150px (far sonar range)
    elif left_s > 0 or right_s > 0:    phase = 1   # something on sides only
    else:                              phase = 0   # nothing detected

    # ── Direction: left/center/right ──────────────────────────
    # Compare left-forward vs right-forward sensor counts
    fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])     # left-forward total
    fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])   # right-forward total
    if fl + fr == 0:       direction = 1   # no forward sensors → unknown/center
    elif fl > fr + 1:      direction = 0   # more left → object on LEFT
    elif fr > fl + 1:      direction = 2   # more right → object on RIGHT
    else:                  direction = 1   # balanced → object CENTERED

    # ── Wall side: which side sensors are firing ──────────────
    # Both sides firing = strong wall indicator (box is small, rarely hits both)
    if left_s > 0 and right_s > 0:   wall_side = 3   # BOTH → likely wall
    elif left_s > 0:                  wall_side = 1   # left only
    elif right_s > 0:                 wall_side = 2   # right only
    else:                             wall_side = 0   # no sides

    # ── Temporal features from history ────────────────────────
    hist = list(history)

    # recent: was ANYTHING detected in last 20 steps?
    recent = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False

    # is_blind: sensors are dark NOW but were active recently = box blinked
    is_blind = int(not any_now and recent)

    # was_close: was IR active in last 15 steps? (box was very near)
    was_close = int(any(bool(h[16] == 1) for h in hist[-15:]) if hist else False)

    # ── Velocity hint from sensor centroid drift ──────────────
    vel_hint = 1   # default: stationary/unknown
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:   # look at last 5 observations
            l = int(h[4]) + int(h[5]) + int(h[6]) + int(h[7])     # left-forward count
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])   # right-forward count
            t = l + r
            if t > 0:
                centroids.append((r - l) / t)   # +1=all right, -1=all left
        if len(centroids) >= 3:
            # Compare recent centroids to older ones
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift > 0.3:    vel_hint = 2   # shifting right → box moving right
            elif drift < -0.3: vel_hint = 0   # shifting left → box moving left

    # ── Moving blind: should robot PURSUE during blink? ───────
    # If box was recently detected AND just disappeared, keep heading
    # in the last known direction (don't stop and oscillate)
    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)

    return (phase, direction, int(enable_push), is_blind,
            vel_hint, wall_side, was_close, moving_blind)


# ══════════════════════════════════════════════════════════════
# ESCAPE CONTROLLER (identical to train.py)
# ══════════════════════════════════════════════════════════════

class _EscapeCtrl:
    """
    Handles wall escape when stuck.

    Sequence: 4 turns (180°) + 6 FW (30px clearance)
    Direction: turns AWAY from the side with more sensor readings
    After completion: caller adds POST_COOL=8 FW steps (40px more)
    Total clearance: 70px from wall
    """
    SEQ_LEN = ESC_TURNS + ESC_FW   # 10 steps total

    def __init__(self):
        self.active = False    # is escape running?
        self._step = 0         # current step in sequence
        self._turn = 4         # turn direction (0=L45 or 4=R45)

    def trigger(self, obs):
        """Start escape. Choose direction based on which side detects wall."""
        lw = sum(int(obs[i]) for i in range(0, 4))    # left sensor count
        rw = sum(int(obs[i]) for i in range(12, 16))   # right sensor count
        # Turn AWAY from stronger side (wall is on that side)
        self._turn = 4 if lw > rw else (0 if rw > lw else 4)
        self._step = 0
        self.active = True

    def next_action(self):
        """Return next action in escape sequence."""
        i = self._step
        self._step += 1
        if self._step >= self.SEQ_LEN:
            self.active = False   # sequence complete
        # Steps 0-3: TURN (180°), Steps 4-9: FORWARD (30px)
        return self._turn if i < ESC_TURNS else 2


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _intercept(vel_hint, step):
    """
    Intercept a moving box by leading the target.

    vel_hint=2 (moving right): turn right every 3rd step, FW otherwise
    vel_hint=0 (moving left): turn left every 3rd step, FW otherwise
    vel_hint=1 (stationary): always FW
    """
    if vel_hint == 2: return 3 if step % 3 == 0 else 2   # R22 + FW + FW pattern
    if vel_hint == 0: return 1 if step % 3 == 0 else 2   # L22 + FW + FW pattern
    return 2   # FW


def _station(step):
    """
    Station keeping during box blink (for STATIONARY boxes only).

    Oscillates in a wide arc: L22,L22,R22,R22,R22,R22,L22,L22
    This scans ~90° left to right while staying in roughly the same area.

    For MOVING boxes, moving_blind triggers PURSUIT instead (priority 5
    in agent, which runs before station keeping at priority 7).
    """
    return [1, 1, 3, 3, 3, 3, 1, 1][step % 8]


# ══════════════════════════════════════════════════════════════
# Q-TABLE LOADING
# ══════════════════════════════════════════════════════════════

_Q = None       # the loaded Q-table: dict mapping state_tuple → numpy array of 5 Q-values
_loaded = False  # has loading been attempted?


def _load_q():
    """
    Load the trained Q-table from a pickle file.

    Tries multiple possible filenames because different training
    runs may save with different prefixes.

    The Q-table is a dict: {state_tuple: np.array([Q_L45, Q_L22, Q_FW, Q_R22, Q_R45])}
    Example entry:
        (3, 1, 0, 0, 1, 0, 1, 0) → [2.1, 3.4, 8.7, 3.2, 1.9]
        This means: phase=3, direction=center, not pushing, not blind,
        vel=stationary, no wall side, was close, not pursuing
        Best action: FW (Q=8.7)
    """
    global _Q, _loaded
    if _loaded:
        return
    _loaded = True
    _Q = {}

    # Look for the Q-table file in the same directory as this script
    d = os.path.dirname(os.path.abspath(__file__))

    # Try multiple possible filenames (different training runs save differently)
    for fname in ["q_final_best.pkl", "q_table_best.pkl", "q_exp06_best.pkl",
                   "q_final_final.pkl", "q_table_final.pkl"]:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)

                # Format 1: single unified Q-table (exp09 format)
                # {"Q": {"(3, 1, 0, ...)": [2.1, 3.4, 8.7, 3.2, 1.9]}}
                if "Q" in data:
                    for k_str, v_list in data["Q"].items():
                        try:
                            _Q[eval(k_str)] = np.array(v_list, dtype=np.float64)
                        except:
                            pass

                # Format 2: separate finder/pusher Q-tables (exp06 format)
                # {"Q_finder": {...}, "Q_pusher": {...}}
                elif "Q_finder" in data:
                    for mod_key in ["Q_finder", "Q_pusher"]:
                        if mod_key in data:
                            for k_str, v_list in data[mod_key].items():
                                try:
                                    key = eval(k_str)
                                    arr = np.array(v_list, dtype=np.float64)
                                    # If same state exists in both tables,
                                    # keep the HIGHER Q-values (element-wise max)
                                    if key not in _Q:
                                        _Q[key] = arr
                                    else:
                                        _Q[key] = np.maximum(_Q[key], arr)
                                except:
                                    pass
                return   # successfully loaded, stop trying other filenames
            except:
                pass   # file exists but failed to load, try next


# ══════════════════════════════════════════════════════════════
# AGENT STATE (persists across steps within an episode)
# ══════════════════════════════════════════════════════════════

class _AgentState:
    """
    All mutable state that persists between policy() calls.

    The policy() function is called once per step with no explicit
    "reset" signal between episodes. This class holds everything
    the agent needs to remember across steps.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque(maxlen=BLINK_MEM)   # last 30 observations
        self.esc = _EscapeCtrl()                  # escape controller
        self.cooldown = 0                         # post-escape FW counter
        self.blink_step = 0                       # station-keeping step counter
        self.last_dir = 2                         # last known box direction (default: FW)
        self.pursuit_st = 999                     # steps since last front detection (999=never)
        self.step_count = 0                       # steps in current episode
        self.prev_obs = None                      # previous observation (for episode detection)
        self.push_confirmed = False               # is box attached? (inferred)
        self.consecutive_fw_moving = 0            # (unused in final version)
        self.last_pos = None                      # (unused in final version)
        self.ir_while_moving = 0                  # (unused in final version)


# Global agent state — single instance that persists across all policy() calls
_S = _AgentState()


# ══════════════════════════════════════════════════════════════
# POLICY FUNCTION (called by Codabench evaluator every step)
# ══════════════════════════════════════════════════════════════

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose an action given the current sensor observation.

    This function is called by the Codabench evaluator at every
    environment step. It must return one of: 'L45','L22','FW','R22','R45'

    Args:
        obs: numpy array shape (18,), binary 0/1
             [0-15] = 8 sonar sensors × 2 zones (far/near)
             [16] = IR sensor (short range, directly ahead)
             [17] = stuck flag (1 = can't move forward)
        rng: numpy random generator (seeded by evaluator)

    Returns:
        action string

    The function maintains internal state via the global _S object.
    Episode boundaries are detected heuristically.
    """

    # ── Load Q-table (only on first call) ─────────────────────
    _load_q()
    s = _S   # shorthand for agent state

    # ══════════════════════════════════════════════════════════
    # EPISODE BOUNDARY DETECTION
    # ══════════════════════════════════════════════════════════
    # The evaluator doesn't tell us when a new episode starts.
    # We must INFER it from observation patterns.
    new_ep = False

    if s.prev_obs is None:
        # Very first call ever → definitely a new episode
        new_ep = True

    elif s.step_count >= 1005:
        # Exceeded max steps → must be a new episode
        # (using 1005 instead of 1000 for safety margin)
        new_ep = True

    elif s.step_count > 10:
        # Heuristic: sudden sensor change suggests new episode
        # (robot teleported to random new position)
        prev_sensor_count = int(np.sum(s.prev_obs[:17]))
        curr_sensor_count = int(np.sum(obs[:17]))
        prev_stuck = int(s.prev_obs[17])
        curr_stuck = int(obs[17])

        # If many sensors were active and now ALL are zero
        # (and not due to being stuck), probably new episode
        if prev_sensor_count >= 3 and curr_sensor_count == 0 and prev_stuck == 0 and curr_stuck == 0:
            new_ep = True

        # If 6+ sensor bits changed AND stuck flag changed
        # (dramatic observation change = likely teleported)
        bit_diff = int(np.sum(np.abs(obs[:17] - s.prev_obs[:17])))
        if bit_diff >= 6 and prev_stuck != curr_stuck:
            new_ep = True

    if new_ep:
        s.reset()   # clear all internal state for new episode

    # ══════════════════════════════════════════════════════════
    # UPDATE INTERNAL STATE
    # ══════════════════════════════════════════════════════════
    s.step_count += 1
    s.history.append(obs.copy())   # add to observation history
    s.prev_obs = obs.copy()        # remember for next step's boundary detection

    # ── Read key sensor values ────────────────────────────────
    stuck = bool(obs[17])          # can't move forward?
    ir_on = bool(obs[16])          # something within 20px ahead?
    any_vis = bool(np.any(obs[:17]))   # ANY sensor active?

    # ── Track pursuit direction ───────────────────────────────
    # When front sensors (or IR) detect something, record which direction
    # This info is used during PURSUIT mode (when box blinks)
    front_detected = bool(obs[16] or any(obs[i] for i in range(4, 12)))
    if front_detected:
        s.pursuit_st = 0   # just detected something ahead
        # Determine direction
        fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
        fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
        if fl > fr + 1:      s.last_dir = 1   # box is LEFT → L22
        elif fr > fl + 1:    s.last_dir = 3   # box is RIGHT → R22
        else:                s.last_dir = 2   # box is CENTERED → FW
    else:
        s.pursuit_st = min(s.pursuit_st + 1, 999)   # no detection, increment counter

    # ── Compute current state ─────────────────────────────────
    state = make_state(obs, s.history, s.push_confirmed, s.last_dir, s.pursuit_st)
    vel = state[4]    # velocity hint (0=left, 1=static, 2=right)
    mb  = state[7]    # moving_blind (should we pursue during blink?)

    # ══════════════════════════════════════════════════════════
    # CONTROLLER TRIGGERS
    # ══════════════════════════════════════════════════════════

    # If stuck and escape not already running → start escape
    if stuck and not s.esc.active:
        s.esc.trigger(obs)
        s.cooldown = 0

    # Decrement cooldown timer
    s.cooldown = max(0, s.cooldown - 1)

    # ══════════════════════════════════════════════════════════
    # ACTION SELECTION (strict priority system)
    # Higher priority ALWAYS overrides lower priority
    # ══════════════════════════════════════════════════════════

    # ── PRIORITY 1: Escape active → follow escape sequence ────
    # The escape controller handles 180° turn + 30px forward movement
    if s.esc.active:
        a = s.esc.next_action()
        if not s.esc.active:
            s.cooldown = POST_COOL   # escape done → start cooldown
        return ACTIONS[a]

    # ── PRIORITY 2: Cooldown → go forward ─────────────────────
    # After escape, move forward for POST_COOL=8 more steps
    # Don't consult Q-table during cooldown (still too close to wall)
    if s.cooldown > 0:
        return "FW"

    # ── PRIORITY 3: Stuck but escape not triggered (safety) ───
    # Shouldn't happen (priority 1 catches it), but just in case
    if stuck:
        s.esc.trigger(obs)
        return ACTIONS[s.esc.next_action()]

    # ── PRIORITY 4: IR on → go forward ────────────────────────
    # Something is within 20px directly ahead
    # Could be box (→ attachment +100) or wall (→ stuck -200)
    # But going forward is the right move either way:
    # - If box: we attach (the whole point of the task)
    # - If wall: we get stuck and escape handles it (priority 1)
    # Not going forward when IR fires = guaranteed failure (never attach)
    if ir_on:
        return "FW"

    # ── PRIORITY 5: Moving blind → pursue last direction ──────
    # Box was recently detected but sensors are now dark (blink)
    # AND we were recently tracking it (pursuit_st <= 18)
    # Keep moving in the last known direction to stay close to box
    # This is THE key Level 3 fix: don't stop when box blinks!
    if mb:
        return ACTIONS[s.last_dir]   # 1=L22, 2=FW, or 3=R22

    # ── Check if anything is in the FRONT sensors ─────────────
    front_vis = bool(obs[16] or obs[4] or obs[5] or obs[6] or obs[7]
                     or obs[8] or obs[9] or obs[10] or obs[11])

    # ── PRIORITY 6: Sensors active + box moving → intercept ───
    # If we can see something ahead AND it's moving (vel_hint ≠ 1),
    # lead the target by occasionally turning in its direction
    if front_vis and vel != 1:
        s.blink_step = 0
        return ACTIONS[_intercept(vel, s.step_count)]

    # ── PRIORITY 7: Blind → station keep ──────────────────────
    # Sensors were recently active but now dark (box blinked)
    # AND we're NOT in moving_blind mode (box was stationary)
    # Oscillate gently to scan for box reappearance
    if state[3]:   # is_blind = 1
        a = _station(s.blink_step)
        s.blink_step += 1
        return ACTIONS[a]

    # ── PRIORITY 8: Sensors active → Q-table or heuristic ─────
    # Something is detected ahead, no special case applies
    # This is where the Q-table is actually used!
    if front_vis:
        s.blink_step = 0
        # Look up Q-table: if this state was seen during training,
        # pick the action with highest Q-value
        if _Q and state in _Q:
            return ACTIONS[int(np.argmax(_Q[state]))]

        # Fallback: if state wasn't in Q-table, use simple heuristic
        # Turn toward the detected object
        direction = state[1]   # 0=left, 1=center, 2=right
        if direction == 0:   return "L22"    # object on left → turn left
        elif direction == 2: return "R22"    # object on right → turn right
        else:                return "FW"     # object centered → go forward

    # ── PRIORITY 9: Nothing detected → random exploration ─────
    # No sensors active, nothing in history, just wander
    # Biased toward forward movement to cover the arena
    s.blink_step = 0
    p = rng.random()
    if p < 0.025:    return "L45"     # 2.5% — sharp left turn
    elif p < 0.175:  return "L22"     # 15%  — gentle left turn
    elif p < 0.825:  return "FW"      # 65%  — move forward (MOST of the time)
    elif p < 0.975:  return "R22"     # 15%  — gentle right turn
    else:            return "R45"     # 2.5% — sharp right turn
    # This creates a forward-biased random walk that sweeps the arena
    # in gentle curves, eventually detecting the box
