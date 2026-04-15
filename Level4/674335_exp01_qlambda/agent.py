import os, pickle
import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_DATA = None

def _load():
    global _DATA
    if _DATA is not None: return
    path = os.path.join(os.path.dirname(__file__), "q_table_exp01_final.pkl")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    _DATA = {
        "finder": {k: np.array(v) for k, v in raw["Q_finder"].items()},
        "pusher": {k: np.array(v) for k, v in raw["Q_pusher"].items()},
    }

def _select_module(obs):
    return "pusher" if (obs[16] == 1 or obs[17] == 1) else "finder"

def _heuristic(obs, rng):
   
    if obs[17] == 1:
        return int(rng.choice([0, 1, 3, 4]))  # L45/L22/R22/R45 — koi bhi turn

    if obs[16] == 1:           return 2   # IR → FW
    if any(obs[[5,7,9,11]]):   return 2   # forward near → FW
    if any(obs[[4,6,8,10]]):   return 2   # forward far  → FW
    if any(obs[0:4]):          return 1   # left sensors → L22
    if any(obs[12:16]):        return 3   # right sensors → R22

    probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05])
    return int(rng.choice(len(ACTIONS), p=probs))

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    state  = tuple(obs.astype(int).tolist())
    module = _select_module(obs)
    Q      = _DATA[module]
    action_idx = int(np.argmax(Q[state])) if state in Q else _heuristic(obs, rng)
    return ACTIONS[action_idx]
