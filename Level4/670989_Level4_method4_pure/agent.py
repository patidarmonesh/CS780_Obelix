"""
agent.py — Pure End-to-End PPO-LSTM Agent
==========================================
ZERO handcrafted controllers. The LSTM decides EVERYTHING.
Input: obs(18) + prev_action_onehot(5) = 23 dims
CPU-only inference. Requires: weights.pth (or pure_best.pth)
"""

import os
import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS  = 5
OBS_DIM    = 18
AUG_DIM    = OBS_DIM + N_ACTIONS   # 23
HIDDEN_DIM = 256

_MODEL = None
_HIDDEN = None
_PREV_ACTION = 2   # FW
_STEP = 0
_PREV_OBS = None


def _load():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class PureLSTMPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = HIDDEN_DIM
            self.encoder = nn.Sequential(nn.Linear(AUG_DIM, 128), nn.ReLU())
            self.lstm = nn.LSTM(128, HIDDEN_DIM, batch_first=True)
            self.actor = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.Tanh(), nn.Linear(64, N_ACTIONS))
            self.critic = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.Tanh(), nn.Linear(64, 1))
        def forward(self, x, h):
            enc = self.encoder(x)
            out, nh = self.lstm(enc, h)
            return self.actor(out), self.critic(out), nh
        def init_hidden(self):
            import torch
            return (torch.zeros(1, 1, HIDDEN_DIM), torch.zeros(1, 1, HIDDEN_DIM))

    model = PureLSTMPolicy()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ["weights.pth", "pure_best.pth", "pure_final.pth"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            import torch
            model.load_state_dict(torch.load(p, map_location="cpu"))
            model.eval()
            _MODEL = model
            print(f"[PureRL] Loaded {f}")
            return
    _MODEL = model
    _MODEL.eval()
    print("[PureRL] WARNING: No weights found")


def _reset():
    global _HIDDEN, _PREV_ACTION, _STEP, _PREV_OBS
    _PREV_ACTION = 2
    _STEP = 0
    _PREV_OBS = None
    if _MODEL is not None:
        _HIDDEN = _MODEL.init_hidden()
    else:
        _HIDDEN = None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    import torch
    global _HIDDEN, _PREV_ACTION, _STEP, _PREV_OBS

    _load()
    if _PREV_OBS is None or _STEP >= 1005:
        _reset()

    _STEP += 1

    # Build augmented observation: [obs, prev_action_onehot]
    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    action_onehot[_PREV_ACTION] = 1.0
    obs_aug = np.concatenate([obs.astype(np.float32), action_onehot])

    # LSTM inference (greedy)
    with torch.no_grad():
        x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0)
        logits, _, _HIDDEN = _MODEL(x, _HIDDEN)
        action_idx = logits.squeeze().argmax().item()

    _PREV_ACTION = action_idx
    _PREV_OBS = obs.copy()
    return ACTIONS[action_idx]