

import os
import numpy as np
from typing import Sequence


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS  = 5
OBS_DIM    = 18
AUG_DIM    = OBS_DIM + N_ACTIONS 
HIDDEN_DIM = 128


ESC_TURNS = 4   
ESC_FW    = 4   


_MODEL       = None   
_HIDDEN      = None   
_PREV_ACTION = 2      
_STEP        = 0      
_PREV_OBS    = None   
_ESC         = None   



class EscapeController:
    TOTAL_STEPS = ESC_TURNS + ESC_FW

    def __init__(self):
        self.active = False
        self._step = 0
        self._turn_action = 4

    def trigger(self, obs):
        left_sensors = sum(int(obs[i]) for i in range(0, 4))
        right_sensors = sum(int(obs[i]) for i in range(12, 16))
        self._turn_action = 4 if left_sensors >= right_sensors else 0
        self._step = 0
        self.active = True

    def get_action(self):
        i = self._step
        self._step += 1
        if self._step >= self.TOTAL_STEPS:
            self.active = False
        return self._turn_action if i < ESC_TURNS else 2



def _load_model():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class ActorCriticLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = HIDDEN_DIM
            self.encoder = nn.Sequential(
                nn.Linear(AUG_DIM, 64),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(64, HIDDEN_DIM, batch_first=True)
            self.actor = nn.Sequential(
                nn.Linear(HIDDEN_DIM, 64),
                nn.Tanh(),
                nn.Linear(64, N_ACTIONS),
            )
            self.critic = nn.Sequential(
                nn.Linear(HIDDEN_DIM, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        def forward(self, x, hidden):
            enc = self.encoder(x)
            out, new_hidden = self.lstm(enc, hidden)
            logits = self.actor(out)
            values = self.critic(out)
            return logits, values, new_hidden

        def init_hidden(self):
            import torch
            h = torch.zeros(1, 1, HIDDEN_DIM)
            c = torch.zeros(1, 1, HIDDEN_DIM)
            return (h, c)

    model = ActorCriticLSTM()

    
    directory = os.path.dirname(os.path.abspath(__file__))
    weight_names = ["weights.pth", "fast_best.pth", "fast_final.pth"]

    for fname in weight_names:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            import torch
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            _MODEL = model
            print(f"[Method7] Loaded weights from {fname}")
            return

    
    model.eval()
    _MODEL = model
    print("[Method7] WARNING: No weights file found, using random policy")


def _reset_episode():
    global _HIDDEN, _PREV_ACTION, _STEP, _PREV_OBS, _ESC
    _PREV_ACTION = 2     
    _STEP = 0
    _PREV_OBS = None
    _ESC = EscapeController()
    if _MODEL is not None:
        _HIDDEN = _MODEL.init_hidden()
    else:
        _HIDDEN = None



def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Main policy function.

    Args:
        obs: numpy array of shape (18,) — binary sensor readings
             obs[0:16]  = sonar sensors (8 pairs of far/near)
             obs[16]    = IR sensor (box very close)
             obs[17]    = stuck flag (collision with wall/boundary)
        rng: numpy random generator (for reproducibility)

    Returns:
        action string: one of "L45", "L22", "FW", "R22", "R45"
    """
    import torch
    global _HIDDEN, _PREV_ACTION, _STEP, _PREV_OBS, _ESC

    
    _load_model()

    
    if _PREV_OBS is None or _STEP >= 1005:
        _reset_episode()

    _STEP += 1
    _PREV_OBS = obs.copy()

   
    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    action_onehot[_PREV_ACTION] = 1.0
    obs_aug = np.concatenate([obs.astype(np.float32), action_onehot])

   
    stuck = bool(obs[17])
    if stuck and not _ESC.active:
        _ESC.trigger(obs)

  
    if _ESC.active:
        action_idx = _ESC.get_action()

        with torch.no_grad():
            x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0)
            _, _, _HIDDEN = _MODEL(x, _HIDDEN)

        _PREV_ACTION = action_idx
        return ACTIONS[action_idx]

    
    with torch.no_grad():
        x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0)
        logits, _, _HIDDEN = _MODEL(x, _HIDDEN)
        action_idx = logits.squeeze().argmax().item()

    _PREV_ACTION = action_idx
    return ACTIONS[action_idx]