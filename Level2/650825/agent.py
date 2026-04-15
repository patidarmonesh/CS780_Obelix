

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS    = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS  = 5
OBS_DIM    = 18
STACK_SIZE = 4
BELIEF_DIM = 4
INPUT_DIM  = STACK_SIZE * OBS_DIM + BELIEF_DIM   

MAX_EP_STEPS = 1000        

DEVICE = torch.device("cpu")   



class DuelingDQN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_actions=N_ACTIONS,
                 hidden=256, hidden2=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),   nn.ReLU(),
            nn.Linear(hidden,    hidden2),  nn.ReLU(),
            nn.Linear(hidden2, hidden2//2), nn.ReLU(),
        )
        mid = hidden2 // 2
        self.value     = nn.Sequential(nn.Linear(mid, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(mid, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        f = self.shared(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)



class InferenceController:
    """
    Identical mode logic to HybridController in train.py.
    update_belief() uses only obs (reward not available at inference).
    select_action() is fully greedy — no epsilon.
    """
    ESCAPE_SEQ = [0, 0, 0, 0, 2, 2, 2]  

    def reset(self):
        self.mode            = "SEARCH"
        self.escape_step     = 0
        self.probe_step      = 0
        self.pursuit_steps   = 0
        self.last_box_dir    = 2
        self.steps_since_box = 0
        self.attach_conf     = 0.0
        self._ir_consec      = 0

    def __init__(self):
        self.reset()

    def update_belief(self, obs: np.ndarray):
        """Obs-only: no reward available at inference time."""
        if obs[16] == 1 and obs[17] == 0:
            self._ir_consec += 1
            bonus = 0.35 if self._ir_consec >= 3 else 0.10
            self.attach_conf = min(1.0, self.attach_conf + bonus)
        else:
            self._ir_consec = 0
            if obs[17] == 1:                # stuck = wall hit, NOT box
                self.attach_conf = max(0.0, self.attach_conf - 0.25)
            else:
                self.attach_conf = max(0.0, self.attach_conf * 0.92)

        if   any(obs[0:4]):    self.last_box_dir = 1   # L22
        elif any(obs[4:12]):   self.last_box_dir = 2   # FW
        elif any(obs[12:16]):  self.last_box_dir = 3   # R22

        if any(obs[:16]):
            self.steps_since_box = 0
        else:
            self.steps_since_box = min(self.steps_since_box + 1, 50)

    def get_belief_vec(self) -> np.ndarray:
        return np.array([
            float(self.attach_conf),
            float(min(self.steps_since_box, 35) / 35.0),
            float(self.last_box_dir == 1),
            float(self.last_box_dir == 3),
        ], dtype=np.float32)

    def select_action(self, obs: np.ndarray, aug_state: np.ndarray,
                      model: DuelingDQN) -> str:
        stuck    = bool(obs[17] == 1)
        ir_on    = bool(obs[16] == 1) and not stuck
        any_sens = bool(any(obs[:17]))
        is_blind = (not any_sens) and (self.steps_since_box < 35) and (self.attach_conf < 0.6)

        
        if stuck and self.mode != "ESCAPE":
            self.mode        = "ESCAPE"
            self.escape_step = 0

        elif self.attach_conf > 0.6 and self.mode not in ("ESCAPE",):
            self.mode = "PUSH"

        elif (ir_on
              and self.probe_step == 0
              and self.attach_conf < 0.4
              and self.mode not in ("ESCAPE", "PROBE")):
            self.mode       = "PROBE"
            self.probe_step = 1

        elif is_blind and self.mode not in ("ESCAPE", "PUSH", "PROBE"):
            self.mode          = "PURSUIT"
            self.pursuit_steps = 0

       
        if (self.mode == "ESCAPE"
                and not stuck
                and self.escape_step >= len(self.ESCAPE_SEQ)):
            self.mode        = "SEARCH"
            self.escape_step = 0

        if self.mode == "PURSUIT" and (not is_blind or self.pursuit_steps > 35):
            self.mode = "SEARCH"

        
        if self.mode == "ESCAPE":
            idx               = min(self.escape_step, len(self.ESCAPE_SEQ) - 1)
            action_idx        = self.ESCAPE_SEQ[idx]
            self.escape_step += 1

        elif self.mode == "PROBE":
            if self.probe_step == 1:
                action_idx      = 1           
                self.probe_step = 2
            elif self.probe_step == 2:
                if not obs[16]:               
                    action_idx = 3            
                    self.mode  = "PURSUIT"
                else:                        
                    action_idx = 4            
                    self.mode  = "SEARCH"
                self.probe_step = 0
            else:
                self.probe_step = 0
                self.mode       = "SEARCH"
                action_idx      = 2           

        elif self.mode == "PURSUIT":
            action_idx          = self.last_box_dir
            self.pursuit_steps += 1

        else:  
            with torch.no_grad():
                t = torch.FloatTensor(aug_state).unsqueeze(0)
                action_idx = int(model(t).argmax(1).item())

        return ACTIONS[action_idx]



_model      = None
_obs_stack  = deque(maxlen=STACK_SIZE)
_ctrl       = InferenceController()
_step_count = 0


def _load_model():
    global _model
    if _model is not None:
        return
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "dqn_obelix_l2.pt")
    ckpt = torch.load(path, map_location="cpu")

    in_dim  = ckpt.get("input_dim",  INPUT_DIM)   
    hidden  = ckpt.get("hidden_dim",  256)
    hidden2 = ckpt.get("hidden_dim2", 128)

    net = DuelingDQN(in_dim, N_ACTIONS, hidden, hidden2)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    _model = net
    print(f"[agent] Loaded {path}  input_dim={in_dim}")


def _make_aug_state() -> np.ndarray:
    stack = list(_obs_stack)
    while len(stack) < STACK_SIZE:
        stack.insert(0, np.zeros(OBS_DIM, dtype=np.float32))
    base = np.concatenate(stack[-STACK_SIZE:]).astype(np.float32)
    return np.concatenate([base, _ctrl.get_belief_vec()])


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _step_count, _model, _ctrl, _obs_stack

    _load_model()

    _step_count += 1
    if _step_count > MAX_EP_STEPS:
        _step_count = 1
        _obs_stack  = deque(maxlen=STACK_SIZE)
        _ctrl.reset()

   
    _obs_stack.append(obs.astype(np.float32))
    _ctrl.update_belief(obs)

    aug_state = _make_aug_state()
    return _ctrl.select_action(obs, aug_state, _model)
