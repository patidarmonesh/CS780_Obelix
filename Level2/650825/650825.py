"""
Hybrid DDQN for OBELIX Level-2
Architecture : Dueling Double DQN, 4-frame stack + 4-dim belief vec (INPUT=76)
Controller   : HybridController with ESCAPE / PROBE / PURSUIT / PUSH / SEARCH
Curriculum   : 4 stages — static→static+wall→blink→blink+wall
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from obelix import OBELIX

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")

# ── Environment ───────────────────────────────────────────────────────────────
ACTIONS    = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS  = 5
MAX_STEPS  = 1000
ARENA_SIZE = 500
SCALE      = 5

# ── Observation / Belief dims ─────────────────────────────────────────────────
OBS_DIM    = 18
STACK_SIZE = 4
BELIEF_DIM = 4                               # 4 hand-crafted floats
INPUT_DIM  = STACK_SIZE * OBS_DIM + BELIEF_DIM  # 76

# ── Network ───────────────────────────────────────────────────────────────────
HIDDEN_DIM  = 256
HIDDEN_DIM2 = 128

# ── Replay Buffer ─────────────────────────────────────────────────────────────
BUFFER_SIZE = 100_000
BATCH_SIZE  = 128
MIN_REPLAY  = 1_000

# ── Learning ──────────────────────────────────────────────────────────────────
LR          = 3e-4
GAMMA       = 0.99
TAU         = 0.005
GRAD_CLIP   = 10.0
UPDATE_FREQ = 4

# ── Exploration (per stage) ───────────────────────────────────────────────────
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY_EPS = 2000      # episodes until EPS_END, reset each stage

# ── Curriculum ────────────────────────────────────────────────────────────────
# (n_episodes, difficulty, wall_obstacles)
STAGES = [
    (2000, 0, False),   # Stage 1: static box, no wall   → learn approach/attach
    (2000, 0, True),    # Stage 2: static box + wall     → probe + wall avoidance
    (2000, 2, False),   # Stage 3: blinking, no wall     → object permanence
    (4000, 2, True),    # Stage 4: blinking + wall       → full Level 2
]

REWARD_SCALE = 200.0


# ─── Dueling DQN ─────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_actions=N_ACTIONS,
                 hidden=HIDDEN_DIM, hidden2=HIDDEN_DIM2):
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


# ─── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros(capacity, dtype=np.int64)
        self.r  = np.zeros(capacity, dtype=np.float32)
        self.d  = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, ns, done):
        self.s[self.ptr]  = s
        self.ns[self.ptr] = ns
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.d[self.ptr]  = float(done)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, batch)
        return (
            torch.FloatTensor(self.s[idx]).to(DEVICE),
            torch.LongTensor(self.a[idx]).to(DEVICE),
            torch.FloatTensor(self.r[idx]).to(DEVICE),
            torch.FloatTensor(self.ns[idx]).to(DEVICE),
            torch.FloatTensor(self.d[idx]).to(DEVICE),
        )

    def __len__(self): return self.size


# ─── Hybrid Controller ────────────────────────────────────────────────────────
class HybridController:
    """
    Wraps DDQN and intercepts for 5 modes:

      ESCAPE   obs[17]==1: L45×4 (180°) then FW×3       — no Q-update
      PROBE    IR fires, not attached: 2-step wiggle      — no Q-update
               L22 → if IR drops = BOX → enter PURSUIT
                     if IR stays = WALL → rotate away
      PURSUIT  box blinked off: keep heading last dir     — Q-update OK
      PUSH     attach_conf > 0.6: DDQN low-epsilon        — Q-update OK
      SEARCH   default: DDQN full epsilon                 — Q-update OK
    """
    ESCAPE_SEQ = [0, 0, 0, 0, 2, 2, 2]   # L45 ×4 = 180°, then FW ×3

    def __init__(self, online_net: DuelingDQN, device):
        self.net    = online_net
        self.device = device
        self.reset_episode()

    def reset_episode(self):
        self.mode            = "SEARCH"
        self.escape_step     = 0
        self.probe_step      = 0
        self.pursuit_steps   = 0
        self.last_box_dir    = 2     # FW default
        self.steps_since_box = 0
        self.attach_conf     = 0.0
        self._ir_consec      = 0    # consecutive IR=1, stuck=0

    def update_belief(self, obs: np.ndarray, reward: float = 0.0):
        """Call every step with the NEW obs and the reward just received."""

        # ── Attachment confidence ─────────────────────────────────────────────
        if reward >= 100:
            self.attach_conf = 1.0          # +100 = definitive attach signal
            self._ir_consec  = 10
        elif obs[16] == 1 and obs[17] == 0:
            self._ir_consec  += 1
            bonus = 0.35 if self._ir_consec >= 3 else 0.10
            self.attach_conf = min(1.0, self.attach_conf + bonus)
        else:
            self._ir_consec = 0
            if obs[17] == 1:                # stuck = wall, NOT box
                self.attach_conf = max(0.0, self.attach_conf - 0.25)
            else:
                self.attach_conf = max(0.0, self.attach_conf * 0.92)

        # ── Last known box direction (sonar only — IR alone unreliable) ───────
        if   any(obs[0:4]):    self.last_box_dir = 1   # L22 toward left
        elif any(obs[4:12]):   self.last_box_dir = 2   # FW  toward front
        elif any(obs[12:16]):  self.last_box_dir = 3   # R22 toward right

        # ── Steps since sonar active ──────────────────────────────────────────
        if any(obs[:16]):
            self.steps_since_box = 0
        else:
            self.steps_since_box = min(self.steps_since_box + 1, 50)

    def get_belief_vec(self) -> np.ndarray:
        """4 floats appended to frame-stacked obs to break perceptual aliasing."""
        return np.array([
            float(self.attach_conf),
            float(min(self.steps_since_box, 35) / 35.0),  # 0→1 as box goes dark
            float(self.last_box_dir == 1),                 # box was on left
            float(self.last_box_dir == 3),                 # box was on right
        ], dtype=np.float32)

    def select_action(self, obs: np.ndarray, aug_state: np.ndarray,
                      epsilon: float, rng) -> tuple:
        """
        Returns (action_idx, update_q, mode_str)
        update_q=False → don't push to replay buffer (hardcoded step).
        """
        stuck    = bool(obs[17] == 1)
        ir_on    = bool(obs[16] == 1) and not stuck
        any_sens = bool(any(obs[:17]))
        is_blind = (not any_sens) and (self.steps_since_box < 35) and (self.attach_conf < 0.6)

        # ── Mode transitions (priority order) ─────────────────────────────────
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

        # ── Exit conditions ────────────────────────────────────────────────────
        if (self.mode == "ESCAPE"
                and not stuck
                and self.escape_step >= len(self.ESCAPE_SEQ)):
            self.mode        = "SEARCH"
            self.escape_step = 0

        if self.mode == "PURSUIT" and (not is_blind or self.pursuit_steps > 35):
            self.mode = "SEARCH"

        # ── Execute mode ───────────────────────────────────────────────────────
        update_q = True

        if self.mode == "ESCAPE":
            idx               = min(self.escape_step, len(self.ESCAPE_SEQ) - 1)
            action_idx        = self.ESCAPE_SEQ[idx]
            self.escape_step += 1
            update_q          = False                        # don't corrupt Q

        elif self.mode == "PROBE":
            if self.probe_step == 1:
                action_idx      = 1                          # L22 probe wiggle
                self.probe_step = 2
                update_q        = False
            elif self.probe_step == 2:
                if not obs[16]:                              # IR dropped → BOX
                    action_idx = 3                           # R22 realign
                    self.mode  = "PURSUIT"
                else:                                        # IR stayed → WALL
                    action_idx = 4                           # R45 rotate away
                    self.mode  = "SEARCH"
                self.probe_step = 0
                update_q        = False
            else:
                self.probe_step = 0
                self.mode       = "SEARCH"
                action_idx      = 2

        elif self.mode == "PURSUIT":
            action_idx          = self.last_box_dir          # head to last known dir
            self.pursuit_steps += 1
            update_q            = True                       # fine to learn from

        else:   # SEARCH or PUSH
            eps_use = epsilon * (0.3 if self.mode == "PUSH" else 1.0)
            if rng.random() < eps_use:
                action_idx = rng.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    t = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
                    action_idx = int(self.net(t).argmax(1).item())
            update_q = True

        return action_idx, update_q, self.mode


# ─── Helpers ──────────────────────────────────────────────────────────────────
def make_aug_state(obs_stack: deque, belief_vec: np.ndarray) -> np.ndarray:
    stack = list(obs_stack)
    while len(stack) < STACK_SIZE:
        stack.insert(0, np.zeros(OBS_DIM, dtype=np.float32))
    return np.concatenate(stack[-STACK_SIZE:] + [belief_vec]).astype(np.float32)


def get_epsilon(stage_ep: int) -> float:
    frac = min(1.0, stage_ep / EPS_DECAY_EPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def ddqn_update(online, target, opt, buf, scaler):
    s, a, r, ns, d = buf.sample(BATCH_SIZE)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        q_cur = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = online(ns).argmax(1, keepdim=True)
            q_next = target(ns).gather(1, best_a).squeeze(1)
            q_tgt  = r + GAMMA * q_next * (1.0 - d)
        loss = nn.SmoothL1Loss()(q_cur, q_tgt)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
    scaler.step(opt)
    scaler.update()
    return float(loss.item())


def soft_update(online, target):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)


# ─── Training loop ────────────────────────────────────────────────────────────
def train(stages=STAGES, max_steps=MAX_STEPS, seed=42, save_path="dqn_obelix_l2.pt"):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    online_net = DuelingDQN(INPUT_DIM, N_ACTIONS).to(DEVICE)
    target_net = DuelingDQN(INPUT_DIM, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    buffer    = ReplayBuffer(BUFFER_SIZE, INPUT_DIM)

    total_steps = 0
    best_mean   = -float("inf")
    all_rewards = []

    for stage_idx, (n_eps, difficulty, wall) in enumerate(stages):
        stage_eps     = 0               # epsilon resets each stage
        stage_rewards = []
        stage_losses  = []

        print(f"\n{'='*60}")
        print(f" STAGE {stage_idx+1}/4 | diff={difficulty} | wall={wall} | n_eps={n_eps}")
        print(f" INPUT_DIM={INPUT_DIM} | DEVICE={DEVICE}")
        print(f"{'='*60}")

        env = OBELIX(
            scaling_factor=SCALE,
            arena_size=ARENA_SIZE,
            max_steps=max_steps,
            wall_obstacles=wall,
            difficulty=difficulty,
            seed=seed,
        )

        for ep in range(n_eps):
            ep_seed   = int(rng.integers(0, 100_000))
            obs       = env.reset(seed=ep_seed)

            obs_stack = deque(maxlen=STACK_SIZE)
            obs_stack.append(obs.astype(np.float32))

            ctrl = HybridController(online_net, DEVICE)
            ctrl.update_belief(obs, reward=0.0)

            state     = make_aug_state(obs_stack, ctrl.get_belief_vec())
            epsilon   = get_epsilon(stage_eps)
            stage_eps += 1

            ep_reward = 0.0
            ep_loss   = 0.0
            n_updates = 0
            mode_counts = {"SEARCH": 0, "PROBE": 0, "PURSUIT": 0,
                           "PUSH": 0, "ESCAPE": 0}

            for step in range(max_steps):
                action_idx, update_q, mode = ctrl.select_action(
                    obs, state, epsilon, random
                )
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                action = ACTIONS[action_idx]

                next_obs, reward, done = env.step(action, render=False)
                ep_reward += reward

                ctrl.update_belief(next_obs, reward=reward)
                obs_stack.append(next_obs.astype(np.float32))
                next_state = make_aug_state(obs_stack, ctrl.get_belief_vec())

                scaled_r = float(reward) / REWARD_SCALE

                if update_q:                              # skip ESCAPE / PROBE steps
                    buffer.push(state, action_idx, scaled_r, next_state, done)

                total_steps += 1

                if len(buffer) >= MIN_REPLAY and total_steps % UPDATE_FREQ == 0:
                    loss      = ddqn_update(online_net, target_net, optimizer, buffer, scaler)
                    ep_loss  += loss
                    n_updates += 1
                    soft_update(online_net, target_net)

                obs   = next_obs
                state = next_state
                if done:
                    break

            all_rewards.append(ep_reward)
            stage_rewards.append(ep_reward)
            if n_updates > 0:
                stage_losses.append(ep_loss / n_updates)

            if (ep + 1) % 100 == 0:
                mean_r = np.mean(stage_rewards[-100:])
                mean_l = np.mean(stage_losses[-100:]) if stage_losses else 0.0
                prev_r = stage_rewards[-200:-100] if len(stage_rewards) > 100 else [mean_r - 1]
                trend  = "✅" if mean_r > np.mean(prev_r) else "⏳"
                buf_p  = 100 * len(buffer) / BUFFER_SIZE
                print(
                    f"[S{stage_idx+1} Ep {ep+1:>5}/{n_eps}] "
                    f"R: {mean_r:>9.1f} | ε: {epsilon:.4f} | "
                    f"L: {mean_l:.4f} | Buf: {buf_p:.0f}% {trend}"
                )

                if mean_r > best_mean:
                    best_mean = mean_r
                    torch.save({
                        "online_state_dict": online_net.state_dict(),
                        "target_state_dict": target_net.state_dict(),
                        "optimizer":         optimizer.state_dict(),
                        "total_steps":       total_steps,
                        "best_mean":         best_mean,
                        "input_dim":         INPUT_DIM,
                        "stack_size":        STACK_SIZE,
                        "obs_dim":           OBS_DIM,
                        "belief_dim":        BELIEF_DIM,
                        "n_actions":         N_ACTIONS,
                        "hidden_dim":        HIDDEN_DIM,
                        "hidden_dim2":       HIDDEN_DIM2,
                    }, save_path)
                    print(f"  💾 Saved  best={best_mean:.1f}")

    print(f"\n✅ Training done | Best mean reward: {best_mean:.1f}")
    return all_rewards


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--save_path",  type=str,  default="dqn_obelix_l2.pt")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--max_steps",  type=int,  default=MAX_STEPS)
    args = p.parse_args()
    train(save_path=args.save_path, seed=args.seed, max_steps=args.max_steps)
