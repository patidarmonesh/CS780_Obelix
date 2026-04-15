"""
METHOD 4: Pure End-to-End PPO-LSTM — The Agent Learns EVERYTHING
=================================================================
Reference: "Recurrent Model-Free RL is a Strong Baseline for Many POMDPs"
           (Ni et al., 2021, arXiv:2110.05038)

PHILOSOPHY:
  No escape controller. No push detection. No wall/box discrimination.
  No reward shaping. The agent receives raw observations and raw rewards,
  and must learn to explore, approach, avoid walls, push, and recover
  from stuck — ALL by itself.

WHY THIS CAN WORK:
  The reward function is actually well-designed for learning:
    -200/step when stuck    → agent learns "don't keep going forward into walls"
    -1/step baseline        → agent learns "be efficient"
    +100 attachment bonus   → agent learns "approach and contact the box"
    +2000 terminal bonus    → agent learns "push box to boundary"
  The LSTM provides memory to distinguish wall from box via temporal patterns.

KEY TRICK (from POMDP literature):
  Feed [obs, prev_action_onehot, prev_reward_scaled] as input.
  This gives the LSTM the action-observation HISTORY it needs.
  The agent can learn: "I went FW and got -200 → I was stuck → turn"
  vs "I went FW and got -1 → I'm exploring → keep going"

Architecture:
  Input: obs(18) + prev_action(5 one-hot) + prev_reward(1 scaled) = 24
  Network: Linear(128) → ReLU → LSTM(256) → [Actor(5), Critic(1)]

Training: GPU   |   Inference: CPU
"""

import sys, os, csv, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "..", "env"),
           os.path.join(_HERE, "..", "..", "env"), os.path.join(_HERE, "..", "..", "..", "env")]:
    if os.path.exists(os.path.join(_c, "obelix.py")):
        sys.path.insert(0, _c); break

from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
OBS_DIM   = 18
AUG_DIM   = OBS_DIM + N_ACTIONS   # 18 + 5 = 23 (obs + prev_action_onehot)
# NOTE: prev_reward removed — not available in policy() at eval time
# obs[17] (stuck flag) already encodes the key reward information

# Architecture
HIDDEN_DIM  = 256

# PPO
LR            = 2.5e-4
GAMMA         = 0.99       # shorter horizon — learn stuck penalty fast
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.05       # HIGH — agent needs to explore without handcoded walk
VALUE_COEF    = 0.5
MAX_GRAD      = 0.5
PPO_EPOCHS    = 4
SEQ_LEN       = 32         # LSTM training chunk length
REWARD_SCALE  = 200.0      # divide rewards by this → -200 becomes -1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Actor-Critic LSTM (end-to-end)
# ══════════════════════════════════════════════════════════════
class PureLSTMPolicy(nn.Module):
    """
    No feature engineering. Raw obs + prev_action + prev_reward → LSTM → actions.
    The LSTM IS the agent's entire brain — escape, approach, push, everything.
    """
    def __init__(self, input_dim=AUG_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Initialize weights with smaller values for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m.out_features == n_actions else np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        """x: (batch, seq, input_dim), hidden: (h,c) each (1, batch, hidden_dim)"""
        enc = self.encoder(x)
        lstm_out, new_hidden = self.lstm(enc, hidden)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, new_hidden

    def init_hidden(self, batch=1):
        return (torch.zeros(1, batch, self.hidden_dim, device=DEVICE),
                torch.zeros(1, batch, self.hidden_dim, device=DEVICE))

    def get_action(self, obs_aug, hidden):
        """Single-step inference for rollout collection."""
        x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits, value, new_hidden = self(x, hidden)
        dist = Categorical(logits=logits.squeeze(0).squeeze(0))
        action = dist.sample()
        return (action.item(), dist.log_prob(action).item(),
                value.squeeze().item(), new_hidden)


# ══════════════════════════════════════════════════════════════
# Augmented Observation Builder
# ══════════════════════════════════════════════════════════════
def augment_obs(obs, prev_action):
    """
    Build augmented input: [obs(18), prev_action_onehot(5)] = 23 dims.
    prev_action tells the LSTM what it did last step, helping it learn:
      "I went FW and now I'm stuck → wall" vs "I went FW and sensors appeared → box"
    """
    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    action_onehot[prev_action] = 1.0
    return np.concatenate([obs.astype(np.float32), action_onehot])


# ══════════════════════════════════════════════════════════════
# Rollout Storage (episode-aware, stores LSTM hidden states)
# ══════════════════════════════════════════════════════════════
class RolloutStorage:
    def __init__(self):
        self.obs = []          # augmented observations
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []        # episode termination flags
        self.hiddens = []      # LSTM hidden states at each step

    def push(self, obs_aug, action, log_prob, reward, value, done, hidden):
        self.obs.append(obs_aug)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward / REWARD_SCALE)   # scale rewards
        self.values.append(value)
        self.dones.append(float(done))
        # Detach hidden state for storage
        self.hiddens.append((hidden[0].detach().cpu(), hidden[1].detach().cpu()))

    def compute_gae(self, last_value):
        """Compute GAE advantages and returns."""
        values = self.values + [last_value]
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def get_batches(self, advantages, returns):
        """
        Split rollout into sequences of SEQ_LEN for LSTM training.
        Returns list of (obs_seq, act_seq, old_lp_seq, adv_seq, ret_seq, hidden_init)
        """
        N = len(self.obs)
        batches = []
        for start in range(0, N - SEQ_LEN + 1, SEQ_LEN):
            end = start + SEQ_LEN
            obs_seq = np.array(self.obs[start:end])
            act_seq = np.array(self.actions[start:end])
            lp_seq  = np.array(self.log_probs[start:end])
            adv_seq = np.array(advantages[start:end])
            ret_seq = np.array(returns[start:end])
            h_init  = self.hiddens[start]  # LSTM state at sequence start
            batches.append((obs_seq, act_seq, lp_seq, adv_seq, ret_seq, h_init))
        return batches

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()
        self.hiddens.clear()

    def __len__(self):
        return len(self.obs)


# ══════════════════════════════════════════════════════════════
# PPO Update
# ══════════════════════════════════════════════════════════════
def ppo_update(model, optimizer, storage, last_value):
    advantages, returns = storage.compute_gae(last_value)

    # Normalize advantages globally
    adv_arr = np.array(advantages)
    adv_mean, adv_std = adv_arr.mean(), adv_arr.std()
    advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]

    batches = storage.get_batches(advantages, returns)
    if not batches:
        return 0.0

    total_loss = 0
    n_updates = 0

    for epoch in range(PPO_EPOCHS):
        random.shuffle(batches)
        for obs_seq, act_seq, lp_seq, adv_seq, ret_seq, h_init in batches:
            obs_t = torch.FloatTensor(obs_seq).unsqueeze(0).to(DEVICE)  # (1, seq, 24)
            act_t = torch.LongTensor(act_seq).to(DEVICE)
            old_lp_t = torch.FloatTensor(lp_seq).to(DEVICE)
            adv_t = torch.FloatTensor(adv_seq).to(DEVICE)
            ret_t = torch.FloatTensor(ret_seq).to(DEVICE)

            # Restore hidden state
            h = (h_init[0].to(DEVICE), h_init[1].to(DEVICE))

            logits, values, _ = model(obs_t, h)
            logits = logits.squeeze(0)   # (seq, 5)
            values = values.squeeze(0).squeeze(-1)  # (seq,)

            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_t)
            entropy = dist.entropy()

            # PPO clipped objective
            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = ((values - ret_t) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

    return total_loss / max(n_updates, 1)


# ══════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════
def train(
    eps0=4000, eps1=5000, eps2=4000, eps3=12000,
    max_steps=1000, seed=42, prefix="pure",
    rollout_episodes=4,   # episodes per PPO update
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = PureLSTMPolicy().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    best_path  = os.path.join(_HERE, f"{prefix}_best.pth")
    final_path = os.path.join(_HERE, f"{prefix}_final.pth")
    log_path   = os.path.join(_HERE, f"{prefix}_rewards.csv")

    stages = [
        (eps0, 0, False, 0, "S0:NoWall"),
        (eps1, 0, True,  0, "S1:Wall  "),
        (eps2, 2, True,  0, "S2:Blink "),
        (eps3, 3, True,  2, "S3:Move  "),
    ]

    all_rewards = []
    best_mean = -np.inf
    t_start = time.time()

    for stage_idx, (n_eps, diff, wall, box_spd, name) in enumerate(stages):
        print(f"\n{'═'*65}")
        print(f" {name} | {n_eps} episodes | diff={diff} | wall={wall}")
        print(f" PURE END-TO-END — no controllers, no shaping")
        print(f"{'═'*65}")

        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=box_spd, seed=seed)
        stage_rewards = []

        ep = 0
        while ep < n_eps:
            # Collect rollout_episodes episodes before updating
            storage = RolloutStorage()
            batch_rewards = []

            for _ in range(rollout_episodes):
                if ep >= n_eps:
                    break

                ep_seed = np.random.randint(0, 300_000)
                obs = env.reset(seed=ep_seed)
                hidden = model.init_hidden(1)

                prev_action = 2    # start with FW
                prev_reward = 0.0
                total_reward = 0.0

                for step in range(max_steps):
                    # Build augmented observation
                    obs_aug = augment_obs(obs, prev_action)

                    # Get action from policy
                    action_idx, log_prob, value, new_hidden = model.get_action(obs_aug, hidden)

                    # Step environment (RAW reward, no shaping!)
                    next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)

                    # Store transition
                    storage.push(obs_aug, action_idx, log_prob, reward, value, done, hidden)

                    total_reward += reward
                    prev_action = action_idx
                    prev_reward = reward
                    obs = next_obs
                    hidden = new_hidden

                    if done:
                        break

                all_rewards.append(total_reward)
                stage_rewards.append(total_reward)
                batch_rewards.append(total_reward)
                ep += 1

            # PPO update
            if len(storage) > SEQ_LEN:
                # Get last value estimate for GAE
                with torch.no_grad():
                    last_aug = augment_obs(obs, prev_action)
                    x = torch.FloatTensor(last_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    _, last_v, _ = model(x, hidden)
                    last_value = last_v.squeeze().item() / 1.0  # already scaled in storage

                avg_loss = ppo_update(model, optimizer, storage, last_value)

            # Logging
            if ep % 200 < rollout_episodes or ep == n_eps:
                m = np.mean(stage_rewards[-200:]) if len(stage_rewards) >= 200 else np.mean(stage_rewards)
                succ = sum(1 for r in stage_rewards[-200:] if r > 500)
                elapsed = time.time() - t_start
                print(f"  [{name} Ep {ep:>5}/{n_eps}] Mean:{m:>9.1f} | "
                      f"Succ:{succ:3d}/200 | "
                      f"Time:{elapsed/60:>5.1f}m")
                if m > best_mean:
                    best_mean = m
                    torch.save(model.state_dict(), best_path)
                    print(f"    💾 New best: {m:.1f}")

    # Final save
    torch.save(model.state_dict(), final_path)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "reward"])
        for i, r in enumerate(all_rewards):
            w.writerow([i+1, f"{r:.2f}"])

    total_time = (time.time() - t_start) / 60
    total_succ = sum(1 for r in all_rewards if r > 500)
    print(f"\n{'═'*65}")
    print(f" TRAINING COMPLETE ({total_time:.1f} minutes)")
    print(f"{'═'*65}")
    print(f"  Best 200-ep mean : {best_mean:.1f}")
    print(f"  Total success    : {total_succ}/{len(all_rewards)}")
    print(f"  Best weights     : {best_path}")
    print(f"  Final weights    : {final_path}")
    return all_rewards


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PURE END-TO-END PPO-LSTM")
    p.add_argument("--eps0", type=int, default=4000)
    p.add_argument("--eps1", type=int, default=5000)
    p.add_argument("--eps2", type=int, default=4000)
    p.add_argument("--eps3", type=int, default=12000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="pure")
    p.add_argument("--rollout_episodes", type=int, default=4)
    train(**vars(p.parse_args()))