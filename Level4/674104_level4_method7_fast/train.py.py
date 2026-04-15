import sys, os, csv, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [
    _HERE,
    os.path.join(_HERE, "env"),
    os.path.join(_HERE, "..", "env"),
    os.path.join(_HERE, "..", "..", "env"),
    os.path.join(_HERE, "..", "..", "..", "env"),
]:
    if os.path.exists(os.path.join(_c, "obelix.py")):
        sys.path.insert(0, _c)
        break

from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
OBS_DIM   = 18
AUG_DIM   = OBS_DIM + N_ACTIONS

HIDDEN_DIM    = 128
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.03
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS    = 4
MINIBATCH     = 64
ROLLOUT_EPS   = 8
REWARD_SCALE  = 100.0

ESC_TURNS = 4
ESC_FW    = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EscapeController:
    TOTAL_STEPS = ESC_TURNS + ESC_FW

    def __init__(self):
        self.active = False
        self._step = 0
        self._turn_action = 4

    def trigger(self, obs):
        left_sensors  = sum(int(obs[i]) for i in range(0, 4))
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


class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim=AUG_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
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

        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module.out_features == n_actions else np.sqrt(2)
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x, hidden):
        encoded = self.encoder(x)
        lstm_out, new_hidden = self.lstm(encoded, hidden)
        return self.actor(lstm_out), self.critic(lstm_out), new_hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        return (h, c)

    def get_action_and_value(self, obs_aug, hidden):
        x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits, value, new_hidden = self(x, hidden)
        dist   = Categorical(logits=logits.squeeze(0).squeeze(0))
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.squeeze().item(), new_hidden


def build_augmented_obs(obs, prev_action):
    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    action_onehot[prev_action] = 1.0
    return np.concatenate([obs.astype(np.float32), action_onehot])


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    values_extended = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t + 1] * (1.0 - dones[t]) - values_extended[t]
        gae   = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(model, optimizer, rollout_data):
    obs_t    = torch.FloatTensor(np.array(rollout_data["obs"])).to(DEVICE)
    act_t    = torch.LongTensor(rollout_data["actions"]).to(DEVICE)
    old_lp_t = torch.FloatTensor(rollout_data["log_probs"]).to(DEVICE)
    adv_t    = torch.FloatTensor(rollout_data["advantages"]).to(DEVICE)
    ret_t    = torch.FloatTensor(rollout_data["returns"]).to(DEVICE)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    N = len(obs_t)
    total_loss = 0.0

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(N, device=DEVICE)
        for start in range(0, N, MINIBATCH):
            mb_idx = indices[start : min(start + MINIBATCH, N)]

            hidden = model.init_hidden(len(mb_idx))
            logits, values, _ = model(obs_t[mb_idx].unsqueeze(1), hidden)
            logits = logits.squeeze(1)
            values = values.squeeze(1).squeeze(-1)

            dist        = Categorical(logits=logits)
            new_lp      = dist.log_prob(act_t[mb_idx])
            entropy     = dist.entropy()
            ratio       = torch.exp(new_lp - old_lp_t[mb_idx])
            policy_loss = -torch.min(
                ratio * adv_t[mb_idx],
                torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[mb_idx],
            ).mean()
            value_loss  = ((values - ret_t[mb_idx]) ** 2).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()

    return total_loss


def collect_episode(env, model, max_steps, ep_seed):
    obs    = env.reset(seed=ep_seed)
    hidden = model.init_hidden(1)
    esc    = EscapeController()
    prev_action   = 2
    rollout_steps = []
    total_reward  = 0.0

    for _ in range(max_steps):
        obs_aug = build_augmented_obs(obs, prev_action)
        stuck   = bool(obs[17])

        if stuck and not esc.active:
            esc.trigger(obs)

        if esc.active:
            action_idx = esc.get_action()
            with torch.no_grad():
                x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
                _, _, hidden = model(x, hidden)
            is_lstm_step = False
        else:
            action_idx, log_prob, value, hidden = model.get_action_and_value(obs_aug, hidden)
            rollout_steps.append({"obs_aug": obs_aug, "action": action_idx,
                                   "log_prob": log_prob, "value": value})
            is_lstm_step = True

        next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
        total_reward += reward

        if is_lstm_step:
            rollout_steps[-1]["reward"] = reward / REWARD_SCALE
            rollout_steps[-1]["done"]   = float(done)

        prev_action = action_idx
        obs = next_obs
        if done:
            break

    for s in rollout_steps:
        if "reward" not in s:
            s["reward"] = -1.0 / REWARD_SCALE
            s["done"]   = 0.0

    return rollout_steps, total_reward


def train(eps0=1500, eps1=3000, eps2=1500, eps3=4000, seed=42, prefix="fast"):
    print(f"Device: {DEVICE} | Method 7 | LSTM hidden: {HIDDEN_DIM}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model     = ActorCriticLSTM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_path  = os.path.join(_HERE, f"{prefix}_best.pth")
    final_path = os.path.join(_HERE, f"{prefix}_final.pth")
    log_path   = os.path.join(_HERE, f"{prefix}_rewards.csv")

    stages = [
        (eps0, 0, False, 0, 500,  "S0:NoWall", False),
        (eps1, 0, True,  0, 1000, "S1:Wall",   True),
        (eps2, 2, True,  0, 1000, "S2:Blink",  False),
        (eps3, 3, True,  2, 1000, "S3:Move",   False),
    ]

    all_rewards = []
    best_mean   = -np.inf
    t_start     = time.time()

    for _, (n_eps, diff, wall, box_spd, max_steps, name, mix_nowall) in enumerate(stages):
        print(f"\n{name} | {n_eps} eps | diff={diff} | wall={wall} | max_steps={max_steps}"
              + (" | 50% mixed" if mix_nowall else ""))

        env_wall = OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                          wall_obstacles=wall, difficulty=diff, box_speed=box_spd, seed=seed)
        env_nowall = (OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                             wall_obstacles=False, difficulty=diff, box_speed=box_spd, seed=seed)
                      if mix_nowall else None)

        stage_rewards = []
        ep = 0

        while ep < n_eps:
            combined_rollout = {"obs": [], "actions": [], "log_probs": [],
                                "rewards": [], "values": [], "dones": []}

            for _ in range(ROLLOUT_EPS):
                if ep >= n_eps:
                    break

                use_env = (env_nowall if env_nowall and np.random.random() < 0.5 else env_wall)
                ep_seed = np.random.randint(0, 300_000)
                steps, total_reward = collect_episode(use_env, model, max_steps, ep_seed)

                for s in steps:
                    combined_rollout["obs"].append(s["obs_aug"])
                    combined_rollout["actions"].append(s["action"])
                    combined_rollout["log_probs"].append(s["log_prob"])
                    combined_rollout["rewards"].append(s["reward"])
                    combined_rollout["values"].append(s["value"])
                    combined_rollout["dones"].append(s["done"])

                all_rewards.append(total_reward)
                stage_rewards.append(total_reward)
                ep += 1

            if len(combined_rollout["obs"]) > MINIBATCH:
                advantages, returns = compute_gae(combined_rollout["rewards"],
                                                  combined_rollout["values"],
                                                  combined_rollout["dones"])
                combined_rollout["advantages"] = advantages
                combined_rollout["returns"]    = returns
                ppo_update(model, optimizer, combined_rollout)

            if ep % 200 < ROLLOUT_EPS or ep >= n_eps:
                window = stage_rewards[-200:]
                mean_r = np.mean(window) if window else 0
                succ   = sum(1 for r in window if r > 500)
                print(f"  [{name} {ep:>5}/{n_eps}] Mean:{mean_r:>9.1f} | "
                      f"Succ:{succ:3d}/200 | Time:{(time.time()-t_start)/60:.1f}m")

                if mean_r > best_mean:
                    best_mean = mean_r
                    torch.save(model.state_dict(), best_path)
                    print(f"    New best: {mean_r:.1f}")

    torch.save(model.state_dict(), final_path)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(all_rewards):
            writer.writerow([i + 1, f"{r:.2f}"])

    total_time = (time.time() - t_start) / 60.0
    total_succ = sum(1 for r in all_rewards if r > 500)
    print(f"\nDone | {total_time:.1f}m | Best mean: {best_mean:.1f} | "
          f"Success: {total_succ}/{len(all_rewards)} ({100*total_succ/max(len(all_rewards),1):.1f}%)")
    return all_rewards


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--eps0",   type=int, default=1500)
    p.add_argument("--eps1",   type=int, default=3000)
    p.add_argument("--eps2",   type=int, default=1500)
    p.add_argument("--eps3",   type=int, default=4000)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--prefix", type=str, default="fast")
    train(**vars(p.parse_args()))