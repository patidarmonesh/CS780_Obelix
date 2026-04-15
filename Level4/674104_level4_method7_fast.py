"""
METHOD 7: Fast Hybrid PPO-LSTM
================================
LESSONS FROM ALL PREVIOUS EXPERIMENTS:

  exp05/06/07 (tabular Q): Perceptual aliasing — wall & box identical sensors.
    → Need neural network with temporal memory (LSTM)

  Method 1 (DDQN+controllers): S0 best -1894, S1 crashed -11k to -20k
    → Controllers hijack actions, LSTM hidden state corrupted

  Method 2 (DRQN+controllers): S0 best -3479, S1 crashed -18k to -23k
    → Same problem: controllers + ε-greedy = poor

  Method 3 (PPO+controllers): S0 best -2793, S1 crashed -12k to -17k
    → Controllers override NN → NN never learns from those situations

  Method 4 (Pure PPO-LSTM): S0 best +76.7 (63% success!) S1 plateau -2500
    → WINNER on no-wall. But LSTM spent 3000 eps learning escape.
    → Then plateau: PPO on-policy, each transition used ~4x then discarded.
    → After 8000 S1 episodes (80 hours!), ZERO improvement.

  Key findings from papers:
    - Ni et al. (ICML 2022): off-policy >> on-policy (5-10x). GRU ≈ LSTM.
      Context length 32-64. prev_action as input. Separate actor/critic.
    - R2D2 (ICLR 2019): Burn-in trick. Stored states. n-step returns. PER.
    - DRQN (AAAI 2015): LSTM handles flickering naturally. End-to-end.
    - Reward shaping HURTS (confirmed in multiple experiments).

  THIS METHOD:
    Hardcode ONLY escape (saves 3000 episodes of wasted learning).
    LSTM handles: explore, approach, wall avoidance, push.
    Escape steps filtered from rollout (LSTM never sees -200 penalties).
    LSTM hidden state still updated during escape (consistency).
    Smaller LSTM (128), shorter S0 (500 max_steps), mixed S1 curriculum.
    Expected: 3-4 hours total (vs 80+ for Method 4).

  CODABENCH EVAL: ALL tests have walls! (difficulty 0,2,3 all with wall=True)
    → Wall performance is THE metric that matters.

Architecture:
  Input: obs(18) + prev_action_onehot(5) = 23
  Network: Linear(64) → ReLU → LSTM(128) → Actor(5) + Critic(1)
  ~47K params (vs Method 4's ~200K)

Training: GPU    Inference: CPU
"""

import sys, os, csv, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ══════════════════════════════════════════════════════════════
# Path setup — find obelix.py
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════
ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
OBS_DIM   = 18
AUG_DIM   = OBS_DIM + N_ACTIONS  # 23 = obs + prev_action_onehot

# ══════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════
HIDDEN_DIM    = 128       # Small = fast (Method 4 used 256)
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.03      # Encourage exploration
VALUE_COEF    = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS    = 4
MINIBATCH     = 64
ROLLOUT_EPS   = 8         # Episodes per PPO update
REWARD_SCALE  = 100.0     # Divide raw rewards by this

# Escape controller constants
ESC_TURNS     = 4         # 4 × 45° = 180° reversal
ESC_FW        = 4         # 4 × 5px = 20px clearance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# Escape Controller (the ONLY hardcoded behavior)
# ══════════════════════════════════════════════════════════════
class EscapeController:
    """
    When robot is stuck (obs[17]==1), execute fixed escape sequence:
      Step 1-4: Turn 180° (away from obstacle)
      Step 5-8: Move forward (clear the wall)

    WHY hardcode this:
      Method 4 spent 3000 episodes learning this 8-step sequence.
      Those episodes had mean -11678 (catastrophic).
      By hardcoding, the LSTM starts learning wall AVOIDANCE from ep 1.
    """
    TOTAL_STEPS = ESC_TURNS + ESC_FW  # 8

    def __init__(self):
        self.active = False
        self._step = 0
        self._turn_action = 4  # R45 default

    def trigger(self, obs):
        """Start escape. Turn away from whichever side has more sensors."""
        left_sensors = sum(int(obs[i]) for i in range(0, 4))
        right_sensors = sum(int(obs[i]) for i in range(12, 16))
        # Turn away from the side with more obstacle readings
        self._turn_action = 4 if left_sensors >= right_sensors else 0  # R45 or L45
        self._step = 0
        self.active = True

    def get_action(self):
        """Return next escape action. Deactivates when sequence complete."""
        i = self._step
        self._step += 1
        if self._step >= self.TOTAL_STEPS:
            self.active = False
        # First ESC_TURNS steps: turn. Then ESC_FW steps: forward.
        return self._turn_action if i < ESC_TURNS else 2  # 2 = FW


# ══════════════════════════════════════════════════════════════
# Neural Network: Actor-Critic with LSTM
# ══════════════════════════════════════════════════════════════
class ActorCriticLSTM(nn.Module):
    """
    Small, fast LSTM policy.
    Input:  obs(18) + prev_action_onehot(5) = 23
    Output: action logits(5) + value(1)

    ~47K parameters (Method 4 had ~200K with hidden=256)
    """
    def __init__(self, input_dim=AUG_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: compress 23-dim input to 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )

        # LSTM: 64 → 128 hidden state
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)

        # Actor head: hidden → action logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

        # Critic head: hidden → state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Orthogonal initialization (PPO best practice)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module.out_features == n_actions else np.sqrt(2)
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x, hidden):
        """
        x:      (batch, seq_len, input_dim)
        hidden: tuple of (h, c), each (1, batch, hidden_dim)
        Returns: logits, values, new_hidden
        """
        encoded = self.encoder(x)
        lstm_out, new_hidden = self.lstm(encoded, hidden)
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)
        return logits, values, new_hidden

    def init_hidden(self, batch_size=1):
        """Create zero initial hidden state on the correct device."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        return (h, c)

    def get_action_and_value(self, obs_aug, hidden):
        """
        Single-step inference during rollout collection.
        Returns: action_idx, log_prob, value, new_hidden
        """
        x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits, value, new_hidden = self(x, hidden)

        dist = Categorical(logits=logits.squeeze(0).squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.squeeze().item(),
            new_hidden,
        )


# ══════════════════════════════════════════════════════════════
# Augmented Observation Builder
# ══════════════════════════════════════════════════════════════
def build_augmented_obs(obs, prev_action):
    """
    Concatenate [obs(18), prev_action_onehot(5)] = 23 dims.
    prev_action tells LSTM what it did last step.
    This helps discriminate wall vs box through action-observation history.
    """
    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    action_onehot[prev_action] = 1.0
    return np.concatenate([obs.astype(np.float32), action_onehot])


# ══════════════════════════════════════════════════════════════
# GAE Computation
# ══════════════════════════════════════════════════════════════
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Generalized Advantage Estimation.
    rewards, values, dones: lists of equal length.
    Returns: advantages (list), returns (list)
    """
    advantages = []
    gae = 0.0
    values_extended = values + [0.0]  # Bootstrap with 0 at terminal

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t + 1] * (1.0 - dones[t]) - values_extended[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ══════════════════════════════════════════════════════════════
# PPO Update
# ══════════════════════════════════════════════════════════════
def ppo_update(model, optimizer, rollout_data):
    """
    PPO clipped objective with minibatch updates.

    rollout_data: dict with keys:
      obs, actions, log_probs, advantages, returns
      (all as lists, converted to tensors here)
    """
    obs_t    = torch.FloatTensor(np.array(rollout_data['obs'])).to(DEVICE)
    act_t    = torch.LongTensor(rollout_data['actions']).to(DEVICE)
    old_lp_t = torch.FloatTensor(rollout_data['log_probs']).to(DEVICE)
    adv_t    = torch.FloatTensor(rollout_data['advantages']).to(DEVICE)
    ret_t    = torch.FloatTensor(rollout_data['returns']).to(DEVICE)

    # Normalize advantages (critical for stable PPO)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    N = len(obs_t)
    total_loss = 0.0

    for epoch in range(PPO_EPOCHS):
        # Shuffle indices
        indices = torch.randperm(N, device=DEVICE)

        for start in range(0, N, MINIBATCH):
            end = min(start + MINIBATCH, N)
            mb_idx = indices[start:end]

            # Forward pass (single-step, no LSTM unrolling — using init hidden)
            # This is simpler and works for PPO (Ni et al. found it sufficient)
            hidden = model.init_hidden(len(mb_idx))
            mb_obs = obs_t[mb_idx].unsqueeze(1)  # (batch, 1, 23)
            logits, values, _ = model(mb_obs, hidden)
            logits = logits.squeeze(1)           # (batch, 5)
            values = values.squeeze(1).squeeze(-1)  # (batch,)

            # Policy loss
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_t[mb_idx])
            entropy = dist.entropy()

            ratio = torch.exp(new_lp - old_lp_t[mb_idx])
            surr1 = ratio * adv_t[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = ((values - ret_t[mb_idx]) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()

    return total_loss


# ══════════════════════════════════════════════════════════════
# Collect One Episode
# ══════════════════════════════════════════════════════════════
def collect_episode(env, model, max_steps, ep_seed):
    """
    Collect one episode.

    Key design decisions:
      1. Escape controller handles stuck (obs[17]==1)
      2. LSTM hidden state updated EVERY step (even during escape)
      3. Escape steps NOT stored in rollout (LSTM doesn't learn -200 penalties)
      4. Raw env rewards, no shaping

    Returns:
      rollout_steps: list of (obs_aug, action, log_prob, reward, value, done)
                     ONLY for LSTM-controlled steps
      total_reward:  raw cumulative reward (including escape penalties)
    """
    obs = env.reset(seed=ep_seed)
    hidden = model.init_hidden(1)
    esc = EscapeController()
    prev_action = 2  # Start with FW

    rollout_steps = []
    total_reward = 0.0

    for step in range(max_steps):
        # Build augmented observation
        obs_aug = build_augmented_obs(obs, prev_action)
        stuck = bool(obs[17])

        # ── Escape trigger ────────────────────────────────
        if stuck and not esc.active:
            esc.trigger(obs)

        # ── Action selection ──────────────────────────────
        is_lstm_step = False

        if esc.active:
            # Hardcoded escape action
            action_idx = esc.get_action()

            # CRITICAL: Still run LSTM forward to keep hidden state consistent
            with torch.no_grad():
                x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(DEVICE)
                _, _, hidden = model(x, hidden)

        else:
            # LSTM decides the action
            is_lstm_step = True
            action_idx, log_prob, value, hidden = model.get_action_and_value(obs_aug, hidden)

            # Store in rollout (reward filled below after env.step)
            rollout_steps.append({
                'obs_aug': obs_aug,
                'action': action_idx,
                'log_prob': log_prob,
                'value': value,
            })

        # ── Environment step (raw reward, no shaping!) ────
        next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
        total_reward += reward

        # Assign reward to the LSTM step that produced this action
        if is_lstm_step:
            rollout_steps[-1]['reward'] = reward / REWARD_SCALE
            rollout_steps[-1]['done'] = float(done)

        prev_action = action_idx
        obs = next_obs

        if done:
            break

    # Make sure all steps have reward/done
    for s in rollout_steps:
        if 'reward' not in s:
            s['reward'] = -1.0 / REWARD_SCALE  # default per-step penalty
            s['done'] = 0.0

    return rollout_steps, total_reward


# ══════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════
def train(
    eps0=1500,    # S0: no wall, max_steps=500 (fast!)
    eps1=3000,    # S1: wall (THE critical stage)
    eps2=1500,    # S2: blinking
    eps3=4000,    # S3: moving + blinking
    seed=42,
    prefix="fast",
):
    """
    Curriculum:
      S0: diff=0, no wall,  max_steps=500  → learn find + approach + push (FAST)
      S1: diff=0, wall=True, max_steps=1000 → learn wall avoidance + gap nav
          Mixed curriculum: 50% wall / 50% no-wall (prevents catastrophic forgetting)
      S2: diff=2, wall=True → add blinking box handling
      S3: diff=3, wall=True → add moving + blinking box

    Total: 10,000 episodes (vs Method 4's 40,000)
    Expected time: 3-4 hours (vs Method 4's 80+)
    """
    print(f"Device: {DEVICE}")
    print(f"Method 7: Fast Hybrid PPO-LSTM")
    print(f"  LSTM hidden: {HIDDEN_DIM}")
    print(f"  Escape controller: {ESC_TURNS} turns + {ESC_FW} forward")
    print(f"  Rollout episodes per update: {ROLLOUT_EPS}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize model and optimizer
    model = ActorCriticLSTM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Paths
    best_path  = os.path.join(_HERE, f"{prefix}_best.pth")
    final_path = os.path.join(_HERE, f"{prefix}_final.pth")
    log_path   = os.path.join(_HERE, f"{prefix}_rewards.csv")

    # Curriculum stages
    # (n_episodes, difficulty, wall, box_speed, max_steps, name, mix_nowall)
    stages = [
        (eps0, 0, False, 0, 500,  "S0:NoWall-500", False),
        (eps1, 0, True,  0, 1000, "S1:Wall-Mixed ", True),   # 50% mixed!
        (eps2, 2, True,  0, 1000, "S2:Blink      ", False),
        (eps3, 3, True,  2, 1000, "S3:Move       ", False),
    ]

    all_rewards = []
    best_mean = -np.inf
    t_start = time.time()

    for stage_idx, (n_eps, diff, wall, box_spd, max_steps, name, mix_nowall) in enumerate(stages):
        print(f"\n{'═' * 70}")
        print(f" {name} | {n_eps} eps | diff={diff} | wall={wall} | max_steps={max_steps}")
        if mix_nowall:
            print(f" ↳ Mixed curriculum: 50% wall + 50% no-wall")
        print(f"{'═' * 70}")

        # Create environment(s)
        env_wall = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall, difficulty=diff, box_speed=box_spd, seed=seed,
        )

        # For mixed curriculum: also create no-wall env
        env_nowall = None
        if mix_nowall:
            env_nowall = OBELIX(
                scaling_factor=5, arena_size=500, max_steps=max_steps,
                wall_obstacles=False, difficulty=diff, box_speed=box_spd, seed=seed,
            )

        stage_rewards = []
        ep = 0

        while ep < n_eps:
            # ── Collect ROLLOUT_EPS episodes ──────────────
            combined_rollout = {
                'obs': [], 'actions': [], 'log_probs': [],
                'rewards': [], 'values': [], 'dones': [],
            }

            for _ in range(ROLLOUT_EPS):
                if ep >= n_eps:
                    break

                # Select environment (mixed curriculum in S1)
                if env_nowall is not None and np.random.random() < 0.5:
                    use_env = env_nowall
                else:
                    use_env = env_wall

                # Collect one episode
                ep_seed = np.random.randint(0, 300_000)
                rollout_steps, total_reward = collect_episode(
                    use_env, model, max_steps, ep_seed
                )

                # Add episode data to combined rollout
                for step_data in rollout_steps:
                    combined_rollout['obs'].append(step_data['obs_aug'])
                    combined_rollout['actions'].append(step_data['action'])
                    combined_rollout['log_probs'].append(step_data['log_prob'])
                    combined_rollout['rewards'].append(step_data['reward'])
                    combined_rollout['values'].append(step_data['value'])
                    combined_rollout['dones'].append(step_data['done'])

                all_rewards.append(total_reward)
                stage_rewards.append(total_reward)
                ep += 1

            # ── PPO Update ────────────────────────────────
            if len(combined_rollout['obs']) > MINIBATCH:
                advantages, returns = compute_gae(
                    combined_rollout['rewards'],
                    combined_rollout['values'],
                    combined_rollout['dones'],
                )
                combined_rollout['advantages'] = advantages
                combined_rollout['returns'] = returns
                ppo_update(model, optimizer, combined_rollout)

            # ── Logging ───────────────────────────────────
            if ep % 200 < ROLLOUT_EPS or ep >= n_eps:
                window = stage_rewards[-200:] if len(stage_rewards) >= 200 else stage_rewards
                mean_r = np.mean(window) if window else 0
                succ = sum(1 for r in window if r > 500)
                elapsed = (time.time() - t_start) / 60.0

                print(
                    f"  [{name} Ep {ep:>5}/{n_eps}] "
                    f"Mean:{mean_r:>9.1f} | "
                    f"Succ:{succ:3d}/200 | "
                    f"Time:{elapsed:>6.1f}m"
                )

                if mean_r > best_mean:
                    best_mean = mean_r
                    torch.save(model.state_dict(), best_path)
                    print(f"    💾 New best: {mean_r:.1f} → {os.path.basename(best_path)}")

    # ── Final save and summary ────────────────────────────────
    torch.save(model.state_dict(), final_path)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(all_rewards):
            writer.writerow([i + 1, f"{r:.2f}"])

    total_time = (time.time() - t_start) / 60.0
    total_succ = sum(1 for r in all_rewards if r > 500)
    total_eps = len(all_rewards)

    print(f"\n{'═' * 70}")
    print(f" TRAINING COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Total time       : {total_time:.1f} minutes")
    print(f"  Total episodes   : {total_eps}")
    print(f"  Best 200-ep mean : {best_mean:.1f}")
    print(f"  Success rate     : {total_succ}/{total_eps} ({100*total_succ/max(total_eps,1):.1f}%)")
    print(f"  Best weights     : {best_path}")
    print(f"  Final weights    : {final_path}")
    print(f"  Reward log       : {log_path}")

    return all_rewards


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Method 7: Fast Hybrid PPO-LSTM")
    p.add_argument("--eps0", type=int, default=1500,
                   help="S0 episodes (no wall, max_steps=500)")
    p.add_argument("--eps1", type=int, default=3000,
                   help="S1 episodes (wall, mixed curriculum)")
    p.add_argument("--eps2", type=int, default=1500,
                   help="S2 episodes (blinking)")
    p.add_argument("--eps3", type=int, default=4000,
                   help="S3 episodes (moving + blinking)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="fast")

    args = p.parse_args()
    train(**vars(args))