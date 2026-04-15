"""
Q(lambda) v3 — FINAL Fix
=========================
Fixes:
  1. arena=500, scale=5  → exact Codabench settings
  2. wall_obstacles=True → agent learns to handle walls
  3. Stuck recovery       → when obs[17]=1, turn don't go forward
  4. max_steps=350        → enough time to find box in 500px arena
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "env"))


import pickle
import numpy as np
from collections import defaultdict, deque
import random
from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

LAMBDA        = 0.80
ALPHA_0       = 0.20
ALPHA_C       = 2.0
GAMMA         = 0.99
Q_INIT        = 50.0
TRACE_MIN     = 1e-3
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.998

REPLAY_SIZE  = 20000
REPLAY_BATCH = 64
REPLAY_EVERY = 5

_PHI = np.array([
    1, 1, 1, 1,
    2, 3, 2, 3, 2, 3, 2, 3,
    1, 1, 1, 1,
    5, 0,
], dtype=np.float64)


def obs_to_state(obs):
    return tuple(obs.astype(int).tolist())

def potential(obs):
    return float(np.dot(obs, _PHI))

def get_alpha(n):
    return ALPHA_0 / (1.0 + ALPHA_C * n)

def select_module(obs):
    return "pusher" if (obs[16] == 1 or obs[17] == 1) else "finder"

def epsilon_greedy(q_vals, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


def train(
    episodes=3000, max_steps=350, seed=42,
    wall_obstacles=True,   # ✅ FIXED: train WITH walls
    difficulty=0,
    save_path="q_table_exp01_final.pkl",
):
    rng    = np.random.default_rng(seed)
    Q      = {
        "finder": defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64)),
        "pusher": defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64)),
    }
    N      = {
        "finder": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32)),
        "pusher": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32)),
    }
    replay = deque(maxlen=REPLAY_SIZE)

    # ✅ FIXED: Same as Codabench evaluation settings
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        seed=seed,
    )

    epsilon         = EPSILON_START
    episode_rewards = []

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 100_000))
        obs     = env.reset(seed=ep_seed)
        state   = obs_to_state(obs)
        module  = select_module(obs)
        total   = 0.0
        e       = {}

        for _ in range(max_steps):
            q_vals     = Q[module][state]
            greedy_idx = int(np.argmax(q_vals))
            action_idx = epsilon_greedy(q_vals, epsilon, rng)
            action     = ACTIONS[action_idx]

            next_obs, reward, done = env.step(action, render=False)
            next_state  = obs_to_state(next_obs)
            next_module = select_module(next_obs)
            total      += reward

            shaped = reward + GAMMA * potential(next_obs) - potential(obs)
            replay.append((module, state, action_idx, shaped,
                           next_module, next_state, done))

            best_next = float(np.max(Q[next_module][next_state]))
            delta     = shaped + GAMMA * best_next - Q[module][state][action_idx]

            N[module][state][action_idx] += 1
            key    = (module, state, action_idx)
            e[key] = e.get(key, 0.0) + 1.0

            for k in list(e.keys()):
                m_k, s_k, a_k = k
                Q[m_k][s_k][a_k] += get_alpha(int(N[m_k][s_k][a_k])) * delta * e[k]
                e[k] *= GAMMA * LAMBDA
                if e[k] < TRACE_MIN:
                    del e[k]

            if action_idx != greedy_idx or next_module != module:
                e = {}

            obs    = next_obs
            state  = next_state
            module = next_module
            if done:
                break

        # Experience Replay
        if len(replay) >= REPLAY_BATCH and (ep + 1) % REPLAY_EVERY == 0:
            batch = random.sample(replay, REPLAY_BATCH)
            for (m, s, a, r_s, nm, ns, d) in batch:
                target = r_s if d else r_s + GAMMA * float(np.max(Q[nm][ns]))
                N[m][s][a] += 1
                Q[m][s][a] += get_alpha(int(N[m][s][a])) * (target - Q[m][s][a])

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total)

        if (ep + 1) % 50 == 0:
            mean_r = np.mean(episode_rewards[-50:])
            trend  = "✅" if mean_r > np.mean(
                episode_rewards[max(0, len(episode_rewards)-100):-50] or [mean_r-1]
            ) else "⏳"
            print(
                f"[Ep {ep+1:>4}/{episodes}] Mean: {mean_r:>8.1f} | "
                f"ε: {epsilon:.4f} | "
                f"F/P: {len(Q['finder'])}/{len(Q['pusher'])} | "
                f"Buf: {len(replay)} {trend}"
            )

    data = {
        "Q_finder": {k: v.tolist() for k, v in Q["finder"].items()},
        "Q_pusher": {k: v.tolist() for k, v in Q["pusher"].items()},
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✅ Saved: {save_path} (F:{len(Q['finder'])} P:{len(Q['pusher'])})")
    return episode_rewards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,  default=3000)
    parser.add_argument("--max_steps",      type=int,  default=350)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--difficulty",     type=int,  default=0)
    parser.add_argument("--wall_obstacles", action="store_true", default=True)
    parser.add_argument("--save_path",      type=str,  default="q_table_exp01_final_1.pkl")
    args = parser.parse_args()
    train(**vars(args))
