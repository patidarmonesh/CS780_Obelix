"""
record_video.py — visualize agent behavior after training
Usage: python record_video.py --difficulty 3 --episodes 3
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "env"))
import cv2, os, sys, pickle
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from obelix import OBELIX
import importlib.util

def load_agent(agent_path):
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy

def record(agent_path="agent.py", difficulty=3, episodes=3,
           out_dir="videos", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    policy = load_agent(agent_path)

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1000,
                 wall_obstacles=True, difficulty=difficulty,
                 box_speed=2, seed=seed)

    for ep in range(episodes):
        obs   = env.reset(seed=seed + ep)
        rng   = np.random.default_rng(seed + ep)
        total = 0.0
        frames = []

        done = False
        while not done:
            env._update_frames(show=False)
            frame = env.frame.copy()
            frame = cv2.flip(frame, 0)

            # Overlay step info
            cv2.putText(frame, f"Ep:{ep+1} R:{total:.0f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1)
            frames.append(frame)

            action = policy(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total += reward

        # Write video
        h, w = frames[0].shape[:2]
        out_path = os.path.join(out_dir, f"ep{ep+1}_d{difficulty}_r{total:.0f}.mp4")
        vw = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                     if f.shape[2] == 3 else f)
        vw.release()
        print(f"  ✅ Saved {out_path}  (score={total:.0f}, steps={len(frames)})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent",      default="agent.py")
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--episodes",   type=int, default=3)
    p.add_argument("--seed",       type=int, default=0)
    a = p.parse_args()
    record(a.agent, a.difficulty, a.episodes, seed=a.seed)