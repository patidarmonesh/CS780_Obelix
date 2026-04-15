"""
visualize.py — Method 7 OBELIX Performance Visualizer
=======================================================
Saves MP4 videos for all 4 levels (S0→S3) + one combined video.

Usage:
  python visualize.py                        # auto-finds weights
  python visualize.py --weights fast_best.pth
  python visualize.py --eps_per_level 5      # more episodes per level
  python visualize.py --fps 20               # slower playback

Outputs (saved next to this script):
  level_S0.mp4 / S1.mp4 / S2.mp4 / S3.mp4
  all_levels_combined.mp4

Requirements: pip install opencv-python torch numpy
"""

import os, sys, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [_HERE,
           os.path.join(_HERE, "env"),
           os.path.join(_HERE, "..", "env"),
           os.path.join(_HERE, "..", "..", "env"),
           os.path.join(_HERE, "..", "..", "..", "env")]:
    if os.path.exists(os.path.join(_c, "obelix.py")):
        sys.path.insert(0, _c); break
from obelix import OBELIX

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS    = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS  = 5
OBS_DIM    = 18
AUG_DIM    = OBS_DIM + N_ACTIONS   # 23
HIDDEN_DIM = 128
ESC_TURNS  = 4
ESC_FW     = 4

FRAME_W    = 500
FRAME_H    = 500
HUD_H      = 95          # pixels below env frame
TOTAL_H    = FRAME_H + HUD_H
FPS_DEF    = 30

# BGR palette
C_WHITE  = (255, 255, 255)
C_GREEN  = (80,  220, 80)
C_RED    = (80,  80,  220)
C_YELLOW = (80,  220, 220)
C_CYAN   = (220, 210, 60)
C_ORANGE = (60,  160, 255)
C_GRAY   = (140, 140, 140)
C_BG_HUD = (18,  18,  26)


# ── Model (mirrors train code exactly) ───────────────────────────────────────
class ActorCriticLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = HIDDEN_DIM
        self.encoder = nn.Sequential(nn.Linear(AUG_DIM, 64), nn.ReLU())
        self.lstm    = nn.LSTM(64, HIDDEN_DIM, batch_first=True)
        self.actor   = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.Tanh(),
                                     nn.Linear(64, N_ACTIONS))
        self.critic  = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.Tanh(),
                                     nn.Linear(64, 1))

    def forward(self, x, hidden):
        enc = self.encoder(x)
        out, new_h = self.lstm(enc, hidden)
        return self.actor(out), self.critic(out), new_h

    def init_hidden(self):
        h = torch.zeros(1, 1, HIDDEN_DIM)
        c = torch.zeros(1, 1, HIDDEN_DIM)
        return (h, c)


# ── Escape Controller (mirrors agent code exactly) ────────────────────────────
class EscapeController:
    TOTAL = ESC_TURNS + ESC_FW

    def __init__(self):
        self.active = False
        self._step  = 0
        self._turn  = 4   # R45 default

    def trigger(self, obs):
        left  = sum(int(obs[i]) for i in range(0, 4))
        right = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 4 if left >= right else 0
        self._step = 0
        self.active = True

    def get_action(self):
        i = self._step; self._step += 1
        if self._step >= self.TOTAL:
            self.active = False
        return self._turn if i < ESC_TURNS else 2


# ── HUD Renderer ──────────────────────────────────────────────────────────────
def _txt(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_hud(env_frame, step, max_steps, reward, active_state,
             stuck, esc_active, action_name,
             ep_num, n_eps, level_name, ep_result=""):
    """
    Compose env_frame (500×500) + HUD panel → canvas (595×500×3 BGR).
    """
    canvas = np.zeros((TOTAL_H, FRAME_W, 3), dtype=np.uint8)
    canvas[:FRAME_H] = env_frame
    canvas[FRAME_H:] = C_BG_HUD
    cv2.line(canvas, (0, FRAME_H), (FRAME_W, FRAME_H), (55, 55, 75), 1)

    # ── Row 1: Level  |  Episode  |  Step bar ──────────────
    y1 = FRAME_H + 20
    _txt(canvas, level_name, (8, y1), 0.52, C_CYAN, 1)
    _txt(canvas, f"Ep {ep_num}/{n_eps}", (200, y1), 0.48, C_WHITE, 1)

    # Step progress bar
    bx, by, bw, bh = 290, y1 - 13, 185, 13
    filled = int(bw * min(step / max_steps, 1.0))
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (45, 45, 58), -1)
    bar_c = C_GREEN if filled < bw * 0.75 else C_ORANGE
    if filled > 0:
        cv2.rectangle(canvas, (bx, by), (bx + filled, by + bh), bar_c, -1)
    _txt(canvas, f"{step}", (bx + bw + 4, y1), 0.38, C_GRAY, 1)

    # ── Row 2: Reward  |  State  |  Action  |  Status ─────
    y2 = FRAME_H + 45
    rew_c = C_GREEN if reward > 200 else (C_RED if reward < -300 else C_WHITE)
    _txt(canvas, f"Rew:{reward:+.0f}", (8, y2), 0.50, rew_c, 1)

    state_c = {"F": C_WHITE, "P": C_GREEN, "U": C_YELLOW}.get(active_state, C_WHITE)
    state_s = {"F": "FIND", "P": "PUSH ✓", "U": "STUCK"}.get(active_state, active_state)
    _txt(canvas, state_s, (165, y2), 0.50, state_c, 1)

    _txt(canvas, f"Act:{action_name:<6}", (295, y2), 0.50, C_WHITE, 1)

    if esc_active:
        cv2.rectangle(canvas, (420, y2 - 14), (496, y2 + 3), (0, 100, 200), -1)
        _txt(canvas, "ESC", (428, y2), 0.50, C_WHITE, 1)
    elif stuck:
        _txt(canvas, "STUCK", (428, y2), 0.50, C_RED, 1)
    else:
        _txt(canvas, "FREE", (428, y2), 0.50, C_GREEN, 1)

    # ── Row 3: Episode result banner ───────────────────────
    y3 = FRAME_H + 72
    if ep_result == "success":
        cv2.rectangle(canvas, (0, FRAME_H + 53), (FRAME_W, TOTAL_H), (0, 50, 0), -1)
        _txt(canvas, "SUCCESS  Box reached boundary!", (10, y3), 0.57, C_GREEN, 2)
    elif ep_result == "timeout":
        _txt(canvas, "TIMEOUT — episode ended", (10, y3), 0.57, C_RED, 1)
    elif ep_result == "negative":
        _txt(canvas, "HIT NEGATIVE OBJECT", (10, y3), 0.57, C_RED, 1)
    else:
        # Sensor bar: show which sensors are active
        bits = np.zeros(18)   # placeholder (real obs passed below)
        _txt(canvas, f"Steps remaining: {max_steps - step}", (10, y3), 0.43, C_GRAY, 1)

    return canvas


# ── Sensor mini-bar overlay (bottom-left of env frame) ────────────────────────
def draw_sensor_overlay(canvas, obs):
    """Draw 18 tiny sensor squares on the env frame corner."""
    sq, gap, ox, oy = 8, 2, 6, FRAME_H - 14
    for i in range(18):
        x = ox + i * (sq + gap)
        color = C_GREEN if obs[i] else (50, 50, 50)
        cv2.rectangle(canvas, (x, oy), (x + sq, oy + sq), color, -1)


# ── Title card generator ──────────────────────────────────────────────────────
def title_card(line1, line2="", n_frames=60):
    frames = []
    for _ in range(n_frames):
        img = np.full((TOTAL_H, FRAME_W, 3), C_BG_HUD, dtype=np.uint8)
        # horizontal accent line
        cv2.line(img, (40, TOTAL_H // 2 - 30), (FRAME_W - 40, TOTAL_H // 2 - 30), (55, 55, 75), 1)
        (tw, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        _txt(img, line1, ((FRAME_W - tw) // 2, TOTAL_H // 2), 0.85, C_CYAN, 2)
        if line2:
            (sw, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            _txt(img, line2, ((FRAME_W - sw) // 2, TOTAL_H // 2 + 32), 0.48, C_GRAY, 1)
        frames.append(img)
    return frames


# ── Run one episode, capture frames ───────────────────────────────────────────
def run_episode(env, model, max_steps, ep_seed, ep_num, n_eps, level_name, hold=70):
    obs       = env.reset(seed=ep_seed)
    hidden    = model.init_hidden()
    esc       = EscapeController()
    prev_act  = 2        # start with FW
    cum_rew   = 0.0
    frames    = []
    act_name  = "FW"

    for step in range(1, max_steps + 1):
        # Augmented observation
        oh = np.zeros(N_ACTIONS, np.float32); oh[prev_act] = 1.0
        obs_aug = np.concatenate([obs.astype(np.float32), oh])

        stuck = bool(obs[17])
        if stuck and not esc.active:
            esc.trigger(obs)

        # Action
        if esc.active:
            action_idx = esc.get_action()
            act_name   = ACTIONS[action_idx] + "*"       # * marks escape
            with torch.no_grad():
                x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0)
                _, _, hidden = model(x, hidden)
        else:
            with torch.no_grad():
                x = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0)
                logits, _, hidden = model(x, hidden)
                action_idx = logits.squeeze().argmax().item()
            act_name = ACTIONS[action_idx]

        next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
        cum_rew += reward

        # Determine result for last frame
        ep_result = ""
        if done:
            if cum_rew > 500:       ep_result = "success"
            elif reward == -100:    ep_result = "negative"
            else:                   ep_result = "timeout"

        # Compose frame
        env_frame = env.frame.copy()   # 500×500 BGR — env already flips it
        canvas = draw_hud(env_frame, step, max_steps, cum_rew,
                          env.active_state, stuck, esc.active,
                          act_name, ep_num, n_eps, level_name, ep_result)
        draw_sensor_overlay(canvas, obs)
        frames.append(canvas)

        prev_act = action_idx
        obs = next_obs

        if done:
            for _ in range(hold):      # hold result screen
                frames.append(canvas.copy())
            break

    # Timeout — add hold frames with timeout banner
    if not (obs[17] and cum_rew > 500):  # rough check
        if len(frames) > 0:
            last_env = frames[-hold - 1][:FRAME_H].copy() if len(frames) > hold else frames[-1][:FRAME_H].copy()
            timeout_frame = draw_hud(last_env, step, max_steps, cum_rew,
                                     env.active_state, False, False,
                                     act_name, ep_num, n_eps, level_name, "timeout")
            draw_sensor_overlay(timeout_frame, obs)
            # Only append if episode didn't already end with done
            # (done already adds hold frames above)

    return frames, cum_rew, (cum_rew > 500)


# ── Save video ────────────────────────────────────────────────────────────────
def save_video(frames, path, fps):
    if not frames:
        print(f"  [warn] no frames → {path}")
        return
    h, w = frames[0].shape[:2]
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved: {os.path.basename(path)}  ({len(frames)} frames, {size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",       default=None)
    p.add_argument("--eps_per_level", type=int, default=3)
    p.add_argument("--fps",           type=int, default=FPS_DEF)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out_dir",       default=None)
    args = p.parse_args()

    out_dir = args.out_dir or _HERE
    os.makedirs(out_dir, exist_ok=True)

    # ── Load weights ──────────────────────────────────────────────────────────
    model = ActorCriticLSTM()
    search = ([args.weights] if args.weights else []) + [
        "weights.pth", "fast_best.pth", "fast_final.pth"
    ]
    loaded = False
    for fname in search:
        path = fname if os.path.isabs(fname) else os.path.join(_HERE, fname)
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            model.load_state_dict(sd)
            model.eval()
            print(f"[Viz] Weights loaded: {path}")
            loaded = True
            break
    if not loaded:
        model.eval()
        print("[Viz] WARNING: No weights — using random policy")

    # ── Level definitions ─────────────────────────────────────────────────────
    levels = [
        dict(label="S0", name="S0: Static Box  |  No Wall",
             diff=0, wall=False, bspd=0, max_steps=500),
        dict(label="S1", name="S1: Static Box  |  Wall",
             diff=0, wall=True,  bspd=0, max_steps=1000),
        dict(label="S2", name="S2: Blinking Box  |  Wall",
             diff=2, wall=True,  bspd=0, max_steps=1000),
        dict(label="S3", name="S3: Moving+Blinking  |  Wall",
             diff=3, wall=True,  bspd=2, max_steps=1000),
    ]

    rng        = np.random.default_rng(args.seed)
    all_frames = title_card("Method 7: Hybrid PPO-LSTM",
                            "OBELIX — All 4 Curriculum Levels", n_frames=90)

    for lv in levels:
        print(f"\n{'─'*60}")
        print(f"  {lv['name']}")
        print(f"{'─'*60}")

        env = OBELIX(
            scaling_factor=5, arena_size=500, max_steps=lv["max_steps"],
            wall_obstacles=lv["wall"], difficulty=lv["diff"],
            box_speed=lv["bspd"],
            seed=int(rng.integers(0, 100_000))
        )

        lv_frames = title_card(lv["label"], lv["name"], n_frames=50)
        all_frames += title_card(lv["label"], lv["name"], n_frames=50)

        ep_rews = []
        n = args.eps_per_level

        for ep in range(1, n + 1):
            ep_seed = int(rng.integers(0, 500_000))
            print(f"  Ep {ep}/{n}  seed={ep_seed}", end="  ", flush=True)

            frames, rew, ok = run_episode(
                env, model, lv["max_steps"], ep_seed,
                ep, n, f"{lv['label']}: {lv['name']}"
            )

            ep_rews.append(rew)
            lv_frames  += frames
            all_frames += frames
            tag = "SUCCESS ✓" if ok else "timeout ✗"
            print(f"reward={rew:+.0f}  {tag}  [{len(frames)} frames]")

        print(f"  Mean reward: {np.mean(ep_rews):.1f}")
        save_video(lv_frames,
                   os.path.join(out_dir, f"level_{lv['label']}.mp4"),
                   fps=args.fps)

    # ── Combined ──────────────────────────────────────────────────────────────
    save_video(all_frames,
               os.path.join(out_dir, "all_levels_combined.mp4"),
               fps=args.fps)

    print(f"\n{'═'*60}")
    print(f"  All videos saved to: {out_dir}")
    print(f"  Files: level_S0.mp4  level_S1.mp4  level_S2.mp4  level_S3.mp4")
    print(f"         all_levels_combined.mp4")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
