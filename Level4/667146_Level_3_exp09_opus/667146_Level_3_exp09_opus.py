"""
train.py — OBELIX Final Agent Training
========================================
Dyna-Q(λ) with replacing traces.

Architecture matches agent.py exactly:
- make_state() identical
- EscapeController identical
- Push detection uses reward > 50 (available during training)
- PushRecovery ONLY used when push is confirmed by reward
- Q-table controls approach phase (sensors active, not stuck, not pushing)
- Zero-sensor exploration is hardcoded random walk (never Q-table)

Run:
  python train.py
  python train.py --eps0 6000 --eps1 6000 --eps2 4000 --eps3 12000
"""

import sys, os, pickle, csv, random
import numpy as np
from collections import defaultdict, deque, OrderedDict
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "..", "env"),
           os.path.join(_HERE, "..", "..", "env"),
           os.path.join(_HERE, "..", "..", "..", "env")]:
    if os.path.exists(os.path.join(_c, "obelix.py")):
        sys.path.insert(0, _c); break

from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5

# Q(λ) core
ALPHA_0  = 0.20;  ALPHA_C = 2.0
GAMMA    = 0.995;  LAMBDA  = 0.70
Q_INIT   = 3.0;    TRACE_MIN = 1e-3

# Exploration
EPS_START = 1.0;  EPS_END = 0.04;  EPS_DECAY = 0.9992

# Controllers
BLINK_MEM   = 30;  VEL_WIN    = 5
ESC_TURNS   = 4;   ESC_FW     = 6;  POST_COOL = 8
PUSH_RECOV  = 6;   PURSUIT_TTL = 18

# Dyna-Q
N_PLAN    = 20;  MODEL_CAP = 3000

# Dense reward (training only)
FW_BONUS  = 0.3
SENSOR_W  = np.array([
    0.2,0.2,0.2,0.2,  0.4,0.8,0.4,0.8,  0.4,0.8,0.4,0.8,  0.2,0.2,0.2,0.2,  2.0
], dtype=np.float32)

VIDEO_EVERY = 500; VIDEO_FPS = 15
VIDEO_DIR   = os.path.join(_HERE, "videos")


# ══════════ State (IDENTICAL to agent.py) ══════════
def make_state(obs, history, enable_push, last_dir, pursuit_steps):
    ir       = int(obs[16])
    fwd_near = int(obs[4] or obs[6] or obs[8] or obs[10])
    fwd_far  = int(obs[5] or obs[7] or obs[9] or obs[11])
    left_s   = int(obs[0]) + int(obs[1]) + int(obs[2]) + int(obs[3])
    right_s  = int(obs[12]) + int(obs[13]) + int(obs[14]) + int(obs[15])
    any_now  = ir or fwd_near or fwd_far or left_s > 0 or right_s > 0

    if ir:                           phase = 4
    elif fwd_near:                   phase = 3
    elif fwd_far:                    phase = 2
    elif left_s > 0 or right_s > 0:  phase = 1
    else:                            phase = 0

    fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
    fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
    if fl + fr == 0:    direction = 1
    elif fl > fr + 1:   direction = 0
    elif fr > fl + 1:   direction = 2
    else:               direction = 1

    if left_s > 0 and right_s > 0:  wall_side = 3
    elif left_s > 0:                 wall_side = 1
    elif right_s > 0:                wall_side = 2
    else:                            wall_side = 0

    hist = list(history)
    recent = any(bool(np.any(h[:17])) for h in hist[-20:]) if hist else False
    is_blind = int(not any_now and recent)
    was_close = int(any(bool(h[16] == 1) for h in hist[-15:]) if hist else False)

    vel_hint = 1
    if len(hist) >= 3:
        centroids = []
        for h in hist[-VEL_WIN:]:
            l = int(h[4]) + int(h[5]) + int(h[6]) + int(h[7])
            r = int(h[8]) + int(h[9]) + int(h[10]) + int(h[11])
            t = l + r
            if t > 0: centroids.append((r - l) / t)
        if len(centroids) >= 3:
            drift = np.mean(centroids[-2:]) - np.mean(centroids[:2])
            if drift > 0.3:    vel_hint = 2
            elif drift < -0.3: vel_hint = 0

    moving_blind = int(is_blind and pursuit_steps <= PURSUIT_TTL)
    return (phase, direction, int(enable_push), is_blind,
            vel_hint, wall_side, was_close, moving_blind)


# ══════════ Controllers ══════════
class EscapeController:
    SEQ_LEN = ESC_TURNS + ESC_FW
    def __init__(self):
        self.active = False; self._step = 0; self._turn = 4
    def trigger(self, obs):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 4 if lw > rw else (0 if rw > lw else 4)
        self._step = 0; self.active = True
    def next_action(self):
        i = self._step; self._step += 1
        if self._step >= self.SEQ_LEN: self.active = False
        return self._turn if i < ESC_TURNS else 2

class PushRecovery:
    def __init__(self):
        self.active = False; self._step = 0; self._turn = 1
    def trigger(self, obs):
        lw = sum(int(obs[i]) for i in range(0, 4))
        rw = sum(int(obs[i]) for i in range(12, 16))
        self._turn = 1 if lw <= rw else 3
        self._step = 0; self.active = True
    def next_action(self):
        i = self._step; self._step += 1
        if self._step >= PUSH_RECOV: self.active = False
        return self._turn if i % 2 == 0 else 2

def intercept_action(vel_hint, step):
    if vel_hint == 2: return 3 if step % 3 == 0 else 2
    if vel_hint == 0: return 1 if step % 3 == 0 else 2
    return 2

def station_action(step):
    return [1, 1, 3, 3, 3, 3, 1, 1][step % 8]


# ══════════ Reward Wrapper (training only) ══════════
class RewardWrapper:
    def __init__(self, raw_env):
        self._env = raw_env; self.enable_push = False
    def reset(self, seed=None):
        self.enable_push = False
        return self._env.reset(seed=seed)
    def step(self, action, render=False):
        obs, reward, done = self._env.step(action, render=render)
        if reward > 50: self.enable_push = True
        dense = float(np.dot(obs[:17].astype(np.float32), SENSOR_W))
        if action == "FW" and obs[17] == 0: dense += FW_BONUS
        return obs, reward + dense, done
    def __getattr__(self, name):
        return getattr(self._env, name)


# ══════════ Q-learning Helpers ══════════
def get_alpha(n): return ALPHA_0 / (1.0 + ALPHA_C * n)
def eps_greedy(q_vals, epsilon, rng):
    if rng.random() < epsilon: return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))

class LRUModel:
    def __init__(self, cap=MODEL_CAP):
        self._store = OrderedDict(); self._cap = cap
    def update(self, key, value):
        if key in self._store: self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._cap: self._store.popitem(last=False)
    def sample(self, k, rng):
        if not self._store: return []
        keys = list(self._store.keys())
        chosen = rng.choice(len(keys), size=min(k, len(keys)), replace=False)
        return [(keys[i], self._store[keys[i]]) for i in chosen]
    def __len__(self): return len(self._store)

def save_Q(Q, N, path):
    data = {"Q": {str(k): v.tolist() for k, v in Q.items()},
            "N": {str(k): v.tolist() for k, v in N.items()}}
    with open(path, "wb") as f: pickle.dump(data, f)
    print(f"  💾 {len(Q)} states → {os.path.basename(path)}")

def log_csv(path, rewards):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards): w.writerow([i+1, f"{r:.2f}"])


# ══════════ Video Recording ══════════
def record_episode(raw_env, Q, ep_num, difficulty, seed, max_steps):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fname = os.path.join(VIDEO_DIR, f"ep{ep_num:05d}_diff{difficulty}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v"); writer = None
    obs = raw_env.reset(seed=seed)
    history = deque(maxlen=BLINK_MEM); history.append(obs.copy())
    esc = EscapeController(); cooldown = 0; blink_step = 0
    last_dir = 2; pursuit_st = 999; total = 0.0
    rng_v = np.random.default_rng(seed + 77777)

    for step in range(max_steps):
        raw_env._update_frames(show=False)
        frame = cv2.flip(raw_env.frame.copy(), 0)
        mode = "ESC" if esc.active else ("COOL" if cooldown > 0 else "FIND")
        cv2.putText(frame, f"S:{step:4d} R:{total:7.0f} [{mode}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(fname, fourcc, VIDEO_FPS, (w, h))
        writer.write(frame)

        stuck = obs[17] == 1; ir_on = obs[16] == 1
        any_vis = bool(np.any(obs[:17]))
        state = make_state(obs, history, False, last_dir, pursuit_st)
        vel = state[4]; mb = state[7]

        if any(int(obs[i]) for i in range(4, 12)) or obs[16]:
            pursuit_st = 0
            fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
            fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
            if fl > fr + 1: last_dir = 1
            elif fr > fl + 1: last_dir = 3
            else: last_dir = 2
        else: pursuit_st = min(pursuit_st + 1, 999)

        if stuck and not esc.active: esc.trigger(obs); cooldown = 0

        if esc.active:
            action_idx = esc.next_action()
            if not esc.active: cooldown = POST_COOL
        elif cooldown > 0: action_idx = 2; cooldown -= 1
        elif stuck: esc.trigger(obs); action_idx = esc.next_action()
        elif ir_on: action_idx = 2
        elif mb: action_idx = last_dir
        elif any_vis and vel != 1: action_idx = intercept_action(vel, step)
        elif state[3]: action_idx = station_action(blink_step); blink_step += 1
        elif any_vis:
            if state in Q: action_idx = int(np.argmax(Q[state]))
            else: action_idx = int(rng_v.integers(0, N_ACTIONS))
        else:
            p = rng_v.random()
            if p < 0.025: action_idx = 0
            elif p < 0.175: action_idx = 1
            elif p < 0.825: action_idx = 2
            elif p < 0.975: action_idx = 3
            else: action_idx = 4

        obs, reward, done = raw_env.step(ACTIONS[action_idx], render=False)
        history.append(obs.copy()); total += reward
        if done: break

    if writer: writer.release()
    print(f"  🎬 {os.path.basename(fname)}  score={total:.0f}")


# ══════════ Core Training Stage ══════════
def run_stage(
    wrapped_env, raw_env, Q, N, model,
    n_eps, epsilon, rng, name, max_steps,
    best_path, best_mean_ref, all_rwd, total_ep,
    vid_offset, difficulty, n_plan=N_PLAN, log_every=200,
):
    stage_rwd = []; best_mean = best_mean_ref
    esc = EscapeController(); push_rec = PushRecovery()

    for ep in range(n_eps):
        ep_seed = int(rng.integers(0, 300_000))
        obs = wrapped_env.reset(seed=ep_seed)
        history = deque(maxlen=BLINK_MEM); history.append(obs.copy())
        enable_push = False; cooldown = 0; blink_step = 0
        last_dir = 2; pursuit_st = 999; total = 0.0; traces = {}

        for step in range(max_steps):
            stuck = obs[17] == 1; ir_on = obs[16] == 1
            any_vis = bool(np.any(obs[:17]))
            state = make_state(obs, history, enable_push, last_dir, pursuit_st)
            vel = state[4]; mb = state[7]

            # Track front-sensor contact for pursuit (not side)
            front_detected = bool(obs[16] or any(obs[i] for i in range(4, 12)))
            if front_detected:
                pursuit_st = 0
                fl = int(obs[4]) + int(obs[5]) + int(obs[6]) + int(obs[7])
                fr = int(obs[8]) + int(obs[9]) + int(obs[10]) + int(obs[11])
                if fl > fr + 1: last_dir = 1
                elif fr > fl + 1: last_dir = 3
                else: last_dir = 2
            else: pursuit_st = min(pursuit_st + 1, 999)

            # Controller triggers (during training, push is confirmed by reward)
            if enable_push and stuck and not push_rec.active:
                push_rec.trigger(obs); traces = {}
            elif not enable_push and stuck and not esc.active:
                esc.trigger(obs); cooldown = 0; traces = {}

            # Action selection
            use_q = False

            if push_rec.active:
                action_idx = push_rec.next_action()
            elif esc.active:
                action_idx = esc.next_action()
                if not esc.active: cooldown = POST_COOL
            elif cooldown > 0: action_idx = 2; cooldown -= 1
            elif stuck: esc.trigger(obs); action_idx = esc.next_action()
            elif ir_on: action_idx = 2
            elif enable_push: action_idx = 2; use_q = True
            elif mb: action_idx = last_dir
            elif any_vis and vel != 1:
                action_idx = intercept_action(vel, step); blink_step = 0
            elif state[3]:
                action_idx = station_action(blink_step); blink_step += 1
            elif any_vis:
                blink_step = 0
                action_idx = eps_greedy(Q[state], epsilon, rng)
                use_q = True
            else:
                blink_step = 0
                p = rng.random()
                if p < 0.025: action_idx = 0
                elif p < 0.175: action_idx = 1
                elif p < 0.825: action_idx = 2
                elif p < 0.975: action_idx = 3
                else: action_idx = 4

            # Environment step
            action = ACTIONS[action_idx]
            next_obs, reward, done = wrapped_env.step(action, render=False)
            history.append(next_obs.copy()); total += reward

            if not enable_push and reward > 50:
                enable_push = True; traces = {}

            # Q(λ) update
            if use_q and not esc.active and not push_rec.active and cooldown == 0:
                next_push = wrapped_env.enable_push
                next_st = pursuit_st + 1 if not any(int(next_obs[i]) for i in range(16)) else 0
                next_state = make_state(next_obs, history, next_push, last_dir, next_st)
                best_next = float(np.max(Q[next_state]))
                td_err = reward + GAMMA * best_next - Q[state][action_idx]

                N[state][action_idx] += 1
                traces[(state, action_idx)] = 1.0

                for (s, a), e_val in list(traces.items()):
                    Q[s][a] += get_alpha(int(N[s][a])) * td_err * e_val
                    new_e = e_val * GAMMA * LAMBDA
                    if new_e < TRACE_MIN: del traces[(s, a)]
                    else: traces[(s, a)] = new_e

                if next_push != enable_push: traces = {}

                model.update((state, action_idx), (reward, next_state))
                for (ms, ma), (mr, mns) in model.sample(n_plan, rng):
                    mp_best = float(np.max(Q[mns]))
                    Q[ms][ma] += get_alpha(int(N[ms][ma])) * (
                        mr + GAMMA * mp_best - Q[ms][ma])

            obs = next_obs
            if done: break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        stage_rwd.append(total); all_rwd.append(total); total_ep[0] += 1

        if (ep + 1) % log_every == 0:
            m_rwd = np.mean(stage_rwd[-log_every:])
            prev = stage_rwd[-(2*log_every):-log_every] if len(stage_rwd) > log_every else [m_rwd-1]
            trend = "✅" if m_rwd > np.mean(prev) else "⏳"
            succ = sum(1 for r in stage_rwd[-log_every:] if r > 500)
            print(f"[{name} Ep {ep+1:>5}/{n_eps}] "
                  f"Mean:{m_rwd:>9.1f} | ε:{epsilon:.4f} | "
                  f"Succ/{log_every}:{succ:3d} | States:{len(Q):5d} | "
                  f"Model:{len(model):4d} {trend}")
            if m_rwd > best_mean:
                best_mean = m_rwd; save_Q(Q, N, best_path)

        if raw_env is not None and (ep + 1) % VIDEO_EVERY == 0:
            record_episode(raw_env, Q, vid_offset+total_ep[0],
                          difficulty, int(rng.integers(0, 10_000)), max_steps)

    return epsilon, best_mean


# ══════════ Main ══════════
def train(
    eps0=4000, eps1=4000, eps2=3000, eps3=8000,
    max_steps=1000, seed=42, prefix="q_final", box_speed=2, n_plan=N_PLAN,
):
    best_path = os.path.join(_HERE, f"{prefix}_best.pkl")
    final_path = os.path.join(_HERE, f"{prefix}_final.pkl")
    log_path = os.path.join(_HERE, f"{prefix}_rewards.csv")
    rng = np.random.default_rng(seed); random.seed(seed)

    Q = defaultdict(lambda: np.full(N_ACTIONS, Q_INIT, dtype=np.float64))
    N = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int32))
    model = LRUModel(cap=MODEL_CAP)

    epsilon = EPS_START; all_rwd = []; best_mean = -np.inf; total_ep = [0]

    def make_env(wall, diff, spd=0):
        raw = OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=spd, seed=seed)
        return RewardWrapper(raw), raw

    stages = [
        (eps0, 0, False, 0, EPS_START, "S0 (no wall, static)"),
        (eps1, 0, True,  0, 0.40,      "S1 (wall, static)   "),
        (eps2, 2, True,  0, 0.30,      "S2 (wall, blink)    "),
        (eps3, 3, True,  box_speed, 0.25, "S3 (wall, move+blink)"),
    ]

    offset = 0
    for n_eps, diff, wall, spd, eps_reset, name in stages:
        epsilon = max(eps_reset, epsilon)
        print(f"\n{'═'*65}\n {name} | {n_eps} eps\n{'═'*65}")
        w, r = make_env(wall, diff, spd)
        epsilon, best_mean = run_stage(
            w, r, Q, N, model, n_eps, epsilon, rng,
            name, max_steps, best_path, best_mean,
            all_rwd, total_ep, offset, diff, n_plan)
        offset += n_eps

    save_Q(Q, N, final_path); log_csv(log_path, all_rwd)
    succ = sum(1 for r in all_rwd if r > 500)
    total = total_ep[0]
    print(f"\n{'═'*65}\n DONE | Best: {best_mean:.1f} | "
          f"Succ: {succ}/{total} ({succ/max(total,1)*100:.1f}%)\n{'═'*65}")

    print("\n🎬 Final videos...")
    for diff_v, wall_v in [(0,False),(0,True),(2,True),(3,True)]:
        env_v = OBELIX(scaling_factor=5, arena_size=500, max_steps=max_steps,
                       wall_obstacles=wall_v, difficulty=diff_v, seed=seed)
        for s in range(3):
            record_episode(env_v, Q, 99_000+s+diff_v*10, diff_v, s*137, max_steps)

    return all_rwd

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OBELIX Final Training")
    p.add_argument("--eps0",      type=int, default=4000)
    p.add_argument("--eps1",      type=int, default=4000)
    p.add_argument("--eps2",      type=int, default=3000)
    p.add_argument("--eps3",      type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--prefix",    type=str, default="q_final")
    p.add_argument("--box_speed", type=int, default=2)
    p.add_argument("--n_plan",    type=int, default=20)
    args = p.parse_args()
    train(**vars(args))