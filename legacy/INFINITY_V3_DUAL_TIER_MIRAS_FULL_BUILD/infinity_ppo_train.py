"""
infinity_ppo_train.py

PPO trainer wired to InfinityActorCritic (Dual-Tier Miras).

Default env: CartPole-v1 (short horizon sanity check).
You can swap to any Gym-like env (e.g., delayed-cue, transformer-killer tasks)
by changing ENV_ID and obs/action handling.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    import gym
except ImportError:
    gym = None

from continuous_memory_mamba_agent_v3_infinity import InfinityActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    learning_rate: float = 3e-4
    train_epochs: int = 10
    batch_size: int = 64
    steps_per_rollout: int = 2048
    hidden_dim: int = 256
    env_id: str = "CartPole-v1"
    max_iters: int = 50


def make_env(env_id: str):
    if gym is None:
        raise ImportError("gym is not installed. Please install gym to run this script.")
    return gym.make(env_id)


def compute_gae(rews, vals, dones, cfg: PPOConfig):
    rews = np.asarray(rews, dtype=np.float32)
    vals = np.asarray(vals, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    adv = np.zeros_like(rews, dtype=np.float32)
    ret = np.zeros_like(rews, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(len(rews))):
        delta = rews[t] + cfg.gamma * vals[t + 1] * (1.0 - dones[t]) - vals[t]
        gae = delta + cfg.gamma * cfg.lam * (1.0 - dones[t]) * gae
        adv[t] = gae
        ret[t] = gae + vals[t]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def ppo_train(cfg: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg.env_id)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = InfinityActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for it in range(cfg.max_iters):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        for step in range(cfg.steps_per_rollout):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            step_out = env.step(action.item())
            if len(step_out) == 5:
                next_obs, reward, done, truncated, _ = step_out
                done_flag = done or truncated
            else:
                next_obs, reward, done, _ = step_out
                done_flag = done

            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            rew_buf.append(float(reward))
            val_buf.append(value.item())
            done_buf.append(float(done_flag))

            obs = next_obs
            if done_flag:
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        # Bootstrap value
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_val = model(last_obs_t)

        vals = np.array(val_buf + [last_val.item()], dtype=np.float32)
        adv, ret = compute_gae(rew_buf, vals, done_buf, cfg)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.long, device=device)
        logp_old_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        idxs = np.arange(len(obs_buf))
        for epoch in range(cfg.train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), cfg.batch_size):
                mb_idx = idxs[start : start + cfg.batch_size]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_logp_old = logp_old_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, value = model(mb_obs, advantage=mb_adv)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)

                ratio = (logp - mb_logp_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)
                entropy_loss = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Simple evaluation
        eval_rews = []
        eval_episodes = 5
        for _ in range(eval_episodes):
            reset_out = env.reset()
            o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done_eval = False
            ep_r = 0.0
            while not done_eval:
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(o_t)
                    a = torch.argmax(logits, dim=-1).item()
                step_out = env.step(a)
                if len(step_out) == 5:
                    o, r, d, trunc, _ = step_out
                    done_eval = d or trunc
                else:
                    o, r, d, _ = step_out
                    done_eval = d
                ep_r += r
            eval_rews.append(ep_r)

        print(f"[Iter {it+1}/{cfg.max_iters}] Eval return mean = {np.mean(eval_rews):.2f}")

    env.close()


if __name__ == "__main__":
    cfg = PPOConfig()
    ppo_train(cfg)
