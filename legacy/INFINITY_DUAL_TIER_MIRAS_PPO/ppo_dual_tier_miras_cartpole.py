import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass

try:
    import gym
except ImportError:
    gym = None

from dual_tier_miras import SSMCompressedMiras, SSMCompressedMirasTitans, DualTierMiras


class MLPBackbone(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticWithMiras(nn.Module):
    """
    PPO actor-critic with Dual-Tier Miras as a parametric memory head.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = MLPBackbone(obs_dim, hidden_dim)
        self.miras_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.miras_val_proj = nn.Linear(hidden_dim, hidden_dim)

        fast = SSMCompressedMiras(d_model=hidden_dim, rank=16, lr=1e-3, l2_reg=1e-4)
        deep = SSMCompressedMirasTitans(
            d_model=hidden_dim,
            rank=16,
            lr=5e-4,
            l2_reg=1e-4,
            momentum=0.9,
            use_huber=True,
            huber_delta=1.0,
        )
        self.miras = DualTierMiras(
            fast_mem=fast,
            deep_mem=deep,
            d_model=hidden_dim,
            init_fast_weight=0.7,
            context_gate=True,
        )

        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        advantage: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        k = self.miras_key_proj(x)
        v_target = self.miras_val_proj(x).detach()
        context = x

        read_out = self.miras.read(k, context=context)
        v_hat = read_out["v"]
        x_aug = x + v_hat

        if self.training:
            w = advantage.detach() if advantage is not None else None
            self.miras.update(k, v_target, weight=w, context=context)

        logits = self.policy_head(x_aug)
        value = self.value_head(x_aug).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    learning_rate: float = 3e-4
    train_epochs: int = 10
    batch_size: int = 64
    steps_per_rollout: int = 2048


def ppo_train_cartpole(num_iterations: int = 50) -> None:
    if gym is None:
        raise ImportError("gym is not installed. Please install gym to run this script.")

    env = gym.make("CartPole-v1")
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PPOConfig()
    model = ActorCriticWithMiras(obs_dim, act_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for it in range(num_iterations):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        ep_rews = []
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

            ep_rews.append(reward)
            obs = next_obs

            if done_flag:
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                ep_rews = []

        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_val = model(last_obs_t)

        vals = np.array(val_buf + [last_val.item()], dtype=np.float32)
        rews = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        adv_buf = np.zeros_like(rews, dtype=np.float32)
        ret_buf = np.zeros_like(rews, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(len(rews))):
            delta = rews[t] + cfg.gamma * vals[t + 1] * (1.0 - dones[t]) - vals[t]
            gae = delta + cfg.gamma * cfg.lam * (1.0 - dones[t]) * gae
            adv_buf[t] = gae
            ret_buf[t] = gae + vals[t]

        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.long, device=device)
        logp_old_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

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

        print(f"[Iter {it+1}] Eval return mean = {np.mean(eval_rews):.2f}")

    env.close()


if __name__ == "__main__":
    ppo_train_cartpole()
