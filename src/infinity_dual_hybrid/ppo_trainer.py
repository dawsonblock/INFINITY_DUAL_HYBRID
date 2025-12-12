"""
ppo_trainer.py

PPO (Proximal Policy Optimization) Trainer for Infinity V3.

Features:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function loss with optional clipping
- Entropy bonus for exploration
- Optional KL penalty against old policy
- Multi-environment rollout collection
- Gradient clipping
- Learning rate scheduling

Usage:
    trainer = PPOTrainer(agent, cfg)
    for iteration in range(cfg.max_iterations):
        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np

import torch
import torch.nn as nn

from .config import PPOConfig
from .agent import InfinityV3DualHybridAgent


class RolloutBatch(NamedTuple):
    """Container for rollout data."""
    observations: torch.Tensor  # [N, obs_dim]
    actions: torch.Tensor       # [N]
    log_probs: torch.Tensor     # [N]
    values: torch.Tensor        # [N]
    rewards: torch.Tensor       # [N]
    dones: torch.Tensor         # [N]
    advantages: torch.Tensor    # [N]
    returns: torch.Tensor       # [N]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: [T] rewards
        values: [T+1] value estimates (includes bootstrap)
        dones: [T] episode termination flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    Returns:
        (advantages, returns) both [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = (
            rewards[t]
            + gamma * values[t + 1] * next_non_terminal
            - values[t]
        )
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


class PPOTrainer:
    """
    PPO Trainer for InfinityV3DualHybridAgent.

    Handles:
    - Rollout collection from environments
    - Advantage estimation with GAE
    - Policy and value optimization with clipping
    - Entropy bonus and optional KL penalty
    """

    def __init__(
        self,
        agent: InfinityV3DualHybridAgent,
        cfg: PPOConfig,
        device: str = "cpu",
    ):
        self.agent = agent
        self.cfg = cfg
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.current_lr = cfg.learning_rate

        # Optional: old policy for KL penalty
        self._old_policy_params = None
        if cfg.use_kl_penalty:
            self._store_old_policy()

        # v2.0: Adaptive KL target
        self.kl_coef = cfg.kl_coef

        # Stats tracking
        self.iteration = 0
        self.grad_explosion_count = 0

    def _store_old_policy(self) -> None:
        """Store copy of policy parameters for KL penalty."""
        self._old_policy_params = {
            name: param.clone().detach()
            for name, param in self.agent.named_parameters()
        }

    def collect_rollouts(
        self,
        envs: List,
        steps: Optional[int] = None,
    ) -> RolloutBatch:
        """
        Collect rollouts from environments.

        Args:
            envs: List of Gym-like environments
            steps: Number of steps per env (default: cfg.steps_per_rollout)
        Returns:
            RolloutBatch with collected data
        """
        steps = steps or self.cfg.steps_per_rollout
        num_envs = len(envs)

        # Storage
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        # Current observations
        current_obs = []
        for env in envs:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            current_obs.append(obs)

        self.agent.eval()

        store_for_ltm = False
        if self.agent.ltm is not None:
            store_for_ltm = bool(self.agent.cfg.ltm.store_on_episode_end)

        for step in range(steps):
            # Convert to tensor
            obs_t = torch.tensor(
                np.array(current_obs),
                dtype=torch.float32,
                device=self.device,
            )

            # Get actions
            with torch.no_grad():
                actions, log_probs, values = self.agent.get_action(
                    obs_t,
                    store_for_ltm=store_for_ltm,
                )

            # Step environments
            next_obs_list = []
            for i, env in enumerate(envs):
                action = actions[i].item()
                step_out = env.step(action)

                if len(step_out) == 5:
                    next_obs, reward, done, truncated, info = step_out
                    done_flag = done or truncated
                else:
                    next_obs, reward, done, info = step_out
                    done_flag = done

                obs_buf.append(current_obs[i])
                act_buf.append(action)
                logp_buf.append(log_probs[i].item())
                rew_buf.append(float(reward))
                val_buf.append(values[i].item())
                done_buf.append(float(done_flag))

                if done_flag:
                    # Reset and commit LTM
                    reset_out = env.reset()
                    if isinstance(reset_out, tuple):
                        next_obs = reset_out[0]
                    else:
                        next_obs = reset_out
                    self.agent.commit_to_ltm()
                    self.agent.reset_episode()

                next_obs_list.append(next_obs)

            current_obs = next_obs_list

        # Bootstrap value for last observations
        with torch.no_grad():
            last_obs_t = torch.tensor(
                np.array(current_obs),
                dtype=torch.float32,
                device=self.device,
            )
            _, _, last_values = self.agent.get_action(last_obs_t)

        # Compute advantages per environment
        all_advantages = []
        all_returns = []

        for env_idx in range(num_envs):
            # Extract this env's data
            env_rewards = np.array(
                rew_buf[env_idx::num_envs],
                dtype=np.float32,
            )
            env_values = np.array(
                val_buf[env_idx::num_envs] + [last_values[env_idx].item()],
                dtype=np.float32,
            )
            env_dones = np.array(done_buf[env_idx::num_envs], dtype=np.float32)

            # Compute GAE
            adv, ret = compute_gae(
                env_rewards,
                env_values,
                env_dones,
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )
            all_advantages.append(adv)
            all_returns.append(ret)

        # Interleave back
        advantages = np.zeros(len(obs_buf), dtype=np.float32)
        returns = np.zeros(len(obs_buf), dtype=np.float32)
        for env_idx in range(num_envs):
            advantages[env_idx::num_envs] = all_advantages[env_idx]
            returns[env_idx::num_envs] = all_returns[env_idx]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Create batch
        return RolloutBatch(
            observations=torch.tensor(np.array(obs_buf), dtype=torch.float32),
            actions=torch.tensor(np.array(act_buf), dtype=torch.long),
            log_probs=torch.tensor(np.array(logp_buf), dtype=torch.float32),
            values=torch.tensor(np.array(val_buf), dtype=torch.float32),
            rewards=torch.tensor(np.array(rew_buf), dtype=torch.float32),
            dones=torch.tensor(np.array(done_buf), dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
        )

    def train_step(self, rollouts: RolloutBatch) -> Dict[str, float]:
        """
        Perform PPO training step on collected rollouts.

        Args:
            rollouts: RolloutBatch from collect_rollouts
        Returns:
            Dict of training statistics
        """
        self.agent.train()
        cfg = self.cfg

        # Move to device
        obs = rollouts.observations.to(self.device)
        actions = rollouts.actions.to(self.device)
        old_log_probs = rollouts.log_probs.to(self.device)
        advantages = rollouts.advantages.to(self.device)
        returns = rollouts.returns.to(self.device)
        old_values = rollouts.values.to(self.device)

        num_samples = obs.shape[0]
        indices = np.arange(num_samples)

        # Stats accumulators
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_grad_norm = 0.0
        total_memory_gate_loss = 0.0
        num_updates = 0

        for epoch in range(cfg.train_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, cfg.batch_size):
                end = start + cfg.batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                _ = old_values[mb_idx]  # Reserved for value clipping

                # Evaluate actions
                new_logp, new_values, entropy, write_prob = (
                    self.agent.evaluate_actions(
                        mb_obs,
                        mb_actions,
                        advantage=mb_adv,
                    )
                )

                # Policy loss (clipped surrogate)
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                memory_gate_loss = -(
                    mb_adv.detach() * torch.log(write_prob + 1e-8)
                ).mean()

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                    + 0.1 * memory_gate_loss
                )

                # Optional KL penalty with adaptive coefficient
                if cfg.use_kl_penalty:
                    kl = (mb_old_logp - new_logp).mean()
                    loss = loss + self.kl_coef * kl
                    total_kl += kl.item()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Track gradient norm before clipping
                grad_norm = 0.0
                if cfg.track_grad_norm:
                    for p in self.agent.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    total_grad_norm += grad_norm

                # Gradient explosion detection and LR reduction
                if grad_norm > cfg.grad_explosion_threshold:
                    self.grad_explosion_count += 1
                    if self.grad_explosion_count >= 3:
                        self.current_lr *= cfg.lr_reduce_factor
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = self.current_lr
                        self.grad_explosion_count = 0

                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_memory_gate_loss += memory_gate_loss.item()
                num_updates += 1

        # Adaptive KL coefficient adjustment
        if cfg.use_kl_penalty and cfg.adaptive_kl:
            avg_kl = total_kl / max(num_updates, 1)
            if avg_kl > cfg.kl_target * cfg.kl_adapt_coef:
                self.kl_coef *= 1.5
            elif avg_kl < cfg.kl_target / cfg.kl_adapt_coef:
                self.kl_coef *= 0.5
            self.kl_coef = max(0.0001, min(self.kl_coef, 10.0))

        # Update old policy if using KL
        if cfg.use_kl_penalty:
            self._store_old_policy()

        self.iteration += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / max(num_updates, 1),
            "grad_norm": total_grad_norm / max(num_updates, 1),
            "memory_gate_loss": total_memory_gate_loss / num_updates,
            "learning_rate": self.current_lr,
            "kl_coef": self.kl_coef,
            "mean_reward": rollouts.rewards.mean().item(),
            "mean_return": rollouts.returns.mean().item(),
            "mean_advantage": rollouts.advantages.mean().item(),
        }

    def evaluate(
        self,
        envs: List,
        num_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent on environments.

        Args:
            envs: List of environments
            num_episodes: Episodes to run
            deterministic: Use deterministic policy
        Returns:
            Dict with evaluation stats
        """
        self.agent.eval()

        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            env = envs[0]  # Use first env
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action, _, _ = self.agent.get_action(
                        obs_t,
                        deterministic=deterministic,
                    )

                step_out = env.step(action[0].item())
                if len(step_out) == 5:
                    obs, reward, done, truncated, _ = step_out
                    done = done or truncated
                else:
                    obs, reward, done, _ = step_out

                ep_reward += reward
                ep_length += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            self.agent.reset_episode()

        return {
            "eval_mean_reward": np.mean(episode_rewards),
            "eval_std_reward": np.std(episode_rewards),
            "eval_mean_length": np.mean(episode_lengths),
        }

    def save(self, path: str) -> None:
        """Save trainer state."""
        state = {
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "config": self.cfg,
        }
        if self.agent.ltm is not None:
            state["ltm"] = self.agent.ltm.state_dict_ltm()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load trainer state."""
        state = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state["agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state["iteration"]
        if "ltm" in state and self.agent.ltm is not None:
            self.agent.ltm.load_state_dict_ltm(state["ltm"])
