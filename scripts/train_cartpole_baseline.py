#!/usr/bin/env python3
"""
Baseline CartPole training - no memory systems.

This script trains a pure backbone agent (no Miras, no LTM)
for comparison against memory-augmented variants.
"""

import sys
sys.path.insert(0, "src")

from infinity_dual_hybrid.config import get_config_for_env
from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent
from infinity_dual_hybrid.ppo_trainer import PPOTrainer
from infinity_dual_hybrid.envs import make_envs


def main():
    cfg = get_config_for_env("CartPole-v1")

    # Pure backbone, no parametric memory, no LTM
    cfg.agent.use_miras_in_forward = False
    cfg.agent.use_ltm_in_forward = False
    cfg.agent.sync_dims()

    envs = make_envs(cfg.env_id, num_envs=1)
    agent = InfinityV3DualHybridAgent(cfg.agent).to(cfg.device)
    trainer = PPOTrainer(agent, cfg.ppo)

    print("=" * 50)
    print("Baseline CartPole Training (No Memory)")
    print("=" * 50)
    print(f"Device: {cfg.device}")
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print("=" * 50)

    for it in range(cfg.ppo.max_iterations):
        rollouts = trainer.collect_rollouts(envs)
        train_stats = trainer.train_step(rollouts)

        if it % 10 == 0:
            eval_stats = trainer.evaluate(envs, num_episodes=5)
            print(
                f"[{it:4d}] "
                f"reward={train_stats['mean_reward']:.2f} "
                f"loss={train_stats['policy_loss']:.4f} "
                f"eval={eval_stats['eval_mean_reward']:.2f}"
            )

    envs.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
