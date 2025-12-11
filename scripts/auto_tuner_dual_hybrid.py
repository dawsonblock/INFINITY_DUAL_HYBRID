#!/usr/bin/env python3
"""
Simple auto-tuner for Infinity Dual Hybrid.

Scans configurations to find good hyperparameters.
"""

import sys
sys.path.insert(0, "src")

import itertools
from copy import deepcopy

from infinity_dual_hybrid.config import get_config_for_env
from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent
from infinity_dual_hybrid.ppo_trainer import PPOTrainer
from infinity_dual_hybrid.envs import make_envs


def run_trial(base_cfg, hidden_dim, use_miras, use_attention, num_iters=10):
    """Run a single trial with given config."""
    cfg = deepcopy(base_cfg)
    cfg.agent.hidden_dim = hidden_dim
    cfg.agent.use_miras_in_forward = use_miras
    cfg.agent.backbone.use_attention = use_attention
    cfg.agent.use_ltm_in_forward = False
    cfg.agent.sync_dims()

    envs = make_envs(cfg.env_id, num_envs=1)
    agent = InfinityV3DualHybridAgent(cfg.agent).to(cfg.device)
    trainer = PPOTrainer(agent, cfg.ppo)

    best_return = -1e9
    for it in range(num_iters):
        rollouts = trainer.collect_rollouts(envs)
        trainer.train_step(rollouts)
        eval_stats = trainer.evaluate(envs, num_episodes=3)
        best_return = max(best_return, eval_stats["eval_mean_reward"])

    envs.close()
    return best_return


def main():
    base_cfg = get_config_for_env("CartPole-v1")

    # Search space
    hidden_dims = [64, 128, 256]
    use_miras_opts = [False, True]
    use_attention_opts = [False, True]

    print("=" * 60)
    print("Infinity Dual Hybrid Auto-Tuner")
    print("=" * 60)

    results = []
    for hidden, miras, attn in itertools.product(
        hidden_dims, use_miras_opts, use_attention_opts
    ):
        print(f"Testing: hidden={hidden}, miras={miras}, attn={attn}...", end=" ")
        score = run_trial(base_cfg, hidden, miras, attn, num_iters=15)
        results.append((hidden, miras, attn, score))
        print(f"score={score:.2f}")

    # Sort by score
    results.sort(key=lambda x: x[3], reverse=True)

    print("\n" + "=" * 60)
    print("Results (sorted by score)")
    print("=" * 60)
    for hidden, miras, attn, score in results:
        print(f"hidden={hidden:3d} miras={str(miras):5s} attn={str(attn):5s} -> {score:.2f}")

    best = results[0]
    print(f"\nBest config: hidden={best[0]}, miras={best[1]}, attn={best[2]}")


if __name__ == "__main__":
    main()
