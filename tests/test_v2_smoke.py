#!/usr/bin/env python3
"""
v2.0 Smoke Tests - End-to-end PPO training sanity checks.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "vendor")

import torch
import pytest


class TestPPOSmoke:
    """Smoke tests for PPO training pipeline."""

    def test_ppo_single_iteration(self):
        """PPO step runs end-to-end on CartPole for 1 iteration."""
        from infinity_dual_hybrid.config import get_config_for_env
        from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent
        from infinity_dual_hybrid.ppo_trainer import PPOTrainer
        from infinity_dual_hybrid.envs import make_envs

        cfg = get_config_for_env("CartPole-v1")
        cfg.agent.use_miras_in_forward = False
        cfg.agent.use_ltm_in_forward = False
        cfg.agent.sync_dims()

        envs = make_envs(cfg.env_id, num_envs=1)
        agent = InfinityV3DualHybridAgent(cfg.agent)
        trainer = PPOTrainer(agent, cfg.ppo)

        # Single rollout + train step
        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)

        for env in envs:
            env.close()

        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "mean_reward" in stats
        assert not torch.isnan(torch.tensor(stats["policy_loss"]))

    def test_ppo_with_miras(self):
        """PPO step with Miras memory enabled."""
        from infinity_dual_hybrid.config import get_config_for_env
        from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent
        from infinity_dual_hybrid.ppo_trainer import PPOTrainer
        from infinity_dual_hybrid.envs import make_envs

        cfg = get_config_for_env("CartPole-v1")
        cfg.agent.use_miras_in_forward = True
        cfg.agent.use_ltm_in_forward = False
        cfg.agent.sync_dims()

        envs = make_envs(cfg.env_id, num_envs=1)
        agent = InfinityV3DualHybridAgent(cfg.agent)
        trainer = PPOTrainer(agent, cfg.ppo)

        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)

        for env in envs:
            env.close()

        assert "policy_loss" in stats
        assert not torch.isnan(torch.tensor(stats["policy_loss"]))


class TestBackboneFallback:
    """Test backbone fallback when Mamba unavailable."""

    def test_attention_only_fallback(self):
        """Runs without Mamba using attention-only fallback."""
        from infinity_dual_hybrid.config import get_config_for_env
        from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent

        cfg = get_config_for_env("CartPole-v1")
        cfg.agent.backbone.use_mamba = False
        cfg.agent.backbone.use_attention = True
        cfg.agent.sync_dims()

        agent = InfinityV3DualHybridAgent(cfg.agent)
        obs = torch.randn(4, cfg.agent.obs_dim)

        with torch.no_grad():
            out = agent(obs)
            logits, values = out["logits"], out["value"]

        assert logits.shape == (4, cfg.agent.act_dim)
        assert values.shape == (4,)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(values).all()

    def test_mlp_only_fallback(self):
        """Runs with MLP-only backbone."""
        from infinity_dual_hybrid.config import get_config_for_env
        from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent

        cfg = get_config_for_env("CartPole-v1")
        cfg.agent.backbone.use_mamba = False
        cfg.agent.backbone.use_attention = False
        cfg.agent.sync_dims()

        agent = InfinityV3DualHybridAgent(cfg.agent)
        obs = torch.randn(4, cfg.agent.obs_dim)

        with torch.no_grad():
            out = agent(obs)
            logits, values = out["logits"], out["value"]

        assert logits.shape == (4, cfg.agent.act_dim)
        assert values.shape == (4,)


class TestSeedReproducibility:
    """Test seed reproducibility."""

    def test_deterministic_forward(self):
        """Same seed produces same outputs."""
        from infinity_dual_hybrid.config import get_config_for_env
        from infinity_dual_hybrid.agent import InfinityV3DualHybridAgent

        cfg = get_config_for_env("CartPole-v1")
        cfg.agent.backbone.use_mamba = False
        cfg.agent.sync_dims()

        # Run 1
        torch.manual_seed(42)
        agent1 = InfinityV3DualHybridAgent(cfg.agent)
        obs = torch.randn(4, cfg.agent.obs_dim)
        torch.manual_seed(123)
        with torch.no_grad():
            result1 = agent1(obs)
            out1, val1 = result1["logits"], result1["value"]

        # Run 2
        torch.manual_seed(42)
        agent2 = InfinityV3DualHybridAgent(cfg.agent)
        torch.manual_seed(123)
        with torch.no_grad():
            result2 = agent2(obs)
            out2, val2 = result2["logits"], result2["value"]

        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(val1, val2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
