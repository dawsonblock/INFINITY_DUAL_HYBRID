"""
train.py

CLI entry point for training Infinity V3 Dual Hybrid Agent.

Usage:
    python -m infinity_v3_dual_hybrid.train --env CartPole-v1 --iterations 50
    python -m infinity_v3_dual_hybrid.train --config config.yaml

This script:
1. Loads configuration
2. Creates environments
3. Builds the agent
4. Trains with PPO
5. Saves checkpoints and logs progress
"""

import argparse
import json
import os
import random
import shlex
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import torch
import numpy as np
from gymnasium.envs.registration import register

from .config import TrainConfig, get_config_for_env
from .agent import InfinityV3DualHybridAgent
from .ppo_trainer import PPOTrainer
from .envs import make_envs, get_env_info
from .logger import create_logger


def set_seed(seed: Optional[int]) -> None:
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_run_name(s: str) -> str:
    safe = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_ ")
    return out or "run"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data) -> None:
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def _write_run_commands(path: str) -> None:
    cmd = "PYTHONPATH=src python3 -m infinity_dual_hybrid.train " + " ".join(
        shlex.quote(a) for a in sys.argv[1:]
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        cmd,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def train(
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
) -> InfinityV3DualHybridAgent:
    """
    Main training loop.

    Args:
        cfg: Training configuration
    Returns:
        Trained agent
    """
    print("=" * 60)
    print("Infinity V3 Dual Hybrid Training")
    print("=" * 60)
    print(f"Environment: {cfg.env_id}")
    print(f"Device: {cfg.device}")
    print(f"Iterations: {cfg.ppo.max_iterations}")
    if out_dir:
        print(f"Output dir: {out_dir}")
    print("=" * 60)

    if cfg.env_register_local:
        try:
            from gymnasium.envs.registration import registry as gym_registry
        except Exception:
            gym_registry = None

        if gym_registry is None or "DelayedCue-v0" not in gym_registry:
            try:
                register(
                    id="DelayedCue-v0",
                    entry_point="infinity_dual_hybrid.envs:DelayedCueEnv",
                )
            except Exception:
                pass

        if (
            gym_registry is None
            or "DelayedCueRegime-v0" not in gym_registry
        ):
            try:
                register(
                    id="DelayedCueRegime-v0",
                    entry_point="infinity_dual_hybrid.envs:DelayedCueRegimeEnv",
                )
            except Exception:
                pass

    # Set seed
    set_seed(cfg.seed)

    # Get environment info and update config
    env_info = get_env_info(cfg.env_id, cfg=cfg)
    cfg.agent.obs_dim = env_info["obs_dim"]
    cfg.agent.act_dim = env_info["act_dim"]

    print(f"Observation dim: {cfg.agent.obs_dim}")
    print(f"Action dim: {cfg.agent.act_dim}")
    print("=" * 60)

    # Create environments
    envs = make_envs(cfg.env_id, cfg.ppo.num_envs, cfg=cfg)

    # Create agent
    agent = InfinityV3DualHybridAgent(cfg.agent).to(cfg.device)

    # Count parameters
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = PPOTrainer(agent, cfg.ppo, device=cfg.device, seed=cfg.seed)

    # Create save directory
    if cfg.save_path:
        os.makedirs(cfg.save_path, exist_ok=True)

    # Training loop
    start_time = time.time()
    best_eval_reward = float("-inf")

    if out_dir:
        _write_json(os.path.join(out_dir, "config.json"), asdict(cfg))

    logger_cm = (
        create_logger(log_dir=out_dir, experiment_name="")
        if out_dir
        else create_logger()
    )
    with logger_cm as logger:
        for iteration in range(1, cfg.ppo.max_iterations + 1):
            iter_start = time.time()

            # Collect rollouts
            rollouts = trainer.collect_rollouts(envs)

            # Train
            train_stats = trainer.train_step(rollouts)

            # Evaluate
            eval_stats = {}
            if iteration % cfg.ppo.eval_interval == 0:
                eval_stats = trainer.evaluate(
                    envs,
                    num_episodes=cfg.ppo.eval_episodes,
                )

                # Track best
                if eval_stats["eval_mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_stats["eval_mean_reward"]
                    if cfg.save_path:
                        trainer.save(
                            os.path.join(cfg.save_path, "best_model.pt")
                        )

            # Log
            if iteration % cfg.log_interval == 0:
                iter_time = time.time() - iter_start

                dbg = agent.debug_state()
                miras_dbg = dbg.get("miras")
                ltm_dbg = dbg.get("ltm")
                backbone_dbg = dbg.get("backbone")

                metrics = {
                    "iteration_time": iter_time,
                    "reward_std": float(rollouts.rewards.std().item()),
                    "return_std": float(rollouts.returns.std().item()),
                    "eval_mean_reward": None,
                    "eval_std_reward": None,
                    "eval_mean_length": None,
                    "mode": dbg.get("mode"),
                    "temperature": dbg.get("temperature"),
                    "backbone_has_mamba": (
                        backbone_dbg.get("has_mamba") if backbone_dbg else None
                    ),
                    "backbone_has_attention": (
                        backbone_dbg.get("has_attention")
                        if backbone_dbg
                        else None
                    ),
                    "miras_fast_B_norm": (
                        miras_dbg.get("fast_B_norm") if miras_dbg else None
                    ),
                    "miras_fast_C_norm": (
                        miras_dbg.get("fast_C_norm") if miras_dbg else None
                    ),
                    "miras_deep_B_norm": (
                        miras_dbg.get("deep_B_norm") if miras_dbg else None
                    ),
                    "miras_deep_C_norm": (
                        miras_dbg.get("deep_C_norm") if miras_dbg else None
                    ),
                    "miras_mix_ratio": (
                        miras_dbg.get("mix_ratio") if miras_dbg else None
                    ),
                    "miras_fast_err_l2": (
                        miras_dbg.get("fast_err_l2") if miras_dbg else None
                    ),
                    "miras_deep_err_l2": (
                        miras_dbg.get("deep_err_l2") if miras_dbg else None
                    ),
                    "miras_deep_retention": (
                        miras_dbg.get("deep_retention") if miras_dbg else None
                    ),
                    "miras_deep_gradB_norm": (
                        miras_dbg.get("deep_gradB_norm") if miras_dbg else None
                    ),
                    "miras_deep_gradC_norm": (
                        miras_dbg.get("deep_gradC_norm") if miras_dbg else None
                    ),
                    "ltm_size": (ltm_dbg.get("size") if ltm_dbg else 0),
                    **train_stats,
                    **eval_stats,
                }

                logger.log(metrics, step=iteration)

                log_str = (
                    f"[Iter {iteration:4d}/{cfg.ppo.max_iterations}] "
                    f"policy_loss={train_stats['policy_loss']:.4f} "
                    f"value_loss={train_stats['value_loss']:.4f} "
                    f"entropy={train_stats['entropy']:.4f} "
                    f"mean_step_rew={train_stats['mean_step_reward']:.2f}"
                )
                if eval_stats:
                    log_str += f" | eval={eval_stats['eval_mean_reward']:.2f}"
                log_str += f" | time={iter_time:.2f}s"

                print(log_str)

            # Save checkpoint
            if cfg.save_path and iteration % cfg.save_interval == 0:
                trainer.save(
                    os.path.join(cfg.save_path, f"checkpoint_{iteration}.pt")
                )

    # Final save
    if cfg.save_path:
        trainer.save(os.path.join(cfg.save_path, "final_model.pt"))

    # Cleanup
    for env in envs:
        env.close()
    agent.shutdown()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training complete in {total_time:.2f}s")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print("=" * 60)

    if out_dir:
        _write_json(
            os.path.join(out_dir, "metrics.json"),
            {
                "env_id": str(cfg.env_id),
                "seed": cfg.seed,
                "device": str(cfg.device),
                "max_iterations": int(cfg.ppo.max_iterations),
                "best_eval_reward": float(best_eval_reward),
                "total_time_seconds": float(total_time),
            },
        )

    return agent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Infinity V3 Dual Hybrid Agent"
    )

    # Environment
    parser.add_argument(
        "--env", type=str, default="CartPole-v1",
        help="Gym environment ID"
    )

    # Training
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--steps-per-rollout", type=int, default=2048,
        help="Steps per rollout"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel environments"
    )

    # Model
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--use-mamba", action="store_true", default=False,
        help="Use Mamba backbone (requires mamba-ssm)"
    )
    parser.add_argument(
        "--use-ltm", action="store_true", default=False,
        help="Use LTM (episodic memory)"
    )
    parser.add_argument(
        "--use-faiss", action="store_true", default=False,
        help="Use FAISS for LTM (requires faiss-cpu)"
    )

    # Optimization
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="PPO epochs per iteration"
    )

    # Other
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--save-path", type=str, default="checkpoints",
        help="Save directory"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=1,
        help="Evaluation interval"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Base results directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (subdir under --out)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    run_name = _sanitize_run_name(args.run_name or f"train_{args.env}")
    tag = _now_tag()
    out_dir = os.path.join(str(args.out), run_name, tag)
    _ensure_dir(out_dir)
    _write_run_commands(os.path.join(out_dir, "run_commands.sh"))

    # Build config from args
    cfg = get_config_for_env(args.env)
    cfg.env_id = args.env
    cfg.device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg.seed = args.seed
    if os.path.isabs(str(args.save_path)):
        cfg.save_path = str(args.save_path)
    else:
        cfg.save_path = os.path.join(out_dir, str(args.save_path))

    # PPO config
    cfg.ppo.max_iterations = args.iterations
    cfg.ppo.steps_per_rollout = args.steps_per_rollout
    cfg.ppo.num_envs = args.num_envs
    cfg.ppo.learning_rate = args.lr
    cfg.ppo.batch_size = args.batch_size
    cfg.ppo.train_epochs = args.epochs
    cfg.ppo.eval_interval = args.eval_interval

    # Agent config
    cfg.agent.hidden_dim = args.hidden_dim
    cfg.agent.backbone.use_mamba = args.use_mamba
    cfg.agent.use_ltm_in_forward = args.use_ltm
    cfg.agent.ltm.use_faiss = args.use_faiss
    cfg.agent.sync_dims()  # Sync dimensions after modifications

    # Train
    train(cfg, out_dir=out_dir)


def quick_test():
    """Quick sanity test - one forward pass and optimization step."""
    print("Running quick sanity test...")

    # Minimal config
    cfg = TrainConfig()
    cfg.env_id = "CartPole-v1"
    cfg.agent.obs_dim = 4
    cfg.agent.act_dim = 2
    cfg.agent.hidden_dim = 64
    cfg.agent.backbone.use_mamba = False
    cfg.agent.backbone.use_attention = False  # Pure MLP for test
    cfg.agent.use_ltm_in_forward = False
    cfg.agent.use_miras_in_forward = True
    cfg.agent.sync_dims()  # Sync after changing hidden_dim
    cfg.device = "cpu"

    # Create agent
    agent = InfinityV3DualHybridAgent(cfg.agent)
    num_params = sum(p.numel() for p in agent.parameters())
    print(
        f"Agent created with {num_params:,} params"
    )

    # Dummy forward pass
    obs = torch.randn(4, 4)  # [B=4, obs_dim=4]
    out = agent(obs)
    logits_shape = out["logits"].shape
    value_shape = out["value"].shape
    print(
        f"Forward pass: logits={logits_shape}, value={value_shape}"
    )

    # Dummy optimization step
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    loss = out["logits"].mean() + out["value"].mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Optimization step completed, loss={loss.item():.4f}")

    print("Sanity test PASSED!")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        quick_test()
    else:
        main()
