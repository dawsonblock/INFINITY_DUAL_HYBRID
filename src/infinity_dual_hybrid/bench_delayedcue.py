import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import get_config_for_env
from .agent import InfinityV3DualHybridAgent
from .ppo_trainer import PPOTrainer
from .envs import make_envs


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_run_commands(
    path: str,
    out_dir: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "PYTHONPATH=src python3 -m infinity_dual_hybrid.bench_delayedcue \\",
        f"  --out {out_dir} \\",
        f"  --seed {args.seed} \\",
        f"  --device {args.device} \\",
        f"  --train-iterations {args.train_iterations} \\",
        f"  --episodes {args.episodes} \\",
        f"  --delays \"{args.delays}\"",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = (x0.std() * y0.std()) + 1e-8
    return float((x0 * y0).mean() / denom)


def _run_train_for_env(
    env_id: str,
    cfg_overrides: Dict,
    iterations: int,
    device: str,
    seed: Optional[int],
) -> Tuple[InfinityV3DualHybridAgent, Dict[str, List[float]]]:
    cfg = get_config_for_env(env_id)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    cfg.device = device
    cfg.seed = int(seed) if seed is not None else None
    cfg.ppo.max_iterations = iterations
    cfg.ppo.num_envs = 1

    if cfg.seed is not None:
        set_seed(int(cfg.seed))

    envs = make_envs(cfg.env_id, num_envs=cfg.ppo.num_envs, cfg=cfg)

    agent = InfinityV3DualHybridAgent(cfg.agent).to(cfg.device)
    trainer = PPOTrainer(agent, cfg.ppo, device=cfg.device, seed=cfg.seed)

    stats_hist: Dict[str, List[float]] = {
        "mean_reward": [],
        "mean_write_prob": [],
        "mean_adv_used": [],
        "write_prob_adv_corr": [],
        "effective_write_mean": [],
    }

    for _ in range(cfg.ppo.max_iterations):
        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)
        for k in list(stats_hist.keys()):
            if k in stats:
                stats_hist[k].append(float(stats[k]))

    for env in envs:
        env.close()
    agent.shutdown()

    return agent, stats_hist


def _eval_success_rate(
    env_id: str,
    cfg_overrides: Dict,
    agent: InfinityV3DualHybridAgent,
    episodes: int,
    device: str,
    seed: Optional[int],
) -> Tuple[float, float, float]:
    cfg = get_config_for_env(env_id)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    cfg.device = device
    envs = make_envs(cfg.env_id, num_envs=1, cfg=cfg)
    env = envs[0]

    rewards = []
    successes = []
    times = []

    agent.eval()

    for ep in range(episodes):
        if seed is not None:
            try:
                reset_out = env.reset(seed=int(seed) + int(ep))
            except TypeError:
                reset_out = env.reset()
        else:
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        ep_reward = 0.0
        t = 0
        success = 0
        time_to_success = None

        while not done:
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_t, deterministic=True)

            step_out = env.step(int(action.item()))
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out

            ep_reward += float(reward)
            t += 1

            if float(reward) >= 10.0:
                success = 1
                if time_to_success is None:
                    time_to_success = t

        agent.reset_episode()

        rewards.append(ep_reward)
        successes.append(success)
        if time_to_success is None:
            time_to_success = float(cfg.delayedcue_episode_len)
        times.append(float(time_to_success))

    env.close()
    agent.reset_episode()

    return (
        float(np.mean(rewards)),
        float(np.mean(successes)),
        float(np.mean(times)),
    )


def run_delay_sweep(
    out_dir: str,
    delays: List[int],
    episodes: int,
    train_iterations: int,
    device: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    _ensure_dir(out_dir)

    sweep_rewards = []
    sweep_success = []
    sweep_tts = []

    corr_points_adv = []
    corr_points_gate = []

    for d in delays:
        cfg_overrides = {
            "delayedcue_delay": int(d),
        }
        agent, hist = _run_train_for_env(
            env_id="DelayedCue-v0",
            cfg_overrides=cfg_overrides,
            iterations=train_iterations,
            device=device,
            seed=seed,
        )

        mean_rew, success_rate, mean_tts = _eval_success_rate(
            env_id="DelayedCue-v0",
            cfg_overrides=cfg_overrides,
            agent=agent,
            episodes=episodes,
            device=device,
            seed=seed,
        )
        sweep_rewards.append(mean_rew)
        sweep_success.append(success_rate)
        sweep_tts.append(mean_tts)

        if hist["mean_adv_used"] and hist["mean_write_prob"]:
            corr_points_adv.append(hist["mean_adv_used"][-1])
            corr_points_gate.append(hist["mean_write_prob"][-1])

    plt.figure(figsize=(7, 4))
    plt.plot(delays, sweep_rewards, marker="o")
    plt.xlabel("delay")
    plt.ylabel("mean reward")
    plt.title("Reward vs Delay")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_vs_delay.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    if corr_points_adv:
        plt.scatter(corr_points_adv, corr_points_gate, s=24, alpha=0.8)
        corr = _pearson_corr(
            np.array(corr_points_adv),
            np.array(corr_points_gate),
        )
        plt.title(f"Advantage vs WriteProb (corr={corr:.3f})")
    else:
        plt.title("Advantage vs WriteProb")
    plt.xlabel("mean adv_used")
    plt.ylabel("mean write_prob")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adv_vs_writeprob.png"))
    plt.close()

    return {
        "delays": [int(x) for x in delays],
        "mean_reward": [float(x) for x in sweep_rewards],
        "success_rate": [float(x) for x in sweep_success],
        "mean_time_to_success": [float(x) for x in sweep_tts],
    }


def run_regime_shift_eval(
    out_dir: str,
    episodes: int,
    train_iterations: int,
    device: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    _ensure_dir(out_dir)

    agent, _ = _run_train_for_env(
        env_id="DelayedCueRegime-v0",
        cfg_overrides={},
        iterations=train_iterations,
        device=device,
        seed=seed,
    )

    cfg = get_config_for_env("DelayedCueRegime-v0")
    cfg.device = device
    envs = make_envs(cfg.env_id, num_envs=1, cfg=cfg)
    env = envs[0]

    rolling_rewards = []

    agent.eval()

    for ep in range(episodes):
        if seed is not None:
            try:
                reset_out = env.reset(seed=int(seed) + 10_000 + int(ep))
            except TypeError:
                reset_out = env.reset()
        else:
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_t, deterministic=True)

            step_out = env.step(int(action.item()))
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out

            ep_reward += float(reward)

        rolling_rewards.append(ep_reward)
        agent.reset_episode()

    env.close()
    agent.shutdown()

    w = min(20, max(1, len(rolling_rewards) // 10))
    smoothed = []
    for i in range(len(rolling_rewards)):
        lo = max(0, i - w + 1)
        smoothed.append(float(np.mean(rolling_rewards[lo: i + 1])))

    plt.figure(figsize=(7, 4))
    plt.plot(smoothed)
    plt.xlabel("episode")
    plt.ylabel("rolling mean reward")
    plt.title("Regime-Shift Recovery")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regime_shift_recovery.png"))
    plt.close()

    return {
        "rolling_reward": [float(x) for x in rolling_rewards],
        "smoothed_reward": [float(x) for x in smoothed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--delays",
        type=str,
        default="50,100,250,500,1000,2000",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train-iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tag = _now_tag()
    out_dir = args.out or os.path.join("results", "bench_delayedcue", tag)
    _ensure_dir(out_dir)
    _write_run_commands(
        os.path.join(out_dir, "run_commands.sh"),
        out_dir,
        args,
    )

    set_seed(int(args.seed))

    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]

    _write_json(
        os.path.join(out_dir, "config.json"),
        {
            **vars(args),
            "out_dir": out_dir,
            "delays": delays,
        },
    )

    delay_metrics = run_delay_sweep(
        out_dir=out_dir,
        delays=delays,
        episodes=args.episodes,
        train_iterations=args.train_iterations,
        device=args.device,
        seed=int(args.seed),
    )

    regime_metrics = run_regime_shift_eval(
        out_dir=out_dir,
        episodes=args.episodes,
        train_iterations=args.train_iterations,
        device=args.device,
        seed=int(args.seed),
    )

    _write_json(
        os.path.join(out_dir, "metrics.json"),
        {
            "seed": int(args.seed),
            "device": str(args.device),
            "delay_sweep": delay_metrics,
            "regime_shift": regime_metrics,
        },
    )


if __name__ == "__main__":
    main()
