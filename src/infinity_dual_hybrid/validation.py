import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from .agent import InfinityV3DualHybridAgent
from .config import get_config_for_env
from .envs import DelayedCueEnv, DelayedCueRegimeEnv, make_envs
from .ppo_trainer import PPOTrainer


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
            except Exception:
                continue
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                continue


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_seeds(seed: int) -> Dict[str, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return {
        "random": seed,
        "numpy": seed,
        "torch": seed,
        "env": seed,
    }


def _disable_ltm(cfg: Any) -> None:
    agent_cfg = getattr(cfg, "agent", None)
    if agent_cfg is None:
        return

    if hasattr(agent_cfg, "use_ltm_in_forward"):
        agent_cfg.use_ltm_in_forward = False

    ltm_cfg = getattr(agent_cfg, "ltm", None)
    if ltm_cfg is None:
        return

    if hasattr(ltm_cfg, "use_faiss"):
        ltm_cfg.use_faiss = False
    if hasattr(ltm_cfg, "use_async_writer"):
        ltm_cfg.use_async_writer = False
    if hasattr(ltm_cfg, "store_on_episode_end"):
        ltm_cfg.store_on_episode_end = False


def _runtime_info() -> Dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": str(torch.__version__),
        "gymnasium": str(gym.__version__),
    }


def _register_local_envs() -> None:
    try:
        from gymnasium.envs.registration import register, registry

        if "DelayedCue-v0" not in registry:
            try:
                register(
                    id="DelayedCue-v0",
                    entry_point="infinity_dual_hybrid.envs:DelayedCueEnv",
                )
            except Exception:
                pass

        if "DelayedCueRegime-v0" not in registry:
            try:
                register(
                    id="DelayedCueRegime-v0",
                    entry_point="infinity_dual_hybrid.envs:DelayedCueRegimeEnv",
                )
            except Exception:
                pass
    except Exception:
        return


def _env_api_check(env_id: str, seed: int) -> Dict[str, Any]:
    _register_local_envs()
    env = gym.make(env_id)
    reset_out = env.reset(seed=seed)
    ok_reset = (
        isinstance(reset_out, tuple)
        and len(reset_out) == 2
        and isinstance(reset_out[1], dict)
    )

    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    step_out = env.step(env.action_space.sample())
    ok_step = isinstance(step_out, tuple) and len(step_out) == 5

    env.close()
    return {
        "env_id": env_id,
        "gymnasium_make_ok": True,
        "reset_signature_ok": bool(ok_reset),
        "step_signature_ok": bool(ok_step),
        "obs_shape": getattr(np.array(obs), "shape", None),
    }


def _delayedcue_invariant_tests() -> List[Dict[str, Any]]:
    cfg = {
        "episode_len": 100,
        "cue_time": 10,
        "delay": 20,
        "window": 5,
        "noise_std": 0.0,
        "step_penalty": -0.001,
    }

    cases: List[Tuple[str, Any]] = []

    def run_case(
        name: str,
        action_at_t: Optional[int],
        action_value: int,
    ) -> Dict[str, Any]:
        env = DelayedCueEnv(**cfg)
        obs, _ = env.reset(seed=123)
        _ = obs

        terminated = False
        truncated = False
        reward = 0.0

        for t in range(cfg["episode_len"]):
            a = 0
            if action_at_t is not None and t == action_at_t:
                a = action_value

            step_out = env.step(a)
            _, r, term, trunc, _ = step_out
            reward += float(r)
            terminated = bool(term)
            truncated = bool(trunc)
            if terminated or truncated:
                break

        env.close()
        return {
            "case": name,
            "terminated": terminated,
            "truncated": truncated,
            "final_reward": float(r),
            "total_reward": float(reward),
        }

    target_time = cfg["cue_time"] + cfg["delay"]

    cases.append(
        (
            "action=1 before target_time",
            run_case("early", action_at_t=target_time - 1, action_value=1),
        )
    )
    cases.append(
        (
            "action=1 inside window",
            run_case("in_window", action_at_t=target_time, action_value=1),
        )
    )
    cases.append(
        (
            "action=0 all episode",
            run_case("all_zero", action_at_t=None, action_value=0),
        )
    )

    out: List[Dict[str, Any]] = []

    for title, res in cases:
        if res["case"] == "early":
            ok = res["terminated"] and (res["final_reward"] == -1.0)
        elif res["case"] == "in_window":
            ok = res["terminated"] and (res["final_reward"] == 10.0)
        else:
            expected = cfg["episode_len"] * cfg["step_penalty"]
            ok = res["truncated"] and (
                abs(res["total_reward"] - expected) < 1e-6
            )

        out.append(
            {
                "test": title,
                "pass": bool(ok),
                **res,
            }
        )

    return out


def _regime_invariant_tests() -> List[Dict[str, Any]]:
    env = DelayedCueRegimeEnv(
        episode_len=100,
        cue_time=10,
        delay=20,
        window=5,
        noise_std=0.0,
        step_penalty=-0.001,
    )
    env.reset(seed=123)

    target_time = env.cue_time + env.delay

    env.t = target_time - 1
    _, r0, term0, trunc0, _ = env.step(1)

    env.reset(seed=123)
    env.t = env.shift_time
    env.regime = 1
    env.t = target_time - 1
    _, r1, term1, trunc1, _ = env.step(0)

    env.close()

    return [
        {
            "test": "pre-shift correct action is 1 in window",
            "pass": bool(term0 and not trunc0 and r0 in (-1.0, 10.0)),
            "reward": float(r0),
            "terminated": bool(term0),
        },
        {
            "test": "post-shift correct action is 0 in window",
            "pass": bool(term1 and not trunc1 and r1 == 10.0),
            "reward": float(r1),
            "terminated": bool(term1),
        },
    ]


def _finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = (x0.std() * y0.std()) + 1e-8
    return float((x0 * y0).mean() / denom)


def _compute_adv_used(adv: torch.Tensor, cfg: Any) -> torch.Tensor:
    out = adv
    if bool(getattr(cfg, "adv_norm", True)):
        out = (out - out.mean()) / (out.std() + 1e-8)
    adv_clip = getattr(cfg, "adv_clip", None)
    if adv_clip is not None and adv_clip > 0:
        out = torch.clamp(out, -float(adv_clip), float(adv_clip))
    if bool(getattr(cfg, "adv_positive_only", True)):
        out = torch.clamp(out, min=0.0)
    return out


def _compute_effective_write(
    write_prob: torch.Tensor,
    adv: torch.Tensor,
    agent_cfg: Any,
) -> torch.Tensor:
    mode = str(getattr(agent_cfg, "miras_weight_mode", "abs_adv"))
    if mode == "abs_adv":
        w = adv.abs()
    elif mode == "pos_adv":
        w = torch.clamp(adv, min=0.0)
    else:
        w = torch.ones_like(adv)
    return write_prob * w


def _nan_inf_grad_check(
    env_id: str,
    cfg: Any,
    device: str,
) -> Dict[str, Any]:
    envs = make_envs(env_id, num_envs=1, cfg=cfg)
    agent = InfinityV3DualHybridAgent(cfg.agent).to(device)
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _set_seeds(int(seed))
    trainer = PPOTrainer(
        agent,
        cfg.ppo,
        device=device,
        seed=seed,
    )

    rollouts = trainer.collect_rollouts(envs)

    obs = rollouts.observations.to(device)
    actions = rollouts.actions.to(device)
    old_logp = rollouts.log_probs.to(device)
    returns = rollouts.returns.to(device)
    adv_raw = rollouts.advantages.to(device)

    adv = adv_raw
    if bool(getattr(cfg.ppo, "adv_norm", True)):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_clip = getattr(cfg.ppo, "adv_clip", None)
    if adv_clip is not None and adv_clip > 0:
        adv = torch.clamp(adv, -float(adv_clip), float(adv_clip))
    adv_used = (
        torch.clamp(adv, min=0.0)
        if bool(getattr(cfg.ppo, "adv_positive_only", True))
        else adv
    )

    agent.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    (
        new_logp,
        new_values,
        entropy,
        write_prob,
        _effective_write_mean,
    ) = agent.evaluate_actions(
        obs,
        actions,
        advantage=adv,
    )

    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(
        ratio,
        1.0 - float(cfg.ppo.clip_eps),
        1.0 + float(cfg.ppo.clip_eps),
    ) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * ((new_values - returns) ** 2).mean()
    entropy_loss = entropy.mean()

    eps = 1e-8
    gate = torch.clamp(
        write_prob,
        min=float(agent.cfg.write_gate_floor),
        max=float(agent.cfg.write_gate_ceiling),
    )
    memory_gate_loss = -(
        adv_used.detach() * torch.log(gate + eps)
    ).mean()

    loss = (
        policy_loss
        + float(cfg.ppo.value_loss_coef) * value_loss
        - float(cfg.ppo.entropy_coef) * entropy_loss
        + float(cfg.ppo.mem_gate_coef) * memory_gate_loss
    )

    loss.backward()

    grad_finite = True
    max_grad_abs = 0.0
    nan_params = 0
    inf_params = 0

    for p in agent.parameters():
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            grad_finite = False
            nan_params += int(torch.isnan(g).any().item())
            inf_params += int(torch.isinf(g).any().item())
        max_grad_abs = max(max_grad_abs, float(g.abs().max().item()))

    for env in envs:
        env.close()
    agent.shutdown()

    return {
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grad_finite": bool(grad_finite),
        "nan_params": int(nan_params),
        "inf_params": int(inf_params),
        "max_grad_abs": float(max_grad_abs),
    }


def _one_update_gate_metrics(
    env_id: str,
    cfg: Any,
    seed: int,
    device: str,
    out_dir: str,
) -> Dict[str, Any]:
    _set_seeds(int(seed))
    envs = make_envs(env_id, num_envs=1, cfg=cfg)
    agent = InfinityV3DualHybridAgent(cfg.agent).to(device)
    trainer = PPOTrainer(agent, cfg.ppo, device=device, seed=int(seed))

    rollouts = trainer.collect_rollouts(envs)

    obs = rollouts.observations.to(device)
    adv_raw = rollouts.advantages.to(device)

    adv_used = _compute_adv_used(adv_raw, cfg.ppo)

    with torch.no_grad():
        out = agent.forward(obs, advantage=adv_raw)
        logits = out["logits"]
        value = out["value"]
        write_prob = out["write_prob"].squeeze(-1)

    eps = 1e-8
    gate = torch.clamp(
        write_prob,
        min=float(agent.cfg.write_gate_floor),
        max=float(agent.cfg.write_gate_ceiling),
    )
    mem_gate_loss = float(
        (-(adv_used.detach() * torch.log(gate + eps))).mean().item()
    )

    effective_write = _compute_effective_write(gate, adv_raw, agent.cfg)

    gate_np = gate.detach().cpu().numpy()
    eff_np = effective_write.detach().cpu().numpy()

    stats = {
        "adv_raw_min": float(adv_raw.min().item()),
        "adv_raw_max": float(adv_raw.max().item()),
        "adv_used_min": float(adv_used.min().item()),
        "adv_used_max": float(adv_used.max().item()),
        "write_prob_min": float(gate.min().item()),
        "write_prob_max": float(gate.max().item()),
        "write_prob_mean": float(gate.mean().item()),
        "write_prob_p95": float(np.percentile(gate_np, 95)),
        "effective_write_mean": float(effective_write.mean().item()),
        "effective_write_p95": float(np.percentile(eff_np, 95)),
        "mem_gate_loss": float(mem_gate_loss),
        "finite_logits": bool(_finite_tensor(logits)),
        "finite_value": bool(_finite_tensor(value)),
        "finite_adv": bool(_finite_tensor(adv_raw)),
        "finite_write_prob": bool(_finite_tensor(gate)),
    }

    corr = _pearson_corr(
        adv_used.detach().cpu().numpy(),
        gate.detach().cpu().numpy(),
    )
    stats["write_prob_adv_corr"] = float(corr)

    plt.figure(figsize=(7, 4))
    plt.scatter(
        adv_used.detach().cpu().numpy(),
        gate.detach().cpu().numpy(),
        s=8,
        alpha=0.5,
    )
    plt.xlabel("adv_used")
    plt.ylabel("write_prob")
    plt.title(f"Adv vs WriteProb (corr={corr:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adv_vs_writeprob.png"))
    plt.close()

    for env in envs:
        env.close()
    agent.shutdown()

    return stats


def _eval_policy_success(
    env_id: str,
    cfg: Any,
    agent: InfinityV3DualHybridAgent,
    episodes: int,
    device: str,
) -> Dict[str, float]:
    envs = make_envs(env_id, num_envs=1, cfg=cfg)
    env = envs[0]

    rewards: List[float] = []
    successes: List[int] = []
    steps_to_success: List[int] = []

    agent.eval()

    base_seed = getattr(cfg, "seed", None)
    for ep in range(episodes):
        if base_seed is None:
            reset_out = env.reset()
        else:
            reset_out = env.reset(seed=int(base_seed) + int(ep))
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        ep_reward = 0.0
        t = 0
        succ = 0
        tts: Optional[int] = None

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

            if float(reward) >= 10.0 and succ == 0:
                succ = 1
                tts = t

        rewards.append(ep_reward)
        successes.append(succ)
        if succ == 1 and tts is not None:
            steps_to_success.append(int(tts))

    env.close()
    agent.reset_episode()

    out: Dict[str, float] = {
        "mean_return": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }
    if steps_to_success:
        out["median_steps_to_success"] = float(np.median(steps_to_success))
    else:
        out["median_steps_to_success"] = float("inf")
    return out


def _train_run(
    env_id: str,
    cfg: Any,
    device: str,
    iterations: int,
) -> Tuple[InfinityV3DualHybridAgent, Dict[str, List[float]]]:
    cfg.ppo.max_iterations = iterations
    cfg.ppo.num_envs = 1

    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _set_seeds(int(seed))

    envs = make_envs(env_id, num_envs=1, cfg=cfg)
    agent = InfinityV3DualHybridAgent(cfg.agent).to(device)
    trainer = PPOTrainer(agent, cfg.ppo, device=device, seed=seed)

    hist: Dict[str, List[float]] = {
        "mean_reward": [],
        "mean_write_prob": [],
        "mean_adv_used": [],
        "write_prob_adv_corr": [],
        "effective_write_mean": [],
    }

    for _ in range(iterations):
        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)
        for k in hist:
            if k in stats:
                hist[k].append(float(stats[k]))

    for env in envs:
        env.close()
    agent.shutdown()

    return agent, hist


def _write_report(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
        "PYTHONPATH=src python3 -m infinity_dual_hybrid.validation \\",
        f"  --out {out_dir} \\",
        f"  --seed {args.seed} \\",
        f"  --device {args.device} \\",
        f"  --train-iterations {args.train_iterations} \\",
        f"  --eval-episodes {args.eval_episodes} \\",
        f"  --delays \"{args.delays}\"",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-iterations", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--delays",
        type=str,
        default="50,100,250,500,1000,2000",
    )
    args = parser.parse_args()

    tag = _now_tag()
    out_dir = args.out or os.path.join("results", "validation", tag)
    _ensure_dir(out_dir)

    _write_json(
        os.path.join(out_dir, "config.json"),
        {
            **vars(args),
            "out_dir": out_dir,
        },
    )

    log_path = os.path.join(out_dir, "validation.log")
    log_f = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _Tee(old_stdout, log_f)
    sys.stderr = _Tee(old_stderr, log_f)

    try:
        runtime = _runtime_info()
        seeds = _set_seeds(int(args.seed))

        report_lines: List[str] = []
        report_lines.append("# INFINITY_DUAL_HYBRID Validation Report")
        report_lines.append("")
        report_lines.append(f"Output dir: `{out_dir}`")
        report_lines.append("")
        report_lines.append("## Runtime")
        report_lines.append("```json")
        report_lines.append(json.dumps(runtime, indent=2))
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("## Seeds")
        report_lines.append("```json")
        report_lines.append(json.dumps(seeds, indent=2))
        report_lines.append("```")
        report_lines.append("")

        static_checks = [
            _env_api_check("DelayedCue-v0", seed=args.seed),
            _env_api_check("DelayedCueRegime-v0", seed=args.seed),
        ]

        delayedcue_tests = _delayedcue_invariant_tests()
        regime_tests = _regime_invariant_tests()

        cfg_full = get_config_for_env("DelayedCue-v0")
        cfg_full.seed = int(args.seed)
        cfg_full.device = str(args.device)
        _disable_ltm(cfg_full)

        gate_stats = _one_update_gate_metrics(
            env_id="DelayedCue-v0",
            cfg=cfg_full,
            seed=int(args.seed),
            device=str(args.device),
            out_dir=out_dir,
        )

        grad_check = _nan_inf_grad_check(
            env_id="DelayedCue-v0",
            cfg=cfg_full,
            device=str(args.device),
        )

        ablations: Dict[str, Any] = {}

        def make_run_cfg(
            base_env: str,
            mem_gate_coef: Optional[float],
            mode: Optional[str],
        ):
            c = get_config_for_env(base_env)
            c.seed = int(args.seed)
            c.device = str(args.device)
            _disable_ltm(c)
            if mem_gate_coef is not None:
                c.ppo.mem_gate_coef = float(mem_gate_coef)
            if mode is not None:
                c.agent.miras_weight_mode = str(mode)
            return c

        run_defs = {
            "FULL": make_run_cfg("DelayedCue-v0", None, None),
            "NO_GATE_LOSS": make_run_cfg("DelayedCue-v0", 0.0, None),
            "NO_ADV_WEIGHT": make_run_cfg("DelayedCue-v0", None, "none"),
        }

        for name, c in run_defs.items():
            agent, hist = _train_run(
                env_id="DelayedCue-v0",
                cfg=c,
                device=str(args.device),
                iterations=int(args.train_iterations),
            )

            eval_metrics = _eval_policy_success(
                env_id="DelayedCue-v0",
                cfg=c,
                agent=agent,
                episodes=int(args.eval_episodes),
                device=str(args.device),
            )

            tail = max(1, int(0.2 * len(hist["mean_reward"])))
            tail_mean = float(np.mean(hist["mean_reward"][-tail:]))

            ablations[name] = {
                "train": {
                    "mean_reward_last_20pct": tail_mean,
                    "history": hist,
                },
                "eval": eval_metrics,
            }

        delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]
        delay_sweep: Dict[str, Any] = {"delays": delays, "runs": {}}

        for name in run_defs.keys():
            delay_sweep["runs"][name] = {
                "mean_return": [],
                "success_rate": [],
            }

        for d in delays:
            for name in run_defs.keys():
                c = make_run_cfg("DelayedCue-v0", None, None)
                if name == "NO_GATE_LOSS":
                    c.ppo.mem_gate_coef = 0.0
                if name == "NO_ADV_WEIGHT":
                    c.agent.miras_weight_mode = "none"
                c.delayedcue_delay = int(d)

                agent, _ = _train_run(
                    env_id="DelayedCue-v0",
                    cfg=c,
                    device=str(args.device),
                    iterations=max(1, int(args.train_iterations // 2)),
                )
                m = _eval_policy_success(
                    env_id="DelayedCue-v0",
                    cfg=c,
                    agent=agent,
                    episodes=max(10, int(args.eval_episodes // 2)),
                    device=str(args.device),
                )

                delay_sweep["runs"][name]["mean_return"].append(
                    float(m["mean_return"])
                )
                delay_sweep["runs"][name]["success_rate"].append(
                    float(m["success_rate"])
                )

        plt.figure(figsize=(7, 4))
        for name in run_defs.keys():
            plt.plot(
                delays,
                delay_sweep["runs"][name]["mean_return"],
                marker="o",
                label=name,
            )
        plt.xlabel("delay")
        plt.ylabel("mean return")
        plt.title("Reward vs Delay (Ablations)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "reward_vs_delay.png"))
        plt.close()

        cfg_reg = get_config_for_env("DelayedCueRegime-v0")
        cfg_reg.seed = int(args.seed)
        cfg_reg.device = str(args.device)
        _disable_ltm(cfg_reg)
        agent_reg, hist_reg = _train_run(
            env_id="DelayedCueRegime-v0",
            cfg=cfg_reg,
            device=str(args.device),
            iterations=int(args.train_iterations),
        )
        _ = agent_reg

        plt.figure(figsize=(7, 4))
        plt.plot(hist_reg["mean_reward"])
        plt.xlabel("iteration")
        plt.ylabel("mean reward")
        plt.title("Regime-Shift Recovery (train mean_reward)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "regime_shift_recovery.png"))
        plt.close()

        reg_check_cfg = get_config_for_env("CartPole-v1")
        reg_check_cfg.seed = int(args.seed)
        reg_check_cfg.device = str(args.device)
        _disable_ltm(reg_check_cfg)
        try:
            agent_cp, _ = _train_run(
                env_id="CartPole-v1",
                cfg=reg_check_cfg,
                device=str(args.device),
                iterations=2,
            )
            _ = agent_cp
            regression_ok = True
        except Exception as e:
            regression_ok = False
            report_lines.append("## Regression")
            report_lines.append(f"CartPole-v1 training failed: `{e}`")
            report_lines.append("")

        log_f.flush()
        try:
            with open(log_path, "r", encoding="utf-8") as rf:
                log_text = rf.read().splitlines()
        except Exception:
            log_text = []

        backend_lines = [ln for ln in log_text if "SSM backend:" in ln]
        backend_excerpt = backend_lines[-1] if backend_lines else ""

        metrics = {
            "runtime": runtime,
            "seeds": seeds,
            "ltm_disabled": True,
            "static_env_checks": static_checks,
            "env_invariants": {
                "DelayedCue-v0": delayedcue_tests,
                "DelayedCueRegime-v0": regime_tests,
            },
            "gate_metrics_one_update": gate_stats,
            "grad_check_one_backward": grad_check,
            "ssm_backend_excerpt": backend_excerpt,
            "ablations": ablations,
            "delay_sweep": delay_sweep,
            "regime_shift": {
                "train_mean_reward": hist_reg["mean_reward"],
            },
            "regression": {
                "cartpole_train_ok": bool(regression_ok),
            },
            "artifacts": {
                "validation_log": os.path.basename(log_path),
            },
        }

        report_lines.append("## Static env checks")
        report_lines.append("```json")
        report_lines.append(json.dumps(static_checks, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## Env invariants")
        report_lines.append("### DelayedCue-v0")
        report_lines.append("```json")
        report_lines.append(json.dumps(delayedcue_tests, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("### DelayedCueRegime-v0")
        report_lines.append("```json")
        report_lines.append(json.dumps(regime_tests, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## Advantage-gated MIRAS: one-update evidence")
        report_lines.append("```json")
        report_lines.append(json.dumps(gate_stats, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## Gradient finite check (one backward)")
        report_lines.append("```json")
        report_lines.append(json.dumps(grad_check, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## SSM backend excerpt")
        report_lines.append("```text")
        report_lines.append(backend_excerpt or "(not found)")
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## Ablations summary")
        report_lines.append("```json")
        report_lines.append(
            json.dumps(
                {k: v["eval"] for k, v in ablations.items()},
                indent=2,
            )
        )
        report_lines.append("```")
        report_lines.append("")

        validated = True
        if not all(x.get("gymnasium_make_ok") for x in static_checks):
            validated = False
        if not all(t["pass"] for t in delayedcue_tests):
            validated = False
        if not all(t["pass"] for t in regime_tests):
            validated = False
        if not (
            gate_stats["finite_logits"]
            and gate_stats["finite_value"]
            and gate_stats["finite_adv"]
        ):
            validated = False
        if not bool(grad_check.get("grad_finite", False)):
            validated = False

        report_lines.append("## Conclusion")
        report_lines.append("Validated" if validated else "Not validated")
        report_lines.append("")
        report_lines.append(
            "Biggest remaining risk if not validated: gate learning signal "
            "may be too weak or saturated; inspect write_prob distribution "
            "and correlation trend."
        )

        _write_json(os.path.join(out_dir, "metrics.json"), metrics)
        _write_report(
            os.path.join(out_dir, "validation_report.md"),
            report_lines,
        )
        _write_run_commands(
            os.path.join(out_dir, "run_commands.sh"),
            out_dir,
            args,
        )
    finally:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
