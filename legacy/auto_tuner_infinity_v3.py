import os
from typing import Any, Dict

import torch

from infinity_v3_stack import (
    SimpleAsciiTokenizer,
    LogicPuzzleEnv,
    LongHorizonMemoryEnv,
    PuzzleOracle,
    LTMConfig,
    ContinuousMemoryInfinityAgent,
    InfinityTrainer,
    GRPOConfig,
    GRPOHarness,
    build_lm_batch_from_env,
)


def train_infinity_trial(config: Dict[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = SimpleAsciiTokenizer()
    env_type = config.get("env_type", "long")
    if env_type == "short":
        env = LogicPuzzleEnv(device=device)
    else:
        env = LongHorizonMemoryEnv(device=device)
    oracle = PuzzleOracle(tokenizer)

    d_model = config["d_model"]
    ltm_cfg = LTMConfig(
        d_key=d_model,
        d_value=d_model,
        max_size=config["ltm_max_size"],
        use_faiss=config.get("use_faiss", True),
        nlist=config.get("nlist", 4096),
        m=config.get("m", 16),
        nprobe=config.get("nprobe", 8),
        device=device,
    )

    agent = ContinuousMemoryInfinityAgent(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config.get("max_seq_len", 1024),
        ltm_cfg=ltm_cfg,
        memory_top_k=config["memory_top_k"],
        use_mamba=config.get("use_mamba", True),
        dropout=config.get("dropout", 0.1),
    ).to(device)

    trainer = InfinityTrainer(
        agent,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
    )

    grpo_cfg = GRPOConfig(
        group_size=config["group_size"],
        max_gen_len=config["max_gen_len"],
        temperature=config["temperature"],
        top_k=config["top_k"],
        kl_beta=config["kl_beta"],
    )
    harness = GRPOHarness(agent, oracle, grpo_cfg)

    max_epochs = config["max_epochs"]
    steps_per_epoch = config["steps_per_epoch"]
    batch_size = config["batch_size"]

    for epoch in range(max_epochs):
        for _ in range(steps_per_epoch):
            rl_prompts, rl_answers = env.sample_batch(batch_size, tokenizer)
            rl_prompts = [p.to(device) for p in rl_prompts]

            lm_prompts, lm_answers = env.sample_batch(batch_size, tokenizer)
            lm_prompts = [p.to(device) for p in lm_prompts]
            lm_input_ids, lm_labels = build_lm_batch_from_env(tokenizer, lm_prompts, lm_answers, device)

            stats = trainer.joint_step_grpo_lm(
                harness,
                rl_prompts,
                rl_answers,
                lm_input_ids,
                lm_labels,
                alpha=config["alpha"],
            )

        print(
            f"[epoch {epoch+1}/{max_epochs}] "
            f"total={stats['total_loss']:.4f} "
            f"rl={stats['rl_loss']:.4f} lm={stats['lm_loss']:.4f} "
            f"avg_reward={stats['avg_reward']:.3f}"
        )


if __name__ == "__main__":
    # Minimal example config for manual runs (no Ray needed)
    config = {
        "d_model": 256,
        "n_heads": 4,
        "num_layers": 4,
        "ltm_max_size": 200_000,
        "memory_top_k": 16,
        "use_faiss": True,
        "nlist": 4096,
        "m": 16,
        "nprobe": 8,
        "use_mamba": True,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "group_size": 4,
        "max_gen_len": 32,
        "temperature": 0.8,
        "top_k": 32,
        "kl_beta": 0.01,
        "max_epochs": 5,
        "steps_per_epoch": 10,
        "batch_size": 16,
        "alpha": 0.5,
        "env_type": "long",
        "max_seq_len": 1024,
    }
    train_infinity_trial(config)
