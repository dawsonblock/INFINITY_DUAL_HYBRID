# Infinity Dual Hybrid - Lean Build

A production-grade reinforcement learning framework combining state-space models, parametric memory, episodic long-term memory, and PPO.

## Highlights

- **Hybrid backbone**: Mamba2 (SSM) + optional attention
- **Dual-Tier Miras**: fast (SGD) + deep (momentum/retention) parametric memory
- **Optional episodic LTM**: FAISS IVF-PQ (when enabled)
- **PPO trainer**: GAE, clipped surrogate, entropy bonus, adaptive KL
- **UnifiedLogger**: console + CSV/JSONL + TensorBoard (optional)

## Architecture

```text
Observation → Encoder → Hybrid Backbone → Memory Fusion → Policy/Value Heads
                              ↓                ↑
                         Mamba2 + Attention    │
                                               ├── Dual-Tier Miras (parametric)
                                               └── FAISS LTM (episodic)
```

### Core Components

- **Hybrid SSM/Attention Backbone**: Combines Mamba2 (O(N) linear-time) with Transformer attention
- **Dual-Tier Miras Memory**: Fast tier (SGD) + Deep tier (momentum, Huber, retention gate)
- **FAISS IVF-PQ LTM**: Scalable episodic memory with approximate nearest neighbor search
- **PPO Trainer**: GAE, clipped surrogate, entropy bonus

## Project Structure

```text
INFINITY_DUAL_HYBRID_LEAN/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── infinity_dual_hybrid/
│       ├── __init__.py
│       ├── config.py          # All hyperparameter dataclasses
│       ├── ssm_backbone.py    # Hybrid Mamba2 + Attention
│       ├── miras.py           # Dual-Tier Miras memory
│       ├── ltm.py             # FAISS IVF-PQ episodic memory
│       ├── agent.py           # InfinityV3DualHybridAgent
│       ├── ppo_trainer.py     # PPO with GAE
│       ├── envs.py            # Gym environment utilities
│       └── train.py           # CLI entry point
├── vendor/
│   └── mamba_ssm/             # Vendored Mamba2 library
├── scripts/
│   ├── train_cartpole_baseline.py
│   ├── train_cartpole_miras.py
│   └── auto_tuner_dual_hybrid.py
├── tests/
│   ├── test_dual_tier_miras_cartpole.py
│   └── test_memory_sanity.py
└── legacy/                    # Archived monolithic versions
```

## Installation

```bash
cd INFINITY_DUAL_HYBRID_LEAN

# Recommended (editable install)
pip install -e .

# Dev tools (tests/formatting)
pip install -e ".[dev]"
```

## Quick Start

### CLI Training

```bash
# Run from repo root
cd INFINITY_DUAL_HYBRID_LEAN

# Quick sanity test
python -m infinity_dual_hybrid.train --test

# Full training via console script
infinity-train --env CartPole-v1 --iterations 50

# Train baseline (no memory)
python scripts/train_cartpole_baseline.py

# Train with Miras memory
python scripts/train_cartpole_miras.py

# Run auto-tuner
python scripts/auto_tuner_dual_hybrid.py
```

### Python API

```python
from infinity_dual_hybrid import (
    get_config_for_env,
    build_agent,
    PPOTrainer,
    make_envs,
)

# Create config
cfg = get_config_for_env("CartPole-v1")

# Enable/disable memory systems
cfg.agent.use_miras_in_forward = True
cfg.agent.use_ltm_in_forward = False
cfg.agent.sync_dims()

# Build components
envs = make_envs(cfg.env_id, num_envs=1)
agent = build_agent(cfg.agent).to(cfg.device)
trainer = PPOTrainer(agent, cfg.ppo)

# Training loop
for i in range(100):
    rollouts = trainer.collect_rollouts(envs)
    stats = trainer.train_step(rollouts)
    print(f"Iter {i}: reward={stats['mean_reward']:.2f}")

for env in envs:
    env.close()
agent.shutdown()
```

## Configuration

All hyperparameters are centralized in dataclasses:

```python
from infinity_dual_hybrid import get_config_for_env

cfg = get_config_for_env("CartPole-v1")

# Modify as needed
cfg.agent.hidden_dim = 128
cfg.agent.use_miras_in_forward = True
cfg.agent.use_ltm_in_forward = False
cfg.ppo.learning_rate = 3e-4
cfg.agent.sync_dims()  # Call after changing hidden_dim
```

### Key Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent.hidden_dim` | 256 | Model hidden dimension |
| `agent.use_miras_in_forward` | True | Enable Dual-Tier Miras |
| `agent.use_ltm_in_forward` | True | Enable episodic LTM |
| `backbone.use_mamba` | True | Use Mamba2 backbone |
| `backbone.use_attention` | True | Use attention layers |
| `ppo.clip_eps` | 0.2 | PPO clip epsilon |
| `ppo.gae_lambda` | 0.95 | GAE lambda |

## Testing

```bash
# Run tests
cd INFINITY_DUAL_HYBRID_LEAN
python -m pytest tests/ -v
```

## Memory Systems

### Dual-Tier Miras (Parametric)

Fast-adapting working memory with two tiers:

- **Fast Tier**: Direct SGD updates for rapid adaptation
- **Deep Tier**: Momentum-based updates with Huber loss and retention gating

### FAISS LTM (Episodic)

Scalable long-term memory using IVF-PQ indexing for efficient retrieval.

## License

MIT
