# INFINITY_DUAL_TIER_MIRAS_PPO (Regenerated)

Dual-Tier Miras parametric memory + PPO demo build.

## Files

- `dual_tier_miras.py`
  - Fast Miras, Titans deep Miras, and DualTier combiner.
- `ppo_dual_tier_miras_cartpole.py`
  - PPO + Actor-Critic wired to Dual-Tier Miras on CartPole-v1.

## Usage

1. Install dependencies:

```bash
pip install torch gym numpy
```

2. Run training:

```bash
python ppo_dual_tier_miras_cartpole.py
```

You should see mean evaluation return increase as the agent learns on CartPole.

To integrate with your Infinity Mamba core, import the memory classes in your
`continuous_memory_mamba_agent_v3_infinity.py` and use the same pattern:

- k_t = W_k h_t
- v_target = stop_grad(W_v h_t)
- context = h_t
- read = DualTierMiras.read(k_t, context)
- advantage-weighted DualTierMiras.update(...) during training.
