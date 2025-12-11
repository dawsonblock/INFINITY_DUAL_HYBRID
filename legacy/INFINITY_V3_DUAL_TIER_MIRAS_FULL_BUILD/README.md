# INFINITY_V3_DUAL_TIER_MIRAS_FULL_BUILD

Infinity-style RL core with Dual-Tier Miras parametric memory.

This build is self-contained and runnable. It does **not** re-create
your entire historical Infinity boss file with FAISS IVF-PQ + Ray HPO,
but it gives you a clean, production-style baseline with:

- Dual-Tier Miras (fast + Titans deep) in `dual_tier_miras.py`
- Infinity actor-critic core in `continuous_memory_mamba_agent_v3_infinity.py`
- PPO trainer in `infinity_ppo_train.py`

Use this as the canonical upgraded core, then merge it into your larger
Infinity ecosystem as needed.

## Files

- `dual_tier_miras.py`
  - `SSMCompressedMiras`   (fast, low-rank parametric memory)
  - `SSMCompressedMirasTitans` (deep, Titans-style with momentum + Huber)
  - `DualTierMiras`        (context-gated combiner + per-tier lr scaling)

- `continuous_memory_mamba_agent_v3_infinity.py`
  - `InfinityBackbone`      (MLP; swap this to Mamba2 if desired)
  - `InfinityActorCritic`   (actor-critic with Dual-Tier Miras)
  - `InfinityConfig`, `build_infinity_actor_critic`

- `infinity_ppo_train.py`
  - `PPOConfig`
  - `ppo_train(cfg)` wired to `InfinityActorCritic`
  - Default env: `CartPole-v1`

- `requirements.txt`
  - Minimal Python deps

## Quickstart

```bash
pip install -r requirements.txt
python infinity_ppo_train.py
```

You should see the mean evaluation return climb as the agent learns
CartPole. Dual-Tier Miras is active and uses PPO advantages as write
weights during training.

## Plugging into your larger Infinity build

1. Drop these three modules next to your existing Infinity codebase.
2. Import `InfinityActorCritic` wherever you currently build your
   policy/value network.
3. Replace your older SimpleLTM wiring with Dual-Tier Miras by following
   the pattern in `InfinityActorCritic`:
   - `k_t = W_k h_t`
   - `v_target = stop_grad(W_v h_t)`
   - `context = h_t`
   - `read = DualTierMiras.read(k_t, context)`
   - `DualTierMiras.update(k_t, v_target, weight=advantage, context=context)` during training.

If you want to swap the MLP backbone for Mamba2 (from the official
`mamba_ssm` package), replace `InfinityBackbone` with a Mamba-based
module that maps `obs` to `hidden_dim`, and keep the Miras wiring
unchanged.
