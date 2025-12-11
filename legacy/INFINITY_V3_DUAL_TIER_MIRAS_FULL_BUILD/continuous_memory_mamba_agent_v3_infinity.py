"""
continuous_memory_mamba_agent_v3_infinity.py

Infinity-style actor-critic core with Dual-Tier Miras parametric memory.

This is a self-contained RL-ready controller:
  - Backbone: simple MLP (you can swap to Mamba2 from mamba_ssm).
  - Memory: Dual-tier Miras (fast + Titans deep).
  - Heads: policy + value.
  - Interface: forward(obs, advantage=None) → (logits, value).

To plug into richer trainers / envs, import this module and reuse
InfinityActorCritic as the policy/value network.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_tier_miras import SSMCompressedMiras, SSMCompressedMirasTitans, DualTierMiras


class InfinityBackbone(nn.Module):
    """
    Simple feedforward backbone.

    NOTE: To use Mamba2 from the official mamba_ssm package, replace this with
    a Mamba-based backbone that maps obs → hidden_dim.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InfinityActorCritic(nn.Module):
    """
    Actor-critic with Dual-Tier Miras memory.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.backbone = InfinityBackbone(obs_dim, hidden_dim)

        # Projections for Miras
        self.miras_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.miras_val_proj = nn.Linear(hidden_dim, hidden_dim)

        # Fast + deep memory
        fast = SSMCompressedMiras(d_model=hidden_dim, rank=32, lr=1e-3, l2_reg=1e-4)
        deep = SSMCompressedMirasTitans(
            d_model=hidden_dim,
            rank=32,
            lr=5e-4,
            l2_reg=1e-4,
            momentum=0.9,
            use_huber=True,
            huber_delta=1.0,
        )
        self.miras = DualTierMiras(
            fast_mem=fast,
            deep_mem=deep,
            d_model=hidden_dim,
            init_fast_weight=0.7,
            context_gate=True,
        )

        # Policy / value heads
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        advantage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: [B, obs_dim]
        advantage: [B] or [B,1], optional, used as write weight.
        """
        x = self.backbone(obs)  # [B, hidden_dim]

        k = self.miras_key_proj(x)
        v_target = self.miras_val_proj(x).detach()
        context = x

        read_out = self.miras.read(k, context=context)
        v_hat = read_out["v"]
        x_aug = x + v_hat

        if self.training:
            w = advantage.detach() if advantage is not None else None
            self.miras.update(k, v_target, weight=w, context=context)

        logits = self.policy_head(x_aug)
        value = self.value_head(x_aug).squeeze(-1)

        return logits, value


@dataclass
class InfinityConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int = 256

    @staticmethod
    def from_spaces(obs_space, act_space, hidden_dim: int = 256) -> "InfinityConfig":
        obs_dim = obs_space.shape[0]
        # Discrete-only for now
        act_dim = act_space.n
        return InfinityConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)


def build_infinity_actor_critic(obs_dim: int, act_dim: int, hidden_dim: int = 256) -> InfinityActorCritic:
    return InfinityActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
