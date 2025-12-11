"""
miras.py

Dual-Tier Miras Parametric Memory System.

Architecture:
- Fast Tier (SSMCompressedMiras): SGD-based low-rank memory for rapid adaptation
- Deep Tier (SSMCompressedMirasTitans): Momentum-based with Huber loss and retention gate
- DualTierMiras: Combines both tiers with context-gated mixing

The two tiers serve different purposes:
- Fast tier: Quick adaptation to recent patterns (high plasticity)
- Deep tier: Stable storage of important patterns (high stability)

Memory operation:
- W = scale * tanh(B @ C.T) + diag(D)  [low-rank + diagonal]
- Read: v = k @ W.T
- Update: gradient descent on reconstruction error ||v - k @ W.T||
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn

from .config import MirasConfig


class SSMCompressedMirasTitans(nn.Module):
    """
    Titans-style deep Miras parametric memory (Deep Tier).

    Features:
    - Momentum-based updates for stable learning
    - Huber loss for robust gradient estimation
    - Adaptive retention gate that controls forgetting
    - Low-rank parameterization: W = scale * tanh(B @ C.T) + diag(D)

    This tier is designed for stable, long-term pattern storage.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        lr: float = 5e-4,
        l2_reg: float = 1e-4,
        momentum: float = 0.9,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr = lr
        self.l2_reg = l2_reg
        self.momentum = momentum
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        # Low-rank factors B, C and diagonal D
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        # Scale for tanh output
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

        # Momentum buffers
        self.register_buffer("S_B", torch.zeros_like(self.B))
        self.register_buffer("S_C", torch.zeros_like(self.C))

        # Retention gate: learns when to forget vs retain
        self.retention_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def W(self) -> torch.Tensor:
        """Compute full weight matrix from low-rank factors."""
        low_rank = self.B @ self.C.t()
        W = self.scale * torch.tanh(low_rank)
        W = W + torch.diag(self.D)
        return W

    def read(self, k: torch.Tensor) -> torch.Tensor:
        """
        Read from memory.

        Args:
            k: Query keys [B, d_model]
        Returns:
            Retrieved values [B, d_model]
        """
        W = self.W()
        return k @ W.t()

    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update memory with new key-value pairs.

        Args:
            k: Keys [B, d_model]
            v: Target values [B, d_model]
            weight: Optional importance weights [B] or [B, 1]
        Returns:
            Dictionary of statistics
        """
        if k.numel() == 0:
            return {}

        device = self.B.device
        k = k.to(device)
        v = v.to(device)
        batch_size = k.shape[0]

        # Compute reconstruction error
        W = self.W()
        v_hat = k @ W.t()
        err = v - v_hat

        # Apply importance weighting
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            err = err * weight.to(device)

        # Huber loss for robust gradients
        if self.use_huber:
            delta = self.huber_delta
            abs_err = err.abs()
            mask = (abs_err <= delta).float()
            err = mask * err + (1.0 - mask) * delta * err.sign()

        # Compute gradients
        gradW = -(err.t() @ k) / (batch_size + 1e-8) + self.l2_reg * W
        gradB = gradW @ self.C
        gradC = gradW.t() @ self.B

        # Momentum update
        self.S_B = self.momentum * self.S_B - self.lr * gradB
        self.S_C = self.momentum * self.S_C - self.lr * gradC

        # Retention gate: determines how much to retain vs update
        k_mean = k.mean(dim=0, keepdim=True)
        alpha_t = torch.sigmoid(self.retention_gate(k_mean)).squeeze()

        # Apply update with retention
        self.B.data = (1.0 - alpha_t) * self.B.data + self.S_B
        self.C.data = (1.0 - alpha_t) * self.C.data + self.S_C

        stats = {
            "B_norm": float(self.B.data.norm().item()),
            "C_norm": float(self.C.data.norm().item()),
            "S_B_norm": float(self.S_B.norm().item()),
            "S_C_norm": float(self.S_C.norm().item()),
            "retention_gate": float(alpha_t.mean().item()) if alpha_t.dim() > 0 else float(alpha_t.item()),
            "err_l2": float(err.norm(dim=-1).mean().item()),
        }
        return stats

    def reset_state(self) -> None:
        """Reset momentum buffers (call at episode boundaries if desired)."""
        self.S_B.zero_()
        self.S_C.zero_()

    def reset_parameters(self) -> None:
        """Full parameter reset."""
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)
        self.S_B.zero_()
        self.S_C.zero_()


class SSMCompressedMiras(nn.Module):
    """
    Simple low-rank Miras parametric memory (Fast Tier).

    Features:
    - Direct SGD updates (no momentum) for rapid adaptation
    - Low-rank parameterization: W = scale * tanh(B @ C.T) + diag(D)

    This tier is designed for quick adaptation to recent patterns.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr = lr
        self.l2_reg = l2_reg

        # Low-rank factors
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

    def W(self) -> torch.Tensor:
        """Compute full weight matrix from low-rank factors."""
        low_rank = self.B @ self.C.t()
        W = self.scale * torch.tanh(low_rank)
        W = W + torch.diag(self.D)
        return W

    def read(self, k: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        W = self.W()
        return k @ W.t()

    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Update memory with SGD."""
        if k.numel() == 0:
            return {}

        W = self.W()
        k = k.to(W.device)
        v = v.to(W.device)
        batch_size = k.shape[0]

        # Reconstruction error
        v_hat = k @ W.t()
        err = v - v_hat

        # Apply importance weighting
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            err = err * weight.to(W.device)

        # Compute and apply gradients directly (SGD)
        gradW = -(err.t() @ k) / (batch_size + 1e-8) + self.l2_reg * W
        gradB = gradW @ self.C
        gradC = gradW.t() @ self.B

        self.B.data.add_(-self.lr * gradB)
        self.C.data.add_(-self.lr * gradC)

        return {
            "B_norm": float(self.B.data.norm().item()),
            "C_norm": float(self.C.data.norm().item()),
            "err_l2": float(err.norm(dim=-1).mean().item()),
        }

    def reset_state(self) -> None:
        """No state to reset for fast tier."""
        pass

    def reset_parameters(self) -> None:
        """Full parameter reset."""
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)


class DualTierMiras(nn.Module):
    """
    Dual-Tier Miras Memory Combiner.

    Combines fast and deep tiers with context-gated mixing:
    - Base mixing ratio is learned (init_fast_weight)
    - Context gate adjusts mixing based on current input
    - Output: v = w_fast * v_fast + w_deep * v_deep

    The context gate allows the model to dynamically decide whether
    to rely more on recent patterns (fast) or stable patterns (deep).
    """

    def __init__(
        self,
        d_model: int,
        fast_rank: int = 32,
        deep_rank: int = 32,
        fast_lr: float = 1e-3,
        deep_lr: float = 5e-4,
        init_fast_weight: float = 0.7,
        context_gate: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_gate_enabled = context_gate
        self.init_fast_weight = init_fast_weight

        # Create fast and deep tiers
        self.fast_mem = SSMCompressedMiras(
            d_model=d_model,
            rank=fast_rank,
            lr=fast_lr,
        )
        self.deep_mem = SSMCompressedMirasTitans(
            d_model=d_model,
            rank=deep_rank,
            lr=deep_lr,
        )

        # Learnable mixing logit (initialized to achieve init_fast_weight)
        init_logit = math.log(init_fast_weight / (1.0 - init_fast_weight + 1e-8))
        self.mix_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # Context-dependent gate
        if context_gate:
            self.mix_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.mix_gate = None

    @classmethod
    def from_config(cls, cfg: MirasConfig) -> "DualTierMiras":
        """Create DualTierMiras from config."""
        return cls(
            d_model=cfg.d_model,
            fast_rank=cfg.fast_rank,
            deep_rank=cfg.deep_rank,
            fast_lr=cfg.fast_lr,
            deep_lr=cfg.deep_lr,
            init_fast_weight=cfg.init_fast_weight,
            context_gate=cfg.context_gate,
        )

    def compute_mix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute mixing weight for fast tier."""
        base = torch.sigmoid(self.mix_logit)
        if self.mix_gate is None or context is None:
            return base
        # Context-dependent adjustment
        delta = torch.sigmoid(self.mix_gate(context))  # [B, 1]
        w_fast = 0.5 * (base + delta)  # Blend base with context
        return w_fast

    def read(
        self,
        k: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Read from both tiers and combine.

        Args:
            k: Query keys [B, d_model]
            context: Context for gating [B, d_model]
        Returns:
            Dictionary with combined output and tier outputs
        """
        v_fast = self.fast_mem.read(k)
        v_deep = self.deep_mem.read(k)

        w_fast = self.compute_mix(context)

        # Handle broadcasting
        if w_fast.dim() == 0:
            w_fast = w_fast.view(1, 1).to(k.device)
        elif w_fast.dim() == 1:
            w_fast = w_fast.unsqueeze(-1)
        w_fast = w_fast.to(k.device)
        w_deep = 1.0 - w_fast

        v = w_fast * v_fast + w_deep * v_deep

        return {
            "v": v,
            "v_fast": v_fast,
            "v_deep": v_deep,
            "w_fast": w_fast,
        }

    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update both tiers.

        Args:
            k: Keys [B, d_model]
            v: Target values [B, d_model]
            weight: Importance weights (e.g., |advantage|) [B]
            context: Context for stats [B, d_model]
        """
        stats: Dict[str, float] = {}

        # Update both tiers
        fast_stats = self.fast_mem.update(k, v, weight)
        for key, val in fast_stats.items():
            stats[f"fast/{key}"] = float(val)

        deep_stats = self.deep_mem.update(k, v, weight)
        for key, val in deep_stats.items():
            stats[f"deep/{key}"] = float(val)

        # Record mixing stats
        w_fast = self.compute_mix(context)
        if w_fast.dim() > 1:
            w_mean = float(w_fast.mean().item())
        elif w_fast.dim() == 1:
            w_mean = float(w_fast.mean().item())
        else:
            w_mean = float(w_fast.item())

        stats["mix/fast_weight"] = w_mean
        stats["mix/deep_weight"] = 1.0 - w_mean
        stats["mix/logit"] = float(self.mix_logit.data.item())

        return stats

    def reset_state(self) -> None:
        """Reset tier states (momentum buffers, etc.)."""
        self.fast_mem.reset_state()
        self.deep_mem.reset_state()

    def reset_parameters(self) -> None:
        """Full parameter reset."""
        self.fast_mem.reset_parameters()
        self.deep_mem.reset_parameters()

        init_logit = math.log(self.init_fast_weight / (1.0 - self.init_fast_weight + 1e-8))
        self.mix_logit.data.copy_(torch.tensor(init_logit, dtype=torch.float32))

        if self.mix_gate is not None:
            for m in self.mix_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)
