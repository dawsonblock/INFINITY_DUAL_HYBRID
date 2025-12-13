"""
miras.py

Dual-Tier Miras Parametric Memory System.

Architecture:
- Fast Tier (SSMCompressedMiras): SGD-based low-rank memory for rapid
  adaptation
- Deep Tier (SSMCompressedMirasTitans): Momentum-based with Huber loss and
  retention gate
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
        norm_reg: float = 0.0,
        momentum: float = 0.9,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        init_scale: float = 0.1,
        grad_clip: float = 1.0,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr = lr
        self.l2_reg = l2_reg
        self.norm_reg = norm_reg
        self.momentum = momentum
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Low-rank factors B, C and diagonal D
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        # Scale for tanh output
        self.register_buffer(
            "scale",
            torch.tensor(init_scale, dtype=torch.float32),
        )

        # Momentum buffers
        self.register_buffer("S_B", torch.zeros_like(self.B))
        self.register_buffer("S_C", torch.zeros_like(self.C))

        # EMA shadow parameters
        if use_ema:
            self.register_buffer("ema_B", self.B.data.clone())
            self.register_buffer("ema_C", self.C.data.clone())

        # Retention gate: learns when to forget vs retain
        self.retention_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def W(self) -> torch.Tensor:
        """Compute full weight matrix from low-rank factors."""
        B = self.B
        C = self.C
        if self.use_ema and hasattr(self, "ema_B") and hasattr(self, "ema_C"):
            B = self.ema_B
            C = self.ema_C

        low_rank = B @ C.t()
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
        low_rank = self.B @ self.C.t()
        W = self.scale * torch.tanh(low_rank)
        W = W + torch.diag(self.D)
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

        if self.norm_reg > 0:
            gradB = gradB + self.norm_reg * self.B
            gradC = gradC + self.norm_reg * self.C

        # Gradient clipping
        if self.grad_clip > 0:
            gradB_norm = gradB.norm()
            gradC_norm = gradC.norm()
            if gradB_norm > self.grad_clip:
                gradB = gradB * (self.grad_clip / (gradB_norm + 1e-8))
            if gradC_norm > self.grad_clip:
                gradC = gradC * (self.grad_clip / (gradC_norm + 1e-8))

        # Momentum update
        self.S_B = self.momentum * self.S_B - self.lr * gradB
        self.S_C = self.momentum * self.S_C - self.lr * gradC

        # Retention gate: determines how much to retain vs update
        k_mean = k.mean(dim=0, keepdim=True)
        alpha_t = torch.sigmoid(self.retention_gate(k_mean)).squeeze()

        # Apply update with retention
        self.B.data = (1.0 - alpha_t) * self.B.data + self.S_B
        self.C.data = (1.0 - alpha_t) * self.C.data + self.S_C

        # EMA update of shadow parameters
        if self.use_ema:
            self.ema_B = (
                self.ema_decay * self.ema_B
                + (1 - self.ema_decay) * self.B.data
            )
            self.ema_C = (
                self.ema_decay * self.ema_C
                + (1 - self.ema_decay) * self.C.data
            )

        b_norm = float(self.B.data.norm().item())
        c_norm = float(self.C.data.norm().item())
        err_l2 = float(err.norm(dim=-1).mean().item())
        retention = (
            float(alpha_t.mean().item())
            if alpha_t.dim() > 0
            else float(alpha_t.item())
        )

        if not (
            math.isfinite(b_norm)
            and math.isfinite(c_norm)
            and math.isfinite(err_l2)
            and math.isfinite(retention)
        ):
            raise FloatingPointError("Non-finite MIRAS deep-tier update")

        stats = {
            "B_norm": b_norm,
            "C_norm": c_norm,
            "S_B_norm": float(self.S_B.norm().item()),
            "S_C_norm": float(self.S_C.norm().item()),
            "gradB_norm": float(gradB.norm().item()),
            "gradC_norm": float(gradC.norm().item()),
            "retention": retention,
            "err_l2": err_l2,
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
        norm_reg: float = 0.0,
        init_scale: float = 0.1,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr = lr
        self.l2_reg = l2_reg
        self.norm_reg = norm_reg
        self.grad_clip = grad_clip

        # Low-rank factors
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer(
            "scale",
            torch.tensor(init_scale, dtype=torch.float32),
        )

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

        # Compute gradients
        gradW = -(err.t() @ k) / (batch_size + 1e-8) + self.l2_reg * W
        gradB = gradW @ self.C
        gradC = gradW.t() @ self.B

        if self.norm_reg > 0:
            gradB = gradB + self.norm_reg * self.B
            gradC = gradC + self.norm_reg * self.C

        # Gradient clipping
        if self.grad_clip > 0:
            gradB_norm = gradB.norm()
            gradC_norm = gradC.norm()
            if gradB_norm > self.grad_clip:
                gradB = gradB * (self.grad_clip / (gradB_norm + 1e-8))
            if gradC_norm > self.grad_clip:
                gradC = gradC * (self.grad_clip / (gradC_norm + 1e-8))

        # Apply SGD update
        self.B.data.add_(-self.lr * gradB)
        self.C.data.add_(-self.lr * gradC)

        b_norm = float(self.B.data.norm().item())
        c_norm = float(self.C.data.norm().item())
        err_l2 = float(err.norm(dim=-1).mean().item())
        if not (
            math.isfinite(b_norm)
            and math.isfinite(c_norm)
            and math.isfinite(err_l2)
        ):
            raise FloatingPointError("Non-finite MIRAS fast-tier update")

        return {
            "B_norm": b_norm,
            "C_norm": c_norm,
            "err_l2": err_l2,
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
        fast_l2_reg: float = 1e-4,
        fast_init_scale: float = 0.1,
        deep_lr: float = 5e-4,
        deep_l2_reg: float = 1e-4,
        deep_momentum: float = 0.9,
        deep_use_huber: bool = True,
        deep_huber_delta: float = 1.0,
        deep_init_scale: float = 0.1,
        init_fast_weight: float = 0.7,
        context_gate: bool = True,
        grad_clip: float = 1.0,
        norm_reg: float = 0.0,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_gate_enabled = context_gate
        self.init_fast_weight = init_fast_weight
        self._last_update_stats: Dict[str, float] = {}

        # Create fast and deep tiers
        self.fast_mem = SSMCompressedMiras(
            d_model=d_model,
            rank=fast_rank,
            lr=fast_lr,
            l2_reg=fast_l2_reg,
            norm_reg=norm_reg,
            init_scale=fast_init_scale,
            grad_clip=grad_clip,
        )
        self.deep_mem = SSMCompressedMirasTitans(
            d_model=d_model,
            rank=deep_rank,
            lr=deep_lr,
            l2_reg=deep_l2_reg,
            norm_reg=norm_reg,
            momentum=deep_momentum,
            use_huber=deep_use_huber,
            huber_delta=deep_huber_delta,
            init_scale=deep_init_scale,
            grad_clip=grad_clip,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

        # Learnable mixing logit (initialized to achieve init_fast_weight)
        init_logit = math.log(
            init_fast_weight / (1.0 - init_fast_weight + 1e-8)
        )
        self.mix_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

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
            fast_l2_reg=cfg.fast_l2_reg,
            fast_init_scale=cfg.fast_init_scale,
            deep_lr=cfg.deep_lr,
            deep_l2_reg=cfg.deep_l2_reg,
            deep_momentum=cfg.deep_momentum,
            deep_use_huber=cfg.deep_use_huber,
            deep_huber_delta=cfg.deep_huber_delta,
            deep_init_scale=cfg.deep_init_scale,
            init_fast_weight=cfg.init_fast_weight,
            context_gate=cfg.context_gate,
            grad_clip=cfg.grad_clip,
            norm_reg=cfg.norm_reg,
            use_ema=cfg.use_ema,
            ema_decay=cfg.ema_decay,
        )

    def compute_mix(
        self,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        self._last_update_stats = dict(stats)

        return stats

    def reset_state(self) -> None:
        """Reset tier states (momentum buffers, etc.)."""
        self.fast_mem.reset_state()
        self.deep_mem.reset_state()

    def reset(self) -> None:
        """Alias for reset_state() - resets memory without resetting learned
        params."""
        self.reset_state()

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics for diagnostics."""
        w_fast = self.compute_mix(None)
        if w_fast.dim() > 0:
            mix_ratio = float(w_fast.mean().item())
        else:
            mix_ratio = float(w_fast.item())

        out = {
            "fast_B_norm": float(self.fast_mem.B.data.norm().item()),
            "fast_C_norm": float(self.fast_mem.C.data.norm().item()),
            "deep_B_norm": float(self.deep_mem.B.data.norm().item()),
            "deep_C_norm": float(self.deep_mem.C.data.norm().item()),
            "mix_ratio": mix_ratio,
            "mix_logit": float(self.mix_logit.data.item()),
        }

        last = getattr(self, "_last_update_stats", None)
        if isinstance(last, dict):
            pairs = {
                "fast/err_l2": "fast_err_l2",
                "deep/err_l2": "deep_err_l2",
                "deep/retention": "deep_retention",
                "deep/gradB_norm": "deep_gradB_norm",
                "deep/gradC_norm": "deep_gradC_norm",
            }
            for src, dst in pairs.items():
                v = last.get(src)
                if v is not None:
                    out[dst] = float(v)

        return out

    def reset_parameters(self) -> None:
        """Full parameter reset."""
        self.fast_mem.reset_parameters()
        self.deep_mem.reset_parameters()

        init_logit = math.log(
            self.init_fast_weight
            / (1.0 - self.init_fast_weight + 1e-8)
        )
        self.mix_logit.data.copy_(
            torch.tensor(init_logit, dtype=torch.float32)
        )

        if self.mix_gate is not None:
            for m in self.mix_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                            m.weight
                        )
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)
