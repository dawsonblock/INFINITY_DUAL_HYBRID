import math
from typing import Optional, Dict

import torch
import torch.nn as nn


class SSMCompressedMirasTitans(nn.Module):
    """
    Titans-style deep Miras parametric memory.

    W = scale * tanh(B C^T) + diag(D)
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        momentum: float = 0.9,
        use_huber: bool = False,
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

        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
        self.register_buffer("S_B", torch.zeros_like(self.B))
        self.register_buffer("S_C", torch.zeros_like(self.C))

        self.retention_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def W(self) -> torch.Tensor:
        low_rank = self.B @ self.C.t()
        W = self.scale * torch.tanh(low_rank)
        W = W + torch.diag(self.D)
        return W

    def read(self, k: torch.Tensor) -> torch.Tensor:
        W = self.W()
        return k @ W.t()

    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if k.numel() == 0:
            return {}

        device = self.B.device
        k = k.to(device)
        v = v.to(device)
        Bsz = k.shape[0]

        W = self.W()
        v_hat = k @ W.t()
        err = v - v_hat

        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            err = err * weight

        if self.use_huber:
            delta = self.huber_delta
            abs_err = err.abs()
            mask = (abs_err <= delta).float()
            err = mask * err + (1.0 - mask) * delta * err.sign()

        gradW = -(err.t() @ k) / (Bsz + 1e-8) + self.l2_reg * W
        gradB = gradW @ self.C
        gradC = gradW.t() @ self.B

        self.S_B = self.momentum * self.S_B - self.lr * gradB
        self.S_C = self.momentum * self.S_C - self.lr * gradC

        k_mean = k.mean(dim=0, keepdim=True)
        alpha_t = torch.sigmoid(self.retention_gate(k_mean)).squeeze()

        self.B.data = (1.0 - alpha_t) * self.B.data + self.S_B
        self.C.data = (1.0 - alpha_t) * self.C.data + self.S_C

        stats = {
            "miras_B_norm": float(self.B.data.norm().item()),
            "miras_C_norm": float(self.C.data.norm().item()),
            "miras_S_B_norm": float(self.S_B.norm().item()),
            "miras_S_C_norm": float(self.S_C.norm().item()),
            "miras_retention_gate": float(alpha_t),
            "miras_err_l2": float(err.norm(dim=-1).mean().item()),
        }
        return stats

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)
        self.S_B.zero_()
        self.S_C.zero_()


class SSMCompressedMiras(nn.Module):
    """
    Simple low-rank Miras-style parametric memory, used as fast tier.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr = lr
        self.l2_reg = l2_reg

        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

    def W(self) -> torch.Tensor:
        low_rank = self.B @ self.C.t()
        W = self.scale * torch.tanh(low_rank)
        W = W + torch.diag(self.D)
        return W

    def read(self, k: torch.Tensor) -> torch.Tensor:
        W = self.W()
        return k @ W.t()

    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if k.numel() == 0:
            return {}
        W = self.W()
        k = k.to(W.device)
        v = v.to(W.device)
        Bsz = k.shape[0]
        v_hat = k @ W.t()
        err = v - v_hat
        if weight is not None:
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            err = err * weight
        gradW = -(err.t() @ k) / (Bsz + 1e-8) + self.l2_reg * W
        gradB = gradW @ self.C
        gradC = gradW.t() @ self.B
        self.B.data.add_(-self.lr * gradB)
        self.C.data.add_(-self.lr * gradC)
        return {
            "miras_B_norm": float(self.B.data.norm().item()),
            "miras_C_norm": float(self.C.data.norm().item()),
            "miras_err_l2": float(err.norm(dim==-1).mean().item()),
        }

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)


class DualTierMiras(nn.Module):
    """
    Dual-tier Miras combiner.
    """

    def __init__(
        self,
        fast_mem: nn.Module,
        deep_mem: nn.Module,
        d_model: int,
        init_fast_weight: float = 0.7,
        context_gate: bool = True,
        init_fast_log_scale: float = 0.0,
        init_deep_log_scale: float = 0.0,
    ):
        super().__init__()
        assert 0.0 <= init_fast_weight <= 1.0

        self.fast_mem = fast_mem
        self.deep_mem = deep_mem
        self.d_model = d_model
        self.context_gate = context_gate

        self.init_fast_weight = init_fast_weight
        self.init_fast_log_scale = init_fast_log_scale
        self.init_deep_log_scale = init_deep_log_scale

        init_logit = math.log(init_fast_weight / (1.0 - init_fast_weight + 1e-8))
        self.mix_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        if context_gate:
            self.mix_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.mix_gate = None

        self.fast_log_lr_scale = nn.Parameter(torch.tensor(init_fast_log_scale))
        self.deep_log_lr_scale = nn.Parameter(torch.tensor(init_deep_log_scale))

    def compute_mix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = torch.sigmoid(self.mix_logit)
        if self.mix_gate is None or context is None:
            return base
        delta = torch.sigmoid(self.mix_gate(context))  # [B,1]
        w_fast = 0.5 * (base + delta)
        return w_fast

    def read(self, k: torch.Tensor, context: Optional[torch.Tensor] = None):
        v_fast = self.fast_mem.read(k)
        v_deep = self.deep_mem.read(k)

        w_fast = self.compute_mix(context)
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
        stats: Dict[str, float] = {}
        fast_scale = self.fast_log_lr_scale.exp()
        deep_scale = self.deep_log_lr_scale.exp()

        k_fast = k * fast_scale
        k_deep = k * deep_scale

        if hasattr(self.fast_mem, "update"):
            fast_stats = self.fast_mem.update(k_fast, v, weight)
            for key, val in fast_stats.items():
                stats[f"mem_fast/{key}"] = float(val)

        if hasattr(self.deep_mem, "update"):
            deep_stats = self.deep_mem.update(k_deep, v, weight)
            for key, val in deep_stats.items():
                stats[f"mem_deep/{key}"] = float(val)

        w_fast = self.compute_mix(context)
        if w_fast.dim() > 1:
            w_mean = float(w_fast.mean().item())
        else:
            w_mean = float(w_fast.item())

        stats["mem_mix/fast_weight_mean"] = w_mean
        stats["mem_mix/deep_weight_mean"] = 1.0 - w_mean
        stats["mem_mix/mix_logit"] = float(self.mix_logit.data.item())
        stats["mem_mix/fast_lr_scale"] = float(fast_scale.item())
        stats["mem_mix/deep_lr_scale"] = float(deep_scale.item())
        return stats

    def reset_parameters(self, preserve_meta: bool = True) -> None:
        if hasattr(self.fast_mem, "reset_parameters"):
            self.fast_mem.reset_parameters()
        if hasattr(self.deep_mem, "reset_parameters"):
            self.deep_mem.reset_parameters()

        if not preserve_meta:
            init_logit = math.log(self.init_fast_weight / (1.0 - self.init_fast_weight + 1e-8))
            self.mix_logit.data.copy_(torch.tensor(init_logit, dtype=torch.float32))
            self.fast_log_lr_scale.data.copy_(torch.tensor(self.init_fast_log_scale))
            self.deep_log_lr_scale.data.copy_(torch.tensor(self.init_deep_log_scale))

        if self.mix_gate is not None:
            for m in self.mix_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)
