"""
infinity_v3_unified.py

Unified Infinity architecture merging the best components:
- Mamba2 backbone (with Transformer fallback)
- Dual-Tier Miras parametric memory (fast + Titans deep)
- FAISS IVF-PQ episodic LTM
- RMD gate for selective LTM writes
- Context-gated memory fusion
- GRPO + LM joint training with advantage-weighted Miras updates

This is the canonical merged build combining:
1. infinity_v3_stack.py (backbone, FAISS LTM, GRPO, environments)
2. dual_tier_miras.py (parametric memory tiers)
3. continuous_memory_mamba_agent_v3_infinity.py (actor-critic pattern)
"""

import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Optional Dependencies
# ============================================================

try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    HAS_MAMBA = True
except Exception:
    Mamba2 = None
    HAS_MAMBA = False


# ============================================================
# Tokenizer
# ============================================================

class SimpleAsciiTokenizer:
    """Byte-level tokenizer (vocab_size=256)."""
    
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.LongTensor:
        return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

    def decode(self, ids: torch.LongTensor) -> str:
        return "".join(chr(int(x) % 256) for x in ids.tolist())


# ============================================================
# Environments
# ============================================================

class LogicPuzzleEnv:
    """Short-horizon arithmetic Q&A."""

    def __init__(self, max_val: int = 99, device: str = "cpu"):
        self.max_val = max_val
        self.device = device

    def sample_batch(self, batch_size: int, tokenizer: SimpleAsciiTokenizer):
        prompts, answers = [], []
        for _ in range(batch_size):
            a = torch.randint(0, self.max_val + 1, (1,)).item()
            b = torch.randint(0, self.max_val + 1, (1,)).item()
            prompts.append(tokenizer.encode(f"Q: {a} + {b} = ").to(self.device))
            answers.append(str(a + b))
        return prompts, answers


class LongHorizonMemoryEnv:
    """
    Long-horizon reasoning task with buried facts and distractors.
    Tests memory retrieval over extended sequences.
    """

    def __init__(self, story_len: int = 6, noise_len: int = 4, device: str = "cpu"):
        self.story_len = story_len
        self.noise_len = noise_len
        self.device = device
        self._subjects = ["The cat", "A robot", "The mountain", "This system", "The engine"]
        self._verbs = ["jumps", "vibrates", "drifts", "syncs", "glows"]
        self._objects = ["in the dark.", "under pressure.", "beyond reason.", "without noise.", "near zero."]

    def _make_noise_sentence(self) -> str:
        import random
        return f"{random.choice(self._subjects)} {random.choice(self._verbs)} {random.choice(self._objects)}"

    def sample_batch(self, batch_size: int, tokenizer: SimpleAsciiTokenizer):
        import random
        prompts, answers = [], []
        for _ in range(batch_size):
            x_val = random.randint(0, 99)
            fact = f"FACT: X = {x_val}."
            noise = [self._make_noise_sentence() for _ in range(self.story_len)]
            insert_pos = random.randint(0, max(1, self.story_len // 2))
            noise.insert(insert_pos, fact)
            more_noise = [self._make_noise_sentence() for _ in range(self.noise_len)]
            question = "QUESTION: What is X? Answer:"
            story = " ".join(noise + more_noise + [question])
            prompts.append(tokenizer.encode(story).to(self.device))
            answers.append(str(x_val))
        return prompts, answers


# ============================================================
# Reward Oracle
# ============================================================

class PuzzleOracle:
    """Extracts last integer from generation and compares to ground truth."""

    def __init__(self, tokenizer: SimpleAsciiTokenizer):
        self.tokenizer = tokenizer

    def _extract_last_int(self, s: str) -> Optional[int]:
        ints = re.findall(r"-?\d+", s)
        return int(ints[-1]) if ints else None

    def score(self, seqs: torch.LongTensor, answers: List[str]) -> torch.Tensor:
        B = seqs.shape[0]
        rewards = torch.zeros(B, dtype=torch.float32, device=seqs.device)
        for i in range(B):
            pred = self._extract_last_int(self.tokenizer.decode(seqs[i]))
            try:
                gt = int(answers[i])
            except Exception:
                gt = None
            if pred is not None and gt is not None and pred == gt:
                rewards[i] = 1.0
        return rewards


# ============================================================
# Dual-Tier Miras Parametric Memory
# ============================================================

class SSMCompressedMiras(nn.Module):
    """
    Fast-tier low-rank parametric memory.
    W = scale * tanh(B @ C^T) + diag(D)
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

        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

    def W(self) -> torch.Tensor:
        return self.scale * torch.tanh(self.B @ self.C.t()) + torch.diag(self.D)

    def read(self, k: torch.Tensor) -> torch.Tensor:
        return k @ self.W().t()

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if k.numel() == 0:
            return {}
        W = self.W()
        k, v = k.to(W.device), v.to(W.device)
        err = v - k @ W.t()
        if weight is not None:
            err = err * (weight.unsqueeze(-1) if weight.dim() == 1 else weight)
        Bsz = k.shape[0]
        gradW = -(err.t() @ k) / (Bsz + 1e-8) + self.l2_reg * W
        self.B.data.add_(-self.lr * (gradW @ self.C))
        self.C.data.add_(-self.lr * (gradW.t() @ self.B))
        return {"fast_B_norm": float(self.B.norm()), "fast_C_norm": float(self.C.norm())}


class SSMCompressedMirasTitans(nn.Module):
    """
    Deep-tier Titans-style parametric memory with:
    - Momentum-based updates
    - Huber loss for robustness
    - Adaptive retention gate
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

        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.C = nn.Parameter(torch.zeros(d_model, rank))
        self.D = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
        self.register_buffer("S_B", torch.zeros_like(self.B))
        self.register_buffer("S_C", torch.zeros_like(self.C))

        # Retention gate: controls how much old memory to keep
        self.retention_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def W(self) -> torch.Tensor:
        return self.scale * torch.tanh(self.B @ self.C.t()) + torch.diag(self.D)

    def read(self, k: torch.Tensor) -> torch.Tensor:
        return k @ self.W().t()

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if k.numel() == 0:
            return {}

        k, v = k.to(self.B.device), v.to(self.B.device)
        W = self.W()
        err = v - k @ W.t()

        if weight is not None:
            err = err * (weight.unsqueeze(-1) if weight.dim() == 1 else weight)

        if self.use_huber:
            abs_err = err.abs()
            mask = (abs_err <= self.huber_delta).float()
            err = mask * err + (1.0 - mask) * self.huber_delta * err.sign()

        Bsz = k.shape[0]
        gradW = -(err.t() @ k) / (Bsz + 1e-8) + self.l2_reg * W

        # Momentum update
        self.S_B = self.momentum * self.S_B - self.lr * (gradW @ self.C)
        self.S_C = self.momentum * self.S_C - self.lr * (gradW.t() @ self.B)

        # Adaptive retention
        alpha = torch.sigmoid(self.retention_gate(k.mean(dim=0, keepdim=True))).squeeze()
        self.B.data = (1.0 - alpha) * self.B.data + self.S_B
        self.C.data = (1.0 - alpha) * self.C.data + self.S_C

        return {
            "deep_B_norm": float(self.B.norm()),
            "deep_C_norm": float(self.C.norm()),
            "deep_retention": float(alpha),
        }


class DualTierMiras(nn.Module):
    """
    Context-gated combination of fast and deep parametric memory tiers.
    Provides working memory with different learning dynamics.
    """

    def __init__(
        self,
        d_model: int,
        fast_rank: int = 32,
        deep_rank: int = 32,
        init_fast_weight: float = 0.7,
        context_gate: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        self.fast_mem = SSMCompressedMiras(d_model, rank=fast_rank, lr=1e-3)
        self.deep_mem = SSMCompressedMirasTitans(d_model, rank=deep_rank, lr=5e-4, momentum=0.9, use_huber=True)

        init_logit = math.log(init_fast_weight / (1.0 - init_fast_weight + 1e-8))
        self.mix_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.mix_gate = None
        if context_gate:
            self.mix_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )

    def compute_mix(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = torch.sigmoid(self.mix_logit)
        if self.mix_gate is None or context is None:
            return base
        delta = torch.sigmoid(self.mix_gate(context))
        return 0.5 * (base + delta)

    def read(self, k: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        v_fast = self.fast_mem.read(k)
        v_deep = self.deep_mem.read(k)
        w_fast = self.compute_mix(context)
        if w_fast.dim() == 0:
            w_fast = w_fast.view(1, 1)
        elif w_fast.dim() == 1:
            w_fast = w_fast.unsqueeze(-1)
        return w_fast * v_fast + (1.0 - w_fast) * v_deep

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor, weight: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> Dict[str, float]:
        stats = {}
        stats.update(self.fast_mem.update(k, v, weight))
        stats.update(self.deep_mem.update(k, v, weight))
        stats["mix_weight"] = float(self.compute_mix(context).mean())
        return stats


# ============================================================
# FAISS Episodic LTM
# ============================================================

@dataclass
class LTMConfig:
    d_key: int = 256
    d_value: int = 256
    max_size: int = 100_000
    use_faiss: bool = True
    nlist: int = 4096
    m: int = 16
    nprobe: int = 8
    device: str = "cpu"


class SimpleLTM(nn.Module):
    """Fallback cosine-similarity LTM when FAISS unavailable."""

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("keys", torch.empty(0, cfg.d_key))
        self.register_buffer("values", torch.empty(0, cfg.d_value))

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.numel() == 0:
            return
        self.keys = torch.cat([self.keys, keys.detach()], dim=0)
        self.values = torch.cat([self.values, values.detach()], dim=0)
        if self.keys.shape[0] > self.cfg.max_size:
            self.keys = self.keys[-self.cfg.max_size:]
            self.values = self.values[-self.cfg.max_size:]

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        B = queries.shape[0]
        if self.keys.shape[0] == 0:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)
        k = F.normalize(self.keys, dim=-1)
        q = F.normalize(queries, dim=-1)
        sim = q @ k.t()
        k_eff = min(top_k, sim.shape[1])
        vals, idx = torch.topk(sim, k_eff, dim=-1)
        weights = F.softmax(vals, dim=-1)
        return (self.values[idx] * weights.unsqueeze(-1)).sum(dim=1)


class FaissIVFPQLTM(nn.Module):
    """FAISS IVF-PQ backed episodic LTM for scalable memory."""

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        if not HAS_FAISS:
            raise RuntimeError("FAISS not available")
        self.cfg = cfg
        self.register_buffer("keys_buf", torch.empty(0, cfg.d_key))
        self.register_buffer("values_buf", torch.empty(0, cfg.d_value))

        self.quantizer = faiss.IndexFlatL2(cfg.d_key)
        self.index = faiss.IndexIVFPQ(self.quantizer, cfg.d_key, cfg.nlist, cfg.m, 8)
        self.index.nprobe = cfg.nprobe
        self.trained = False

    @torch.no_grad()
    def _rebuild_index(self):
        self.index.reset()
        if self.keys_buf.shape[0] == 0:
            self.trained = False
            return
        keys_np = self.keys_buf.cpu().numpy().astype("float32")
        if not self.index.is_trained:
            self.index.train(keys_np)
        self.index.add(keys_np)
        self.trained = True

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.numel() == 0:
            return
        self.keys_buf = torch.cat([self.keys_buf, keys.detach()], dim=0)
        self.values_buf = torch.cat([self.values_buf, values.detach()], dim=0)
        if self.keys_buf.shape[0] > self.cfg.max_size:
            self.keys_buf = self.keys_buf[-self.cfg.max_size:]
            self.values_buf = self.values_buf[-self.cfg.max_size:]
        self._rebuild_index()

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        B = queries.shape[0]
        if self.keys_buf.shape[0] == 0 or not self.trained:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)
        q = queries.cpu().numpy().astype("float32")
        k_eff = min(top_k, self.index.ntotal)
        D, I = self.index.search(q, k_eff)
        I_t = torch.from_numpy(I).to(self.values_buf.device)
        D_t = torch.from_numpy(D).to(self.values_buf.device)
        weights = F.softmax(-D_t, dim=-1)
        return (self.values_buf[I_t] * weights.unsqueeze(-1)).sum(dim=1)


def build_ltm(cfg: LTMConfig) -> nn.Module:
    if cfg.use_faiss and HAS_FAISS:
        return FaissIVFPQLTM(cfg)
    return SimpleLTM(cfg)


# ============================================================
# Backbone: Mamba2 / Transformer
# ============================================================

class TransformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, mask=attn_mask)


class MambaBackbone(nn.Module):
    def __init__(self, d_model: int, num_layers: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError("mamba-ssm not available")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=d_state),
            })
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer["mamba"](layer["norm"](x))
        return x


class UnifiedBackbone(nn.Module):
    """Auto-selects Mamba2 if available, else Transformer."""

    def __init__(self, d_model: int, n_heads: int, num_layers: int, use_mamba: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_mamba = use_mamba and HAS_MAMBA
        if self.use_mamba:
            self.core = MambaBackbone(d_model, num_layers)
        else:
            self.core = TransformerBackbone(d_model, n_heads, num_layers, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_mamba:
            return self.core(x)
        return self.core(x, attn_mask)


# ============================================================
# Unified Infinity Agent
# ============================================================

@dataclass
class UnifiedAgentConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 1024
    use_mamba: bool = True
    dropout: float = 0.1
    # Dual-Tier Miras
    miras_fast_rank: int = 32
    miras_deep_rank: int = 32
    miras_init_fast_weight: float = 0.7
    # Episodic LTM
    ltm_max_size: int = 200_000
    ltm_use_faiss: bool = True
    ltm_nlist: int = 4096
    ltm_m: int = 16
    ltm_nprobe: int = 8
    # Memory retrieval
    memory_top_k: int = 16
    rmd_commit_ratio: float = 0.25  # top 25% states committed to LTM


class UnifiedInfinityAgent(nn.Module):
    """
    Unified agent combining:
    - Mamba2/Transformer backbone
    - Dual-Tier Miras (parametric working memory)
    - FAISS LTM (episodic long-term memory)
    - RMD gate for selective LTM writes
    - Advantage-weighted Miras updates
    """

    def __init__(self, cfg: UnifiedAgentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, d)

        # Backbone
        self.backbone = UnifiedBackbone(d, cfg.n_heads, cfg.num_layers, cfg.use_mamba, cfg.dropout)

        # Dual-Tier Miras (parametric working memory)
        self.miras = DualTierMiras(
            d_model=d,
            fast_rank=cfg.miras_fast_rank,
            deep_rank=cfg.miras_deep_rank,
            init_fast_weight=cfg.miras_init_fast_weight,
            context_gate=True,
        )
        self.miras_key_proj = nn.Linear(d, d)
        self.miras_val_proj = nn.Linear(d, d)

        # FAISS LTM (episodic long-term memory)
        ltm_cfg = LTMConfig(
            d_key=d, d_value=d, max_size=cfg.ltm_max_size,
            use_faiss=cfg.ltm_use_faiss, nlist=cfg.ltm_nlist,
            m=cfg.ltm_m, nprobe=cfg.ltm_nprobe,
        )
        self.ltm = build_ltm(ltm_cfg)
        self.ltm_key_proj = nn.Linear(d, d)
        self.ltm_val_proj = nn.Linear(d, d)

        # RMD gate: decides which states to commit to LTM
        self.rmd_gate = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Memory fusion
        fused_dim = d + d + d  # backbone + miras + ltm
        self.fuse_proj = nn.Linear(fused_dim, d)
        self.fuse_ln = nn.LayerNorm(d)

        # Output heads
        self.lm_head = nn.Linear(d, cfg.vocab_size)
        self.value_head = nn.Linear(d, 1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        advantage: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        B, T = input_ids.shape
        cfg = self.cfg

        # Embeddings
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # Backbone
        attn_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        h = self.backbone(x, attn_mask=attn_mask)  # [B, T, d]

        # ---- Miras Read (parametric working memory) ----
        miras_k = self.miras_key_proj(h)  # [B, T, d]
        miras_v = self.miras.read(miras_k.view(B * T, -1), context=h.view(B * T, -1))
        miras_v = miras_v.view(B, T, -1)

        # ---- LTM Read (episodic long-term memory) ----
        ltm_q = self.ltm_key_proj(h[:, -1])  # [B, d] - query from last token
        ltm_v = self.ltm.retrieve(ltm_q, top_k=cfg.memory_top_k)  # [B, d]
        ltm_v_full = ltm_v.unsqueeze(1).expand(B, T, -1)  # [B, T, d]

        # ---- Memory Fusion ----
        fused = torch.cat([h, miras_v, ltm_v_full], dim=-1)
        fused = self.fuse_ln(self.fuse_proj(fused))  # [B, T, d]

        # ---- Memory Updates (no grad) ----
        if self.training:
            with torch.no_grad():
                # Miras update with advantage weighting
                miras_target = self.miras_val_proj(h).detach()
                
                # Handle advantage shape: [B] -> [B*T] by broadcasting across timesteps
                if advantage is not None:
                    adv = advantage.abs()
                    if adv.dim() == 1 and adv.shape[0] == B:
                        weight = adv.unsqueeze(-1).expand(B, T).reshape(-1)
                    elif adv.numel() == B * T:
                        weight = adv.view(-1)
                    else:
                        weight = None  # Shape mismatch, skip weighting
                else:
                    weight = None
                
                # Flatten for batch update
                self.miras.update(
                    miras_k.view(B * T, -1),
                    miras_target.view(B * T, -1),
                    weight=weight,
                    context=h.view(B * T, -1),
                )

                # RMD-gated LTM write
                rmd_scores = torch.sigmoid(self.rmd_gate(h))  # [B, T, 1]
                k_commit = max(1, int(T * cfg.rmd_commit_ratio))
                _, topk_idx = torch.topk(rmd_scores.squeeze(-1), k_commit, dim=-1)

                # Gather states to commit
                commit_h = []
                for b in range(B):
                    commit_h.append(h[b, topk_idx[b]])
                commit_h = torch.cat(commit_h, dim=0)  # [B * k_commit, d]

                if commit_h.numel() > 0:
                    ltm_keys = self.ltm_key_proj(commit_h)
                    ltm_vals = self.ltm_val_proj(commit_h)
                    self.ltm.add(ltm_keys, ltm_vals)

        # ---- Output Heads ----
        logits = self.lm_head(fused)
        values = self.value_head(fused).squeeze(-1)

        return {"logits": logits, "values": values, "hidden": fused}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.LongTensor:
        self.eval()
        seq = input_ids.clone()
        for _ in range(max_new_tokens):
            if seq.shape[1] >= self.cfg.max_seq_len:
                break
            out = self.forward(seq)
            logits = out["logits"][:, -1] / max(temperature, 1e-6)
            if top_k > 0:
                vals, idx = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
        return seq


# ============================================================
# GRPO Harness
# ============================================================

@dataclass
class GRPOConfig:
    group_size: int = 4
    max_gen_len: int = 32
    temperature: float = 0.8
    top_k: int = 32
    kl_beta: float = 0.01


class GRPOHarness:
    """Group Relative Policy Optimization with advantage computation."""

    def __init__(self, agent: UnifiedInfinityAgent, oracle: PuzzleOracle, cfg: GRPOConfig):
        self.agent = agent
        self.oracle = oracle
        self.cfg = cfg

    def compute_grpo_loss(
        self,
        prompts: List[torch.LongTensor],
        answers: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Returns (loss, advantages, stats)."""
        device = next(self.agent.parameters()).device
        B = len(prompts)

        # Pad prompts
        max_len = max(p.shape[0] for p in prompts)
        input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompts):
            input_ids[i, :p.shape[0]] = p.to(device)

        # Generate
        with torch.no_grad():
            seqs = self.agent.generate(input_ids, max_new_tokens=self.cfg.max_gen_len,
                                        temperature=self.cfg.temperature, top_k=self.cfg.top_k)

        # Forward with advantage placeholder (will compute below)
        out = self.agent(seqs, advantage=None)
        logits = out["logits"]
        values = out["values"].mean(dim=1)

        # Log probs
        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(2, seqs.unsqueeze(-1)).squeeze(-1)
        logprob_episode = token_logprobs.sum(dim=1)

        # Rewards
        rewards = self.oracle.score(seqs, answers)

        # Group-wise baselines
        G = self.cfg.group_size
        B_eff = (B // G) * G
        rewards_eff = rewards[:B_eff]
        logprob_eff = logprob_episode[:B_eff]
        values_eff = values[:B_eff]

        rewards_group = rewards_eff.view(-1, G)
        baseline = rewards_group.mean(dim=1, keepdim=True)
        advantages = (rewards_group - baseline).view(-1)

        # Losses
        pg_loss = -(advantages.detach() * logprob_eff).mean()
        value_loss = F.mse_loss(values_eff, rewards_eff)
        
        # Entropy bonus for exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * logprobs).sum(dim=-1).mean()
        
        kl_proxy = (logprobs ** 2).mean()
        loss = pg_loss + 0.5 * value_loss - 0.01 * entropy + self.cfg.kl_beta * kl_proxy

        stats = {
            "rl_loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "value_loss": float(value_loss.item()),
            "kl_proxy": float(kl_proxy.item()),
            "avg_reward": float(rewards.mean().item()),
        }
        return loss, advantages, stats


# ============================================================
# Unified Trainer
# ============================================================

class UnifiedTrainer:
    """Joint GRPO + LM trainer with advantage-weighted memory updates."""

    def __init__(
        self,
        agent: UnifiedInfinityAgent,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 10,
        total_steps: int = 100,
    ):
        self.agent = agent
        self.optimizer = torch.optim.AdamW(
            agent.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lr = lr

    def get_lr_scale(self) -> float:
        """Warmup then cosine decay."""
        if self.step_count < self.warmup_steps:
            return (self.step_count + 1) / self.warmup_steps
        progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def update_lr(self):
        scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * scale

    def lm_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = self.agent(input_ids)
        logits = out["logits"]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return loss, {"lm_loss": float(loss.item())}

    def joint_step(
        self,
        harness: GRPOHarness,
        rl_prompts: List[torch.LongTensor],
        rl_answers: List[str],
        lm_input_ids: torch.LongTensor,
        lm_labels: torch.LongTensor,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        # Update learning rate
        self.update_lr()
        
        self.optimizer.zero_grad(set_to_none=True)

        # RL loss with advantages
        rl_loss, advantages, rl_stats = harness.compute_grpo_loss(
            rl_prompts, rl_answers
        )

        # LM loss
        lm_loss, lm_stats = self.lm_loss(lm_input_ids, lm_labels)

        # Combined loss
        total_loss = alpha * lm_loss + (1.0 - alpha) * rl_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.step_count += 1

        stats = {"total_loss": float(total_loss.item())}
        stats.update(rl_stats)
        stats.update(lm_stats)
        stats["lr"] = self.optimizer.param_groups[0]['lr']
        return stats


# ============================================================
# Utility Functions
# ============================================================

def build_lm_batch_from_env(
    tokenizer: SimpleAsciiTokenizer,
    prompts: List[torch.LongTensor],
    answers: List[str],
    device: str,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Build LM inputs by concatenating prompt + answer."""
    B = len(prompts)
    seqs = []
    for i in range(B):
        ans_ids = tokenizer.encode(answers[i])
        seq = torch.cat([prompts[i], ans_ids.to(prompts[i].device)], dim=0)
        seqs.append(seq)
    max_len = max(s.shape[0] for s in seqs)
    input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        input_ids[i, :L] = s.to(device)
        labels[i, :L-1] = s[1:].to(device)
    return input_ids, labels


# ============================================================
# Demo Entry Point
# ============================================================

def unified_train_demo(
    steps: int = 100,
    batch_size: int = 16,
    use_long_env: bool = True,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Unified Infinity v3] Device: {device}")
    print(f"[Unified Infinity v3] Mamba2 available: {HAS_MAMBA}")
    print(f"[Unified Infinity v3] FAISS available: {HAS_FAISS}")

    tokenizer = SimpleAsciiTokenizer()
    env = LongHorizonMemoryEnv(device=device) if use_long_env else LogicPuzzleEnv(device=device)
    oracle = PuzzleOracle(tokenizer)

    cfg = UnifiedAgentConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=4,
        max_seq_len=1024,
        use_mamba=True,
        dropout=0.1,
        miras_fast_rank=32,
        miras_deep_rank=32,
        miras_init_fast_weight=0.7,
        ltm_max_size=200_000,
        ltm_use_faiss=True,
        memory_top_k=16,
        rmd_commit_ratio=0.25,
    )

    agent = UnifiedInfinityAgent(cfg).to(device)
    trainer = UnifiedTrainer(
        agent,
        lr=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=10,
        total_steps=steps,
    )
    grpo_cfg = GRPOConfig(
        group_size=4, max_gen_len=32, temperature=0.8, top_k=32, kl_beta=0.01
    )
    harness = GRPOHarness(agent, oracle, grpo_cfg)

    print(f"\n[Training for {steps} steps with batch_size={batch_size}]\n")

    for step in range(1, steps + 1):
        rl_prompts, rl_answers = env.sample_batch(batch_size, tokenizer)
        rl_prompts = [p.to(device) for p in rl_prompts]

        lm_prompts, lm_answers = env.sample_batch(batch_size, tokenizer)
        lm_prompts = [p.to(device) for p in lm_prompts]
        lm_input_ids, lm_labels = build_lm_batch_from_env(
            tokenizer, lm_prompts, lm_answers, device
        )

        stats = trainer.joint_step(
            harness, rl_prompts, rl_answers, lm_input_ids, lm_labels, alpha=0.5
        )

        if step % 10 == 0:
            lr = stats.get('lr', 0)
            print(
                f"[step {step:3d}] total={stats['total_loss']:7.4f} "
                f"rl={stats['rl_loss']:7.4f} lm={stats['lm_loss']:7.4f} "
                f"reward={stats['avg_reward']:.3f} lr={lr:.2e}"
            )

    print("\n[Training complete]")


if __name__ == "__main__":
    steps = int(os.environ.get("INFINITY_DEMO_STEPS", "100"))
    unified_train_demo(steps=steps, batch_size=16, use_long_env=True)
