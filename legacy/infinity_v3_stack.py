import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
    HAS_MAMBA = True
except Exception:
    Mamba2 = None
    HAS_MAMBA = False


# ============================================================
# Tokenizer
# ============================================================

class SimpleAsciiTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.LongTensor:
        data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        return data

    def decode(self, ids: torch.LongTensor) -> str:
        return "".join(chr(int(x) % 256) for x in ids.tolist())


# ============================================================
# Environments
# ============================================================

class LogicPuzzleEnv:
    """Short-horizon arithmetic Q&A: Q: a + b = ?"""

    def __init__(self, max_val: int = 99, device: str = "cpu"):
        self.max_val = max_val
        self.device = device

    def sample_batch(self, batch_size: int, tokenizer: SimpleAsciiTokenizer):
        prompts = []
        answers = []
        for _ in range(batch_size):
            a = torch.randint(0, self.max_val + 1, (1,)).item()
            b = torch.randint(0, self.max_val + 1, (1,)).item()
            prompt_str = f"Q: {a} + {b} = "
            ans_str = f"{a + b}"
            prompts.append(tokenizer.encode(prompt_str).to(self.device))
            answers.append(ans_str)
        return prompts, answers


class LongHorizonMemoryEnv:
    """
    Long sequence reasoning task.

    Builds a story with several distractor sentences, then a question that
    requires remembering a specific fact from early in the sequence.

    Example:
      "Fact: X = 7. Noise ... Noise ... Question: What is X?"
    """

    def __init__(self, story_len: int = 6, noise_len: int = 4, device: str = "cpu"):
        self.story_len = story_len
        self.noise_len = noise_len
        self.device = device

    def _make_noise_sentence(self) -> str:
        subjects = ["The cat", "A robot", "The mountain", "This system", "The engine"]
        verbs = ["jumps", "vibrates", "drifts", "syncs", "glows"]
        objs = ["in the dark.", "under pressure.", "beyond reason.", "without noise.", "near zero."]
        import random
        return f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objs)}"

    def sample_batch(self, batch_size: int, tokenizer: SimpleAsciiTokenizer):
        import random
        prompts = []
        answers = []
        for _ in range(batch_size):
            x_val = random.randint(0, 99)
            # Core fact sentence
            fact_sentence = f"FACT: X = {x_val}."
            # Distractors
            noise_sentences = [self._make_noise_sentence() for _ in range(self.story_len)]
            # Insert fact near the beginning
            insert_pos = random.randint(0, max(1, self.story_len // 2))
            noise_sentences.insert(insert_pos, fact_sentence)
            # Extra noise block
            more_noise = [self._make_noise_sentence() for _ in range(self.noise_len)]
            # Question at end
            question = "QUESTION: What is X? Answer:"
            story = " ".join(noise_sentences + more_noise + [question])
            prompt_ids = tokenizer.encode(story).to(self.device)
            prompts.append(prompt_ids)
            answers.append(str(x_val))
        return prompts, answers


# ============================================================
# Reward Oracle
# ============================================================

class PuzzleOracle:
    """
    Reward oracle: parses final generated string and returns 1.0 if the
    last integer matches the ground truth answer, else 0.0.
    """

    def __init__(self, tokenizer: SimpleAsciiTokenizer):
        self.tokenizer = tokenizer

    def _extract_last_int(self, s: str) -> Optional[int]:
        import re
        ints = re.findall(r"-?\d+", s)
        if not ints:
            return None
        try:
            return int(ints[-1])
        except Exception:
            return None

    def score(self, seqs: torch.LongTensor, answers: List[str]) -> torch.Tensor:
        # seqs: [B, T]
        B, _ = seqs.shape
        rewards = torch.zeros(B, dtype=torch.float32, device=seqs.device)
        for i in range(B):
            s = self.tokenizer.decode(seqs[i])
            pred = self._extract_last_int(s)
            try:
                gt = int(answers[i])
            except Exception:
                gt = None
            if pred is not None and gt is not None and pred == gt:
                rewards[i] = 1.0
        return rewards


# ============================================================
# LTM: Simple + FAISS IVF-PQ
# ============================================================

@dataclass
class LTMConfig:
    d_key: int
    d_value: int
    max_size: int = 100_000
    use_faiss: bool = True
    nlist: int = 4096
    m: int = 16
    nprobe: int = 8
    device: str = "cpu"


class SimpleLTM(nn.Module):
    """Baseline cosine-sim memory for fallback / small tests."""

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("keys", torch.empty(0, cfg.d_key))
        self.register_buffer("values", torch.empty(0, cfg.d_value))

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.numel() == 0:
            return
        k = keys.detach()
        v = values.detach()
        self.keys = torch.cat([self.keys, k], dim=0)
        self.values = torch.cat([self.values, v], dim=0)
        if self.keys.shape[0] > self.cfg.max_size:
            self.keys = self.keys[-self.cfg.max_size :]
            self.values = self.values[-self.cfg.max_size :]

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        B, d = queries.shape
        if self.keys.shape[0] == 0:
            return torch.zeros(B, self.cfg.d_value, device=queries.device)
        k = F.normalize(self.keys, dim=-1)
        q = F.normalize(queries, dim=-1)
        sim = torch.matmul(q, k.t())
        k_eff = min(top_k, sim.shape[1])
        vals, idx = torch.topk(sim, k_eff, dim=-1)
        weights = F.softmax(vals, dim=-1)
        chosen = self.values[idx]  # [B, k_eff, d_value]
        out = torch.sum(chosen * weights.unsqueeze(-1), dim=1)
        return out


class FaissIVFPQLTM(nn.Module):
    """
    FAISS IVF-PQ backed LTM.

    Maintains torch buffers for keys/values for correctness and (re)builds
    a FAISS index over keys. Designed for moderate sizes (<= 1e6 entries),
    not for extreme streaming production without further tuning.
    """

    def __init__(self, cfg: LTMConfig):
        super().__init__()
        if not HAS_FAISS:
            raise RuntimeError("FaissIVFPQLTM requires faiss to be installed.")
        self.cfg = cfg
        self.d_key = cfg.d_key
        self.d_value = cfg.d_value
        self.max_size = cfg.max_size
        self.nlist = cfg.nlist
        self.m = cfg.m
        self.nprobe = cfg.nprobe

        self.register_buffer("keys_buf", torch.empty(0, self.d_key))
        self.register_buffer("values_buf", torch.empty(0, self.d_value))

        # Coarse quantizer + IVF-PQ index
        self.quantizer = faiss.IndexFlatL2(self.d_key)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.d_key, self.nlist, self.m, 8)
        self.index.nprobe = self.nprobe
        self.trained = False

    @torch.no_grad()
    def _rebuild_index(self):
        # Rebuild from keys_buf
        self.index.reset()
        if self.keys_buf.shape[0] == 0:
            self.trained = False
            return
        keys_np = self.keys_buf.cpu().numpy().astype("float32")
        if not self.index.is_trained:
            # Use all keys for training
            self.index.train(keys_np)
        self.index.add(keys_np)
        self.trained = True

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.numel() == 0:
            return
        k = keys.detach().to(self.keys_buf.device)
        v = values.detach().to(self.values_buf.device)
        # Append
        self.keys_buf = torch.cat([self.keys_buf, k], dim=0)
        self.values_buf = torch.cat([self.values_buf, v], dim=0)
        # Enforce max_size with sliding window
        if self.keys_buf.shape[0] > self.max_size:
            self.keys_buf = self.keys_buf[-self.max_size :]
            self.values_buf = self.values_buf[-self.max_size :]
        # Rebuild index after changes
        self._rebuild_index()

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        B, d = queries.shape
        if self.keys_buf.shape[0] == 0 or not self.trained or self.index.ntotal == 0:
            return torch.zeros(B, self.d_value, device=queries.device)
        q = queries.detach().to(self.keys_buf.device).cpu().numpy().astype("float32")
        k_eff = min(top_k, self.index.ntotal)
        D, I = self.index.search(q, k_eff)
        # D: [B, k_eff], I: [B, k_eff]
        I_t = torch.from_numpy(I).to(self.values_buf.device)
        D_t = torch.from_numpy(D).to(self.values_buf.device)
        # Convert distances to similarity weights (smaller distance => larger weight)
        weights = F.softmax(-D_t, dim=-1)  # [B, k_eff]
        chosen = self.values_buf[I_t]      # [B, k_eff, d_value]
        out = torch.sum(chosen * weights.unsqueeze(-1), dim=1)
        return out


def build_ltm(cfg: LTMConfig) -> nn.Module:
    if cfg.use_faiss and HAS_FAISS:
        return FaissIVFPQLTM(cfg)
    return SimpleLTM(cfg)


# ============================================================
# Backbone: Mamba2 + fallback Transformer
# ============================================================

class TransformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, mask=attn_mask)


class MambaBackbone(nn.Module):
    """
    Thin wrapper around Mamba2 blocks.
    If mamba-ssm is unavailable, this module should not be instantiated.
    """

    def __init__(self, d_model: int, num_layers: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError("MambaBackbone requires mamba-ssm to be installed.")
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "mamba": Mamba2(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            headdim=d_state,
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attn_mask is ignored: Mamba2 is inherently causal when configured that way
        h = x
        for layer in self.layers:
            residual = h
            h = layer["norm"](h)
            h = layer["mamba"](h)
            h = h + residual
        return h


class InfinityBackbone(nn.Module):
    """
    Unified interface:
      - If Mamba2 available and use_mamba=True -> MambaBackbone
      - Else -> TransformerBackbone
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        use_mamba: bool = True,
        dropout: float = 0.0,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.use_mamba = use_mamba and HAS_MAMBA
        if self.use_mamba:
            self.core = MambaBackbone(
                d_model=d_model,
                num_layers=num_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.core = TransformerBackbone(
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.core, TransformerBackbone):
            return self.core(x, attn_mask)
        return self.core(x, attn_mask=None)


# ============================================================
# ContinuousMemoryInfinityAgent (upgraded)
# ============================================================

class ContinuousMemoryInfinityAgent(nn.Module):
    """
    Infinity agent with:
      - Mamba2 or Transformer backbone
      - FAISS IVF-PQ LTM (fallback to SimpleLTM)
      - Recurrent memory distillation gate for writes
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 4096,
        ltm_cfg: Optional[LTMConfig] = None,
        memory_top_k: int = 8,
        use_mamba: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.memory_top_k = memory_top_k

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = InfinityBackbone(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            use_mamba=use_mamba,
            dropout=dropout,
        )

        if ltm_cfg is None:
            ltm_cfg = LTMConfig(d_key=d_model, d_value=d_model, max_size=100_000, device="cpu")
        self.ltm_cfg = ltm_cfg
        self.ltm = build_ltm(ltm_cfg)

        self.key_proj = nn.Linear(d_model, ltm_cfg.d_key)
        self.val_proj = nn.Linear(d_model, ltm_cfg.d_value)

        fused_dim = d_model + ltm_cfg.d_value
        self.fuse_proj = nn.Linear(fused_dim, d_model)
        self.ln = nn.LayerNorm(d_model)

        # Recurrent Memory Distillation gate: selects which states to commit to LTM
        self.rmd_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # Causal mask only used when backbone is Transformer
        attn_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        h = self.backbone(x, attn_mask=attn_mask)

        # RMD gate for LTM writes (no grad into LTM)
        with torch.no_grad():
            rmd_scores = torch.sigmoid(self.rmd_gate(h))  # [B, T, 1]
            # Select top ~25% timesteps per sequence to commit
            k_commit = max(1, T // 4)
            commit_mask = torch.zeros_like(rmd_scores, dtype=torch.bool)
            topk_vals, topk_idx = torch.topk(rmd_scores.squeeze(-1), k_commit, dim=-1)
            for b in range(B):
                commit_mask[b, topk_idx[b]] = True
            kv = h[commit_mask.squeeze(-1)]  # [N_commit, d_model]
            if kv.numel() > 0:
                keys = self.key_proj(kv)
                vals = self.val_proj(kv)
                self.ltm.add(keys, vals)

        # LTM read from last token state
        q = self.key_proj(h[:, -1])  # [B, d_key]
        v_mem = self.ltm.retrieve(q, top_k=self.memory_top_k)  # [B, d_value]
        v_mem_full = v_mem.unsqueeze(1).expand(B, T, v_mem.shape[-1])
        fused = torch.cat([h, v_mem_full], dim=-1)
        fused = self.fuse_proj(fused)
        fused = self.ln(fused)

        logits = self.head(fused)
        values = self.value_head(fused).squeeze(-1)

        return {"logits": logits, "values": values}

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
        device = seq.device
        for _ in range(max_new_tokens):
            out = self.forward(seq)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
            if seq.shape[1] >= self.max_seq_len:
                break
        return seq


# ============================================================
# RL Harness: GRPO-style
# ============================================================

@dataclass
class GRPOConfig:
    group_size: int = 4
    max_gen_len: int = 32
    temperature: float = 0.8
    top_k: int = 20
    kl_beta: float = 0.01


class GRPOHarness:
    def __init__(
        self,
        agent: ContinuousMemoryInfinityAgent,
        oracle: PuzzleOracle,
        grpo_cfg: GRPOConfig,
    ):
        self.agent = agent
        self.oracle = oracle
        self.cfg = grpo_cfg

    def compute_grpo_loss(
        self,
        prompts: List[torch.LongTensor],
        answers: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = self.agent.head.weight.device
        B = len(prompts)
        # Pad prompts to batch tensor
        max_len = max(p.shape[0] for p in prompts)
        input_ids = torch.full((B, max_len), 0, dtype=torch.long, device=device)
        for i, p in enumerate(prompts):
            input_ids[i, : p.shape[0]] = p.to(device)

        # Generate sequences
        with torch.no_grad():
            seqs = self.agent.generate(
                input_ids,
                max_new_tokens=self.cfg.max_gen_len,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
            )

        out = self.agent(seqs)
        logits = out["logits"]   # [B, T, V]
        values = out["values"].mean(dim=1)  # [B]

        logprobs_all = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs_all.gather(2, seqs.unsqueeze(-1)).squeeze(-1)
        logprob_episode = token_logprobs.sum(dim=1)  # [B]

        rewards = self.oracle.score(seqs, answers)   # [B]

        # Group-wise baselines
        G = self.cfg.group_size
        B_eff = (B // G) * G
        rewards = rewards[:B_eff]
        logprob_episode = logprob_episode[:B_eff]
        values = values[:B_eff]

        rewards_group = rewards.view(-1, G)
        baseline = rewards_group.mean(dim=1, keepdim=True)
        advantages = (rewards_group - baseline).view(-1)

        pg_loss = - (advantages.detach() * logprob_episode).mean()
        value_loss = torch.mean((values - rewards) ** 2)

        kl_proxy = (logprobs_all ** 2).mean()
        loss = pg_loss + value_loss + self.cfg.kl_beta * kl_proxy

        stats = {
            "rl_loss": float(loss.detach().cpu().item()),
            "pg_loss": float(pg_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "kl_proxy": float(kl_proxy.detach().cpu().item()),
            "avg_reward": float(rewards.mean().detach().cpu().item()),
        }
        return loss, stats


# ============================================================
# Trainer
# ============================================================

class InfinityTrainer:
    def __init__(
        self,
        agent: ContinuousMemoryInfinityAgent,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.agent = agent
        self.optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm

    def lm_forward_loss(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = self.agent(input_ids)
        logits = out["logits"]
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        stats = {"lm_loss": float(loss.detach().cpu().item())}
        return loss, stats

    def joint_step_grpo_lm(
        self,
        harness: GRPOHarness,
        rl_prompts: List[torch.LongTensor],
        rl_answers: List[str],
        lm_input_ids: torch.LongTensor,
        lm_labels: torch.LongTensor,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)

        rl_loss, rl_stats = harness.compute_grpo_loss(rl_prompts, rl_answers)
        lm_loss, lm_stats = self.lm_forward_loss(lm_input_ids, lm_labels)

        total_loss = alpha * lm_loss + (1.0 - alpha) * rl_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats = {"total_loss": float(total_loss.detach().cpu().item())}
        stats.update(rl_stats)
        stats.update(lm_stats)
        return stats


# ============================================================
# Manual training demo
# ============================================================

def build_lm_batch_from_env(
    tokenizer: SimpleAsciiTokenizer,
    env_prompts: List[torch.LongTensor],
    env_answers: List[str],
    device: str,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Build LM inputs by concatenating prompt + answer, predicting all tokens."""
    B = len(env_prompts)
    concat_seqs = []
    for i in range(B):
        ans_ids = tokenizer.encode(env_answers[i])
        seq = torch.cat([env_prompts[i], ans_ids.to(env_prompts[i].device)], dim=0)
        concat_seqs.append(seq)
    max_len = max(s.shape[0] for s in concat_seqs)
    input_ids = torch.full((B, max_len), 0, dtype=torch.long, device=device)
    labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for i, s in enumerate(concat_seqs):
        L = s.shape[0]
        input_ids[i, :L] = s.to(device)
        labels[i, 1:L] = s[:-1].to(device)
    return input_ids, labels


def manual_train_demo(
    steps: int = 200,
    batch_size: int = 16,
    use_long_env: bool = True,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = SimpleAsciiTokenizer()
    if use_long_env:
        env = LongHorizonMemoryEnv(device=device)
    else:
        env = LogicPuzzleEnv(device=device)
    oracle = PuzzleOracle(tokenizer)

    d_model = 256
    ltm_cfg = LTMConfig(
        d_key=d_model,
        d_value=d_model,
        max_size=200_000,
        use_faiss=True,
        nlist=4096,
        m=16,
        nprobe=8,
        device=device,
    )

    agent = ContinuousMemoryInfinityAgent(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=4,
        num_layers=4,
        max_seq_len=1024,
        ltm_cfg=ltm_cfg,
        memory_top_k=16,
        use_mamba=True,
        dropout=0.1,
    ).to(device)

    trainer = InfinityTrainer(agent, lr=3e-4, weight_decay=0.0, max_grad_norm=1.0)
    grpo_cfg = GRPOConfig(group_size=4, max_gen_len=32, temperature=0.8, top_k=32, kl_beta=0.01)
    harness = GRPOHarness(agent, oracle, grpo_cfg)

    for step in range(1, steps + 1):
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
            alpha=0.5,
        )
        if step % 10 == 0:
            print(
                f"[step {step}] total={stats['total_loss']:.4f} "
                f"rl={stats['rl_loss']:.4f} lm={stats['lm_loss']:.4f} "
                f"avg_reward={stats['avg_reward']:.3f}"
            )


if __name__ == '__main__':
    steps = int(os.environ.get("INFINITY_DEMO_STEPS", "50"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manual_train_demo(steps=steps, batch_size=8, use_long_env=True, device=device)
