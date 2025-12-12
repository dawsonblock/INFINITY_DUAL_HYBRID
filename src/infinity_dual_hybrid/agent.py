"""
agent.py

Unified InfinityV3DualHybridAgent.

This is the canonical agent combining:
- Hybrid SSM/Attention backbone for encoding
- Dual-Tier Miras for parametric working memory
- FAISS LTM for episodic long-term memory
- Policy and value heads for RL

Data Flow:
    obs [B, obs_dim]
        |
        v
    ObservationEncoder -> [B, d_model]
        |
        v
    HybridSSMAttentionBackbone -> encoded [B, d_model]
        |
        +---> Miras.read() -> miras_v [B, d_model]
        +---> LTM.retrieve() -> ltm_v [B, d_model]
        |
        v
    MemoryFusion [encoded, miras_v, ltm_v] -> fused [B, d_model]
        |
        +---> PolicyHead -> logits [B, act_dim]
        +---> ValueHead -> value [B, 1]

Memory Updates (during training):
    - Miras.update() weighted by |advantage|
    - LTM.store() for high-importance states (RMD-gated or episode-end)
"""

from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn

from .config import AgentConfig
from .miras import DualTierMiras
from .ltm import build_ltm, LTMWrapper
from .ssm_backbone import (
    HybridSSMAttentionBackbone,
    ObservationEncoder,
)


class MemoryFusion(nn.Module):
    """
    Fuses backbone output with Miras and LTM retrievals.

    Concatenates [encoded, miras_v, ltm_v] and projects to d_model.
    """

    def __init__(
        self,
        d_model: int,
        use_miras: bool = True,
        use_ltm: bool = True,
    ):
        super().__init__()
        self.use_miras = use_miras
        self.use_ltm = use_ltm

        # Compute input dimension
        num_sources = 1  # backbone
        if use_miras:
            num_sources += 1
        if use_ltm:
            num_sources += 1

        self.proj = nn.Linear(d_model * num_sources, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        encoded: torch.Tensor,
        miras_v: Optional[torch.Tensor] = None,
        ltm_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoded: [B, d_model] backbone output
            miras_v: [B, d_model] Miras retrieval (optional)
            ltm_v: [B, d_model] LTM retrieval (optional)
        Returns:
            [B, d_model] fused representation
        """
        parts = [encoded]
        if self.use_miras and miras_v is not None:
            parts.append(miras_v)
        if self.use_ltm and ltm_v is not None:
            parts.append(ltm_v)

        fused = torch.cat(parts, dim=-1)
        fused = self.proj(fused)
        fused = self.norm(fused)
        return fused


class RMDGate(nn.Module):
    """
    Recurrent Memory Distillation gate for selective LTM writes.

    Computes importance scores for states to determine which
    should be committed to long-term memory.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model]
        Returns:
            [B, 1] importance scores (sigmoid applied)
        """
        return torch.sigmoid(self.gate(x))


class InfinityV3DualHybridAgent(nn.Module):
    """
    Unified Infinity V3 Dual Hybrid Agent.

    Combines:
    - Hybrid SSM/Attention backbone
    - Dual-Tier Miras parametric memory
    - FAISS-backed episodic LTM
    - Policy and value heads for PPO

    This is the canonical agent for the Infinity V3 system.
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            obs_dim=cfg.obs_dim,
            d_model=d,
            is_sequential=False,
        )

        # Backbone
        self.backbone = HybridSSMAttentionBackbone(cfg.backbone)

        if cfg.log_mamba_backend:
            backend = (
                "Mamba2" if len(self.backbone.mamba_layers) > 0 else "fallback"
            )
            print(f"SSM backend: {backend}")

        # Dual-Tier Miras
        if cfg.use_miras_in_forward:
            self.miras = DualTierMiras.from_config(cfg.miras)
            self.miras_key_proj = nn.Linear(d, d)
            self.miras_val_proj = nn.Linear(d, d)
        else:
            self.miras = None
            self.miras_key_proj = None
            self.miras_val_proj = None

        # LTM
        if cfg.use_ltm_in_forward:
            self.ltm: Optional[LTMWrapper] = build_ltm(cfg.ltm)
            self.ltm_key_proj = nn.Linear(d, d)
            self.ltm_val_proj = nn.Linear(d, d)
            self.rmd_gate = RMDGate(d)
        else:
            self.ltm = None
            self.ltm_key_proj = None
            self.ltm_val_proj = None
            self.rmd_gate = None

        # Memory fusion
        self.fusion = MemoryFusion(
            d_model=d,
            use_miras=cfg.use_miras_in_forward,
            use_ltm=cfg.use_ltm_in_forward,
        )

        # Policy head (discrete actions)
        self.policy_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.act_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self.memory_write_gate = nn.Sequential(
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )

        # Buffer for episode tracking
        self._episode_states: list = []

        # v2.0: Runtime state
        self._mode = cfg.mode  # "train", "eval", "inference"
        self._temperature = cfg.temperature

    def set_mode(self, mode: str) -> None:
        """Set agent mode: 'train', 'eval', or 'inference'."""
        assert mode in ("train", "eval", "inference")
        self._mode = mode

    def set_temperature(self, t: float) -> None:
        """Set policy temperature for action sampling."""
        self._temperature = t

    def debug_state(self) -> Dict[str, Any]:
        """
        Return diagnostic state for debugging and logging.

        Returns:
            Dict with backbone, Miras, and LTM state info
        """
        state = {
            "mode": self._mode,
            "temperature": self._temperature,
            "episode_buffer_size": len(self._episode_states),
        }

        # Backbone info
        state["backbone"] = {
            "has_mamba": len(self.backbone.mamba_layers) > 0,
            "has_attention": len(self.backbone.attention_layers) > 0,
            "d_model": self.backbone.d_model,
        }

        # Miras info
        if self.miras is not None:
            miras_stats = self.miras.get_stats()
            state["miras"] = {
                "fast_B_norm": miras_stats.get("fast_B_norm", 0),
                "fast_C_norm": miras_stats.get("fast_C_norm", 0),
                "deep_B_norm": miras_stats.get("deep_B_norm", 0),
                "deep_C_norm": miras_stats.get("deep_C_norm", 0),
                "mix_ratio": miras_stats.get("mix_ratio", 0),
            }
        else:
            state["miras"] = None

        # LTM info
        if self.ltm is not None:
            state["ltm"] = {
                "size": self.ltm.size,
            }
        else:
            state["ltm"] = None

        return state

    def forward(
        self,
        obs: torch.Tensor,
        advantage: Optional[torch.Tensor] = None,
        store_for_ltm: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the agent.

        Args:
            obs: [B, obs_dim] observations
            advantage: [B] advantages for Miras weighting (optional)
            store_for_ltm: If True, buffer states for LTM storage
        Returns:
            Dict with 'logits', 'value', 'encoded'
        """
        # Encode observations
        x = self.obs_encoder(obs)  # [B, d_model]

        # Pass through backbone
        encoded = self.backbone(x)  # [B, d_model]

        write_prob = self.memory_write_gate(encoded)
        gate = torch.clamp(
            write_prob,
            min=float(self.cfg.write_gate_floor),
            max=float(self.cfg.write_gate_ceiling),
        )

        effective_write = None
        if advantage is not None:
            mode = self.cfg.miras_weight_mode
            if mode == "abs_adv":
                miras_w = advantage.abs()
            elif mode == "pos_adv":
                miras_w = torch.clamp(advantage, min=0.0)
            else:
                miras_w = torch.ones_like(advantage)

            effective_write = gate.squeeze(-1) * miras_w

        if store_for_ltm and self.ltm is not None and self._mode == "train":
            self._episode_states.append(encoded.detach().cpu())

        # Miras read
        miras_v = None
        if self.miras is not None:
            miras_k = self.miras_key_proj(encoded)
            read_out = self.miras.read(miras_k, context=encoded)
            miras_v = read_out["v"]

        # LTM read
        ltm_v = None
        if self.ltm is not None:
            ltm_q = self.ltm_key_proj(encoded)
            ltm_v = self.ltm.retrieve(ltm_q, top_k=self.cfg.ltm.top_k)

        # Memory fusion
        fused = self.fusion(encoded, miras_v, ltm_v)

        # Policy and value heads
        logits = self.policy_head(fused)
        value = self.value_head(fused).squeeze(-1)

        # Memory updates only in train mode
        if self.training and self._mode == "train":
            self._update_memories(
                encoded=encoded,
                advantage=advantage,
                write_strength=effective_write,
                store_for_ltm=store_for_ltm,
            )

        return {
            "logits": logits,
            "value": value,
            "encoded": encoded,
            "fused": fused,
            "write_prob": gate,
            "effective_write_mean": (
                effective_write.mean()
                if effective_write is not None
                else torch.tensor(0.0, device=encoded.device)
            ),
        }

    @torch.no_grad()
    def _update_memories(
        self,
        encoded: torch.Tensor,
        advantage: Optional[torch.Tensor],
        write_strength: Optional[torch.Tensor],
        store_for_ltm: bool,
    ) -> None:
        """Update Miras and optionally buffer states for LTM."""
        # Miras update with advantage weighting
        if self.miras is not None:
            miras_k = self.miras_key_proj(encoded)
            miras_target = self.miras_val_proj(encoded).detach()

            weight = None
            if write_strength is not None:
                weight = write_strength.detach()

            self.miras.update(
                miras_k,
                miras_target,
                weight=weight,
                context=encoded,
            )

    def commit_to_ltm(
        self,
        rewards: Optional[torch.Tensor] = None,
        force: bool = False,
    ) -> None:
        """
        Commit buffered states to LTM.

        Called at episode boundaries or when force=True.
        Uses RMD gate to select top-k% important states.
        """
        if not force and self._mode != "train":
            return

        if self.ltm is None or not self._episode_states:
            return

        # Concatenate buffered states
        states = torch.cat(self._episode_states, dim=0)  # [T, d_model]
        device = next(self.parameters()).device
        states = states.to(device)

        if states.shape[0] == 0:
            self._episode_states = []
            return

        # Compute importance scores with RMD gate
        if self.rmd_gate is not None and not force:
            with torch.no_grad():
                scores = self.rmd_gate(states).squeeze(-1)  # [T]
                k = max(1, int(states.shape[0] * self.cfg.rmd_commit_ratio))
                _, topk_idx = torch.topk(scores, k)
                states = states[topk_idx]

        # Commit to LTM
        if states.shape[0] > 0:
            keys = self.ltm_key_proj(states)
            values = self.ltm_val_proj(states)
            self.ltm.store(keys, values)

        # Clear buffer
        self._episode_states = []

    def reset_episode(self) -> None:
        """Reset episode-level state."""
        self._episode_states = []
        self.backbone.reset_state()

        if self.cfg.reset_miras_on_episode and self.miras is not None:
            self.miras.reset_state()

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        store_for_ltm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.

        Args:
            obs: [B, obs_dim] observations
            deterministic: If True, return argmax action
        Returns:
            (action, log_prob, value)
        """
        out = self.forward(obs, store_for_ltm=store_for_ltm)
        logits = out["logits"]
        value = out["value"]

        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.zeros_like(value)
        else:
            # Apply temperature scaling
            scaled_logits = logits / max(self._temperature, 1e-8)
            dist = torch.distributions.Categorical(logits=scaled_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantage: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: [B, obs_dim]
            actions: [B]
            advantage: [B] for Miras weighting
        Returns:
            (log_prob, value, entropy)
        """
        out = self.forward(obs, advantage=advantage)
        logits = out["logits"]
        value = out["value"]
        write_prob = out["write_prob"].squeeze(-1)
        effective_write_mean = out["effective_write_mean"]

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy, write_prob, effective_write_mean

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> "InfinityV3DualHybridAgent":
        """Create agent from config."""
        return cls(cfg)

    def save(self, path: str) -> None:
        """Save agent state."""
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.cfg,
        }
        if self.ltm is not None:
            state["ltm_state"] = self.ltm.state_dict_ltm()
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
    ) -> "InfinityV3DualHybridAgent":
        """Load agent from checkpoint."""
        state = torch.load(path, map_location=device)
        agent = cls(state["config"])
        agent.load_state_dict(state["model_state_dict"])
        if "ltm_state" in state and agent.ltm is not None:
            agent.ltm.load_state_dict_ltm(state["ltm_state"])
        return agent.to(device)

    def shutdown(self) -> None:
        """Clean shutdown (stop async LTM writer if active)."""
        if self.ltm is not None:
            self.ltm.shutdown()


def build_agent(cfg: AgentConfig) -> InfinityV3DualHybridAgent:
    """Build agent from config."""
    return InfinityV3DualHybridAgent(cfg)
