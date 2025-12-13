"""
ssm_backbone.py

Hybrid SSM + Attention Backbone for Infinity V3.

Architecture options:
1. Mamba2 layers followed by Attention layers (mamba_first)
2. Attention layers followed by Mamba2 layers (attention_first)
3. Pure Mamba2 (if attention disabled)
4. Pure Attention/MLP fallback (if Mamba unavailable)

The hybrid approach combines:
- Mamba2: O(N) linear-time sequence modeling with state-space recurrence
- Attention: Global context aggregation with learned attention patterns

For RL, we also provide a simple MLP backbone for non-sequential observations.
"""

from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

from .config import BackboneConfig

# Optional Mamba2 import - vendored (vendor/mamba_ssm) or system-installed
HAS_MAMBA = False
Mamba2 = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    HAS_MAMBA = True
except ImportError:
    Mamba2 = None
    HAS_MAMBA = False


def detect_ssm_backend(cfg: BackboneConfig) -> str:
    use_mamba = bool(getattr(cfg, "use_mamba", False))
    if use_mamba and HAS_MAMBA:
        return "mamba2"
    return "fallback"


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) for regularization.

    Randomly drops entire residual branches during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLPBackbone(nn.Module):
    """
    Simple MLP backbone for non-sequential RL tasks.

    Used when observations are single vectors (e.g., CartPole).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] or [B, T, input_dim]
        Returns:
            [B, hidden_dim] or [B, T, hidden_dim]
        """
        return self.net(x)

    def reset_state(self) -> None:
        """No state to reset for MLP."""
        pass


class Mamba2Block(nn.Module):
    """
    Single Mamba2 block with pre-norm and residual connection.

    Structure: x -> LayerNorm -> Mamba2 -> + x
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError("mamba-ssm not available")

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=d_state,
        )
        self._inference_state = None

    def forward(
        self,
        x: torch.Tensor,
        inference_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Args:
            x: [B, T, d_model]
            inference_state: Optional state from previous step
        Returns:
            (output, new_state)
        """
        residual = x
        x = self.norm(x)

        # Mamba2 forward (handles state internally)
        if inference_state is not None and hasattr(self.mamba, 'step'):
            # Step mode for inference
            x, new_state = self.mamba.step(x, inference_state)
        else:
            x = self.mamba(x)
            new_state = None

        return residual + x, new_state

    def reset_state(self) -> None:
        """Reset inference state."""
        self._inference_state = None


class AttentionBlock(nn.Module):
    """
    Standard Transformer attention block with pre-norm and optional DropPath.

    Structure: x -> LayerNorm -> MultiHeadAttention -> DropPath -> + x
                 -> LayerNorm -> FFN -> DropPath -> + x
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            attn_mask: Optional [T, T] causal mask
        Returns:
            [B, T, d_model]
        """
        # Self-attention with residual + drop_path
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + self.drop_path1(x)

        # FFN with residual + drop_path
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.drop_path2(x)

        return x


class HybridSSMAttentionBackbone(nn.Module):
    """
    Hybrid backbone combining Mamba2 (SSM) and Attention layers.

    The hybrid order can be configured:
    - 'mamba_first': SSM layers process sequence first, then attention
    - 'attention_first': Attention layers first, then SSM

    Falls back to pure attention/MLP if Mamba is unavailable.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        require_mamba2 = bool(getattr(cfg, "require_mamba2", False))
        if require_mamba2 and not bool(getattr(cfg, "use_mamba", False)):
            raise ValueError(
                "require_mamba2=True requires use_mamba=True. "
                "Set BackboneConfig.use_mamba=True or disable require_mamba2."
            )
        if require_mamba2 and not HAS_MAMBA:
            raise RuntimeError(
                "require_mamba2=True but Mamba2 is not available. "
                "Install mamba-ssm and its Triton/CUDA dependencies, "
                "or set require_mamba2=False to allow fallback."
            )

        # Determine what's available
        use_mamba = cfg.use_mamba and HAS_MAMBA
        use_attention = cfg.use_attention

        # Build Mamba layers
        self.mamba_layers = nn.ModuleList()
        if use_mamba:
            for _ in range(cfg.num_mamba_layers):
                self.mamba_layers.append(Mamba2Block(
                    d_model=cfg.d_model,
                    d_state=cfg.mamba_d_state,
                    d_conv=cfg.mamba_d_conv,
                    expand=cfg.mamba_expand,
                ))

        # Build Attention layers with optional DropPath
        self.attention_layers = nn.ModuleList()
        if use_attention:
            # Stochastic depth: linearly increase drop rate
            drop_rate = cfg.drop_path_rate if cfg.use_drop_path else 0.0
            for i in range(cfg.num_attention_layers):
                layer_drop = drop_rate * i / max(cfg.num_attention_layers - 1, 1)
                self.attention_layers.append(AttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    drop_path=layer_drop,
                ))

        # If neither available, use MLP fallback
        if not self.mamba_layers and not self.attention_layers:
            self.fallback_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
                nn.GELU(),
            )
        else:
            self.fallback_mlp = None

        self.hybrid_order = cfg.hybrid_order
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] or [B, d_model] for single-step
            attn_mask: Optional attention mask [T, T]
        Returns:
            [B, T, d_model] or [B, d_model]
        """
        # Handle single vector input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, d_model]
            squeeze_output = True

        # Fallback path
        if self.fallback_mlp is not None:
            x = self.fallback_mlp(x)
            x = self.output_norm(x)
            if squeeze_output:
                x = x.squeeze(1)
            return x

        # Hybrid path
        if self.hybrid_order == 'mamba_first':
            # Mamba layers first
            for layer in self.mamba_layers:
                x, _ = layer(x)
            # Then attention
            for layer in self.attention_layers:
                x = layer(x, attn_mask=attn_mask)
        else:
            # Attention first
            for layer in self.attention_layers:
                x = layer(x, attn_mask=attn_mask)
            # Then Mamba
            for layer in self.mamba_layers:
                x, _ = layer(x)

        x = self.output_norm(x)

        if squeeze_output:
            x = x.squeeze(1)

        return x

    def reset_state(self) -> None:
        """Reset all stateful layers (Mamba inference states)."""
        for layer in self.mamba_layers:
            layer.reset_state()

    @classmethod
    def from_config(cls, cfg: BackboneConfig) -> "HybridSSMAttentionBackbone":
        return cls(cfg)


class ObservationEncoder(nn.Module):
    """
    Encodes raw observations into d_model embeddings.

    For vector observations: Linear projection
    For sequential observations: Embedding + positional encoding
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        max_seq_len: int = 1024,
        is_sequential: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.is_sequential = is_sequential

        if is_sequential:
            # For token/discrete observations
            self.embed = nn.Embedding(obs_dim, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
        else:
            # For continuous vector observations
            self.proj = nn.Sequential(
                nn.Linear(obs_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim] or [B, T] for sequential
        Returns:
            [B, d_model] or [B, T, d_model]
        """
        if self.is_sequential:
            B, T = obs.shape
            device = obs.device
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            return self.embed(obs) + self.pos_embed(pos)
        else:
            return self.proj(obs)


def build_backbone(cfg: BackboneConfig) -> HybridSSMAttentionBackbone:
    """Build backbone from config."""
    return HybridSSMAttentionBackbone(cfg)


def check_mamba_available() -> bool:
    """Check if Mamba2 is available."""
    return HAS_MAMBA
