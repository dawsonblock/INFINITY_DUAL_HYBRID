"""
config.py

Unified configuration for Infinity V3 Dual Hybrid.
All hyperparameters in one place for easy tuning and experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MirasConfig:
    """Dual-Tier Miras parametric memory configuration."""

    d_model: int = 256

    # Fast tier (SGD-based, rapid adaptation)
    fast_rank: int = 32
    fast_lr: float = 1e-3
    fast_l2_reg: float = 1e-4
    fast_init_scale: float = 0.1

    # Deep tier (Titans-style with momentum, Huber, retention gate)
    deep_rank: int = 32
    deep_lr: float = 5e-4
    deep_l2_reg: float = 1e-4
    deep_momentum: float = 0.9
    deep_use_huber: bool = True
    deep_huber_delta: float = 1.0
    deep_init_scale: float = 0.1

    # Mixing
    init_fast_weight: float = 0.7  # Initial blend: 70% fast, 30% deep
    context_gate: bool = True       # Use context-dependent gating

    # v2.0: Stability upgrades
    grad_clip: float = 1.0          # Per-tier gradient clipping
    norm_reg: float = 0.0           # Norm regularization coefficient
    use_ema: bool = False           # EMA smoothing of updates
    ema_decay: float = 0.99         # EMA decay rate


@dataclass
class LTMConfig:
    """Long-Term Memory (episodic) configuration."""

    d_key: int = 256
    d_value: int = 256
    max_size: int = 100_000

    # FAISS settings
    use_faiss: bool = True
    nlist: int = 1024          # Number of IVF clusters
    m: int = 16                # PQ subquantizers
    nprobe: int = 8            # Clusters to search

    # Retrieval
    top_k: int = 8             # Top-k neighbors to retrieve

    # Async writer
    use_async_writer: bool = False
    write_batch_size: int = 64

    # Storage policy
    store_on_episode_end: bool = True
    store_high_reward_threshold: Optional[float] = None


@dataclass
class BackboneConfig:
    """Hybrid SSM + Attention backbone configuration."""

    d_model: int = 256

    # SSM (Mamba2) settings
    use_mamba: bool = True
    require_mamba2: bool = False
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    num_mamba_layers: int = 2

    # Attention settings
    use_attention: bool = True
    num_attention_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1

    # Hybrid order: 'mamba_first' or 'attention_first'
    hybrid_order: str = 'mamba_first'

    # v2.0: Backbone upgrades
    use_residual: bool = True         # Residual connections
    use_drop_path: bool = False       # Stochastic depth
    drop_path_rate: float = 0.1       # Drop path probability
    pre_norm: bool = True             # Pre-LayerNorm vs Post-LayerNorm


@dataclass
class AgentConfig:
    """Unified agent configuration."""

    obs_dim: int = 4           # Observation dimension (e.g., CartPole=4)
    act_dim: int = 2           # Action dimension (e.g., CartPole=2)
    hidden_dim: int = 256      # Internal model dimension

    # Sub-configs
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    miras: MirasConfig = field(default_factory=MirasConfig)
    ltm: LTMConfig = field(default_factory=LTMConfig)

    # Memory fusion
    use_ltm_in_forward: bool = True
    use_miras_in_forward: bool = True

    # RMD gate for LTM writes (selects top-k% salient states)
    rmd_commit_ratio: float = 0.25

    # State management
    reset_mamba_state_on_episode: bool = True
    reset_miras_on_episode: bool = False

    # v2.0: Agent mode and diagnostics
    mode: str = "train"  # "train", "eval", "inference"
    temperature: float = 1.0  # Policy temperature for discrete actions

    log_mamba_backend: bool = True

    miras_weight_mode: str = "abs_adv"  # ["abs_adv", "pos_adv", "none"]
    write_gate_floor: float = 0.0
    write_gate_ceiling: float = 1.0

    def __post_init__(self):
        self.sync_dims()

    def sync_dims(self):
        """Ensure all sub-config dimensions match hidden_dim."""
        self.backbone.d_model = self.hidden_dim
        self.miras.d_model = self.hidden_dim
        self.ltm.d_key = self.hidden_dim
        self.ltm.d_value = self.hidden_dim


@dataclass
class PPOConfig:
    """PPO / GRPO trainer configuration."""

    # Core PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2

    # Learning
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # Loss weights
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Optional KL penalty
    use_kl_penalty: bool = False
    kl_target: float = 0.01
    kl_coef: float = 0.2

    # Training schedule
    train_epochs: int = 10
    batch_size: int = 64
    steps_per_rollout: int = 2048
    num_envs: int = 1

    # Total training
    max_iterations: int = 100

    # Evaluation
    eval_episodes: int = 5
    eval_interval: int = 1

    # v2.0: PPO upgrades
    normalize_gae: bool = True        # Normalize advantages
    adaptive_kl: bool = False         # Adaptive KL target tuning
    kl_adapt_coef: float = 1.5        # KL adaptation multiplier
    grad_explosion_threshold: float = 100.0  # Grad norm threshold
    lr_reduce_factor: float = 0.5     # LR reduction on explosion
    track_grad_norm: bool = True      # Log gradient norms

    mem_gate_coef: float = 0.01
    adv_norm: bool = True
    adv_clip: float = 5.0
    adv_positive_only: bool = True


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Environment
    env_id: str = "CartPole-v1"

    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"

    # Sub-configs
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Logging
    log_interval: int = 1
    save_interval: int = 10
    save_path: str = "checkpoints"

    env_register_local: bool = True

    delayedcue_episode_len: int = 2000
    delayedcue_cue_time: int = 50
    delayedcue_delay: int = 1000
    delayedcue_window: int = 25
    delayedcue_noise_std: float = 0.1
    delayedcue_step_penalty: float = -0.001

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_default_config() -> TrainConfig:
    """Returns default configuration for CartPole."""
    return TrainConfig()


def get_config_for_env(env_id: str) -> TrainConfig:
    """Returns configuration tuned for specific environment."""
    cfg = TrainConfig(env_id=env_id)

    if "CartPole" in env_id:
        cfg.agent.obs_dim = 4
        cfg.agent.act_dim = 2
        cfg.agent.hidden_dim = 128
        cfg.ppo.steps_per_rollout = 2048
        cfg.ppo.max_iterations = 50

    elif "LunarLander" in env_id:
        cfg.agent.obs_dim = 8
        cfg.agent.act_dim = 4
        cfg.agent.hidden_dim = 256
        cfg.ppo.steps_per_rollout = 4096
        cfg.ppo.max_iterations = 200

    elif "Pendulum" in env_id:
        cfg.agent.obs_dim = 3
        cfg.agent.act_dim = 1  # Continuous - will need different head
        cfg.agent.hidden_dim = 128

    elif env_id in ("DelayedCue-v0", "DelayedCueRegime-v0"):
        cfg.agent.obs_dim = 5
        cfg.agent.act_dim = 2
        cfg.agent.hidden_dim = 256
        cfg.ppo.steps_per_rollout = 4096
        cfg.ppo.gamma = 0.999
        cfg.ppo.gae_lambda = 0.97

        cfg.delayedcue_episode_len = 2000
        cfg.delayedcue_cue_time = 50
        cfg.delayedcue_delay = 1000
        cfg.delayedcue_window = 25
        cfg.delayedcue_noise_std = 0.1
        cfg.delayedcue_step_penalty = -0.001

    return cfg
