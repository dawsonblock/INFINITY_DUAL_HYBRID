"""
Infinity Dual Hybrid v2.0

A production-grade, research-ready RL framework combining:
- Hybrid SSM/Attention backbone (Mamba2 + Transformer) with residual + DropPath
- Dual-Tier Miras parametric memory with gradient clipping + EMA
- FAISS IVF-PQ episodic LTM with automatic commit engine
- PPO trainer with adaptive KL, GAE normalization, gradient protection
- Unified logging (CSV, JSONL, TensorBoard)
"""

__version__ = "2.0.0"

from .config import (
    TrainConfig,
    AgentConfig,
    PPOConfig,
    MirasConfig,
    LTMConfig,
    BackboneConfig,
    get_config_for_env,
)

from .agent import InfinityV3DualHybridAgent

from .ppo_trainer import PPOTrainer

from .miras import (
    DualTierMiras,
    SSMCompressedMiras,
    SSMCompressedMirasTitans,
)

from .ltm import build_ltm, SimpleLTM, LTMWrapper

from .ssm_backbone import HybridSSMAttentionBackbone, ObservationEncoder

from .envs import make_envs, make_env, get_env_info

from .commit_engine import LTMCommitEngine, CommitConfig, CommitMode

from .logger import UnifiedLogger, LoggerConfig, create_logger


def build_agent(cfg: AgentConfig) -> InfinityV3DualHybridAgent:
    """Build an agent from config."""
    return InfinityV3DualHybridAgent(cfg)


def get_default_config() -> TrainConfig:
    """Get default training config."""
    return TrainConfig()


__all__ = [
    # Config
    "TrainConfig",
    "AgentConfig",
    "PPOConfig",
    "MirasConfig",
    "LTMConfig",
    "BackboneConfig",
    "get_config_for_env",
    "get_default_config",
    # Agent
    "InfinityV3DualHybridAgent",
    "build_agent",
    # Trainer
    "PPOTrainer",
    # Memory
    "DualTierMiras",
    "SSMCompressedMiras",
    "SSMCompressedMirasTitans",
    "build_ltm",
    "SimpleLTM",
    "LTMWrapper",
    # Backbone
    "HybridSSMAttentionBackbone",
    "ObservationEncoder",
    # Envs
    "make_envs",
    "make_env",
    "get_env_info",
    # v2.0: Commit Engine
    "LTMCommitEngine",
    "CommitConfig",
    "CommitMode",
    # v2.0: Logger
    "UnifiedLogger",
    "LoggerConfig",
    "create_logger",
]
