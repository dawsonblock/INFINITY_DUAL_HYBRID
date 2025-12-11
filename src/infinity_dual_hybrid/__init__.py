"""
Infinity Dual Hybrid - Lean Build

A production-grade RL architecture combining:
- Hybrid SSM/Attention backbone (Mamba2 + Transformer)
- Dual-Tier Miras parametric memory
- FAISS IVF-PQ episodic LTM
- PPO trainer with GAE
"""

__version__ = "1.0.0"

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

from .ltm import build_ltm, SimpleLTM

from .ssm_backbone import HybridSSMAttentionBackbone, ObservationEncoder

from .envs import make_envs, make_env, get_env_info


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
    # Backbone
    "HybridSSMAttentionBackbone",
    "ObservationEncoder",
    # Envs
    "make_envs",
    "make_env",
    "get_env_info",
]
