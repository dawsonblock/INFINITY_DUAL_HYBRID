"""
commit_engine.py

Automatic LTM Commit Engine for Infinity Dual Hybrid v2.0.

Handles intelligent memory commitment to Long-Term Memory:
- Episode boundary commits
- Periodic forced commits
- Advantage/reward-weighted commit paths
- Sliding window commits
"""

from typing import Optional, Dict, List, NamedTuple
from enum import Enum
from dataclasses import dataclass, field

import torch
import torch.nn as nn


class CommitMode(Enum):
    """LTM commit trigger modes."""
    EPISODE_END = "episode_end"
    PERIODIC = "periodic"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


class CommitWeighting(Enum):
    """How to weight states for LTM commitment."""
    UNIFORM = "uniform"
    ADVANTAGE = "advantage"
    REWARD = "reward"
    TD_ERROR = "td_error"


@dataclass
class CommitConfig:
    """Configuration for LTM commit engine."""
    mode: CommitMode = CommitMode.EPISODE_END
    weighting: CommitWeighting = CommitWeighting.ADVANTAGE
    
    # Periodic commit settings
    commit_interval: int = 100  # Steps between forced commits
    
    # Sliding window settings
    window_size: int = 64
    window_stride: int = 32
    
    # Filtering thresholds
    min_advantage_threshold: float = 0.0  # Only commit if |adv| > threshold
    min_reward_threshold: Optional[float] = None
    
    # Capacity limits
    max_commit_batch: int = 256
    
    # RMD gating
    use_rmd_gating: bool = True
    rmd_threshold: float = 0.5


class PendingCommit(NamedTuple):
    """A state pending commitment to LTM."""
    key: torch.Tensor
    value: torch.Tensor
    weight: float
    step: int


class LTMCommitEngine(nn.Module):
    """
    Automatic LTM Commit Engine.
    
    Manages when and how states are committed to Long-Term Memory,
    supporting multiple commit strategies and intelligent filtering.
    """
    
    def __init__(self, cfg: CommitConfig, ltm: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.ltm = ltm
        
        # Buffers for pending commits
        self.pending_keys: List[torch.Tensor] = []
        self.pending_values: List[torch.Tensor] = []
        self.pending_weights: List[float] = []
        self.pending_steps: List[int] = []
        
        # Sliding window buffer
        self.window_keys: List[torch.Tensor] = []
        self.window_values: List[torch.Tensor] = []
        self.window_weights: List[float] = []
        
        # Episode accumulator for RMD gating
        self.episode_rewards: List[float] = []
        self.episode_states: List[torch.Tensor] = []
        
        # Stats
        self.total_commits = 0
        self.total_states_committed = 0
        self.step_counter = 0
        self.last_commit_step = 0
        self.last_commit_score = 0.0
    
    def add_state(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        advantage: Optional[float] = None,
        reward: Optional[float] = None,
        td_error: Optional[float] = None,
        rmd_gate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Add a state to the pending commit buffer.
        
        Args:
            key: State encoding [d_key]
            value: Value to store [d_value]
            advantage: Advantage estimate for weighting
            reward: Reward for weighting
            td_error: TD error for weighting
            rmd_gate: RMD gate score (0-1)
        
        Returns:
            Stats dict
        """
        self.step_counter += 1
        stats: Dict[str, float] = {}
        
        # Compute weight based on config
        weight = self._compute_weight(advantage, reward, td_error)
        
        # RMD gating check
        if self.cfg.use_rmd_gating and rmd_gate is not None:
            if rmd_gate < self.cfg.rmd_threshold:
                stats["commit/filtered_by_rmd"] = 1.0
                return stats
        
        # Threshold filtering
        if self.cfg.min_advantage_threshold > 0 and advantage is not None:
            if abs(advantage) < self.cfg.min_advantage_threshold:
                stats["commit/filtered_by_advantage"] = 1.0
                return stats
        
        if self.cfg.min_reward_threshold is not None and reward is not None:
            if reward < self.cfg.min_reward_threshold:
                stats["commit/filtered_by_reward"] = 1.0
                return stats
        
        # Add to appropriate buffer based on mode
        if self.cfg.mode == CommitMode.SLIDING_WINDOW:
            self._add_to_window(key, value, weight)
        else:
            self.pending_keys.append(key.detach())
            self.pending_values.append(value.detach())
            self.pending_weights.append(weight)
            self.pending_steps.append(self.step_counter)
        
        # Track for episode accumulation
        if reward is not None:
            self.episode_rewards.append(reward)
        self.episode_states.append(key.detach())
        
        stats["commit/pending_size"] = float(len(self.pending_keys))
        stats["commit/weight"] = weight
        
        # Check for periodic commit
        if self.cfg.mode in [CommitMode.PERIODIC, CommitMode.HYBRID]:
            if self.step_counter - self.last_commit_step >= self.cfg.commit_interval:
                commit_stats = self._execute_commit()
                stats.update(commit_stats)
        
        return stats
    
    def _compute_weight(
        self,
        advantage: Optional[float],
        reward: Optional[float],
        td_error: Optional[float],
    ) -> float:
        """Compute commit weight based on config."""
        if self.cfg.weighting == CommitWeighting.UNIFORM:
            return 1.0
        elif self.cfg.weighting == CommitWeighting.ADVANTAGE:
            return abs(advantage) if advantage is not None else 1.0
        elif self.cfg.weighting == CommitWeighting.REWARD:
            return max(0.0, reward) if reward is not None else 1.0
        elif self.cfg.weighting == CommitWeighting.TD_ERROR:
            return abs(td_error) if td_error is not None else 1.0
        return 1.0
    
    def _add_to_window(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        weight: float,
    ) -> None:
        """Add state to sliding window buffer."""
        self.window_keys.append(key.detach())
        self.window_values.append(value.detach())
        self.window_weights.append(weight)
        
        # Check if window is full
        if len(self.window_keys) >= self.cfg.window_size:
            # Move window contents to pending
            for k, v, w in zip(
                self.window_keys[:self.cfg.window_stride],
                self.window_values[:self.cfg.window_stride],
                self.window_weights[:self.cfg.window_stride],
            ):
                self.pending_keys.append(k)
                self.pending_values.append(v)
                self.pending_weights.append(w)
                self.pending_steps.append(self.step_counter)
            
            # Slide window
            self.window_keys = self.window_keys[self.cfg.window_stride:]
            self.window_values = self.window_values[self.cfg.window_stride:]
            self.window_weights = self.window_weights[self.cfg.window_stride:]
    
    def on_episode_end(self) -> Dict[str, float]:
        """
        Called at episode boundary to trigger commits.
        
        Returns:
            Commit stats
        """
        stats: Dict[str, float] = {}
        
        if self.cfg.mode in [CommitMode.EPISODE_END, CommitMode.HYBRID]:
            commit_stats = self._execute_commit()
            stats.update(commit_stats)
        
        # Compute episode-level stats
        if self.episode_rewards:
            stats["commit/episode_return"] = sum(self.episode_rewards)
            stats["commit/episode_length"] = float(len(self.episode_rewards))
        
        # Reset episode accumulators
        self.episode_rewards = []
        self.episode_states = []
        
        return stats
    
    def force_commit(self) -> Dict[str, float]:
        """Force an immediate commit of all pending states."""
        return self._execute_commit()
    
    def _execute_commit(self) -> Dict[str, float]:
        """Execute the actual LTM commit."""
        stats: Dict[str, float] = {}
        
        if not self.pending_keys:
            stats["commit/committed_states"] = 0.0
            return stats
        
        # Limit batch size
        n_commit = min(len(self.pending_keys), self.cfg.max_commit_batch)
        
        # Select top-weighted states if over limit
        if len(self.pending_keys) > self.cfg.max_commit_batch:
            indices = sorted(
                range(len(self.pending_weights)),
                key=lambda i: self.pending_weights[i],
                reverse=True,
            )[:n_commit]
            
            keys = [self.pending_keys[i] for i in indices]
            values = [self.pending_values[i] for i in indices]
            weights = [self.pending_weights[i] for i in indices]
        else:
            keys = self.pending_keys[:n_commit]
            values = self.pending_values[:n_commit]
            weights = self.pending_weights[:n_commit]
        
        # Stack tensors
        keys_tensor = torch.stack(keys, dim=0)
        values_tensor = torch.stack(values, dim=0)
        
        # Commit to LTM
        try:
            self.ltm.store(keys_tensor, values_tensor)
            
            self.total_commits += 1
            self.total_states_committed += n_commit
            self.last_commit_step = self.step_counter
            self.last_commit_score = sum(weights) / len(weights) if weights else 0.0
            
            stats["commit/committed_states"] = float(n_commit)
            stats["commit/mean_weight"] = self.last_commit_score
            stats["commit/total_commits"] = float(self.total_commits)
            stats["commit/ltm_size"] = float(self.ltm.size())
        except Exception as e:
            stats["commit/error"] = 1.0
            stats["commit/committed_states"] = 0.0
        
        # Clear committed states
        if len(self.pending_keys) > self.cfg.max_commit_batch:
            # Keep uncommitted states
            remaining = set(range(len(self.pending_keys))) - set(
                sorted(
                    range(len(self.pending_weights)),
                    key=lambda i: self.pending_weights[i],
                    reverse=True,
                )[:n_commit]
            )
            self.pending_keys = [self.pending_keys[i] for i in remaining]
            self.pending_values = [self.pending_values[i] for i in remaining]
            self.pending_weights = [self.pending_weights[i] for i in remaining]
            self.pending_steps = [self.pending_steps[i] for i in remaining]
        else:
            self.pending_keys = self.pending_keys[n_commit:]
            self.pending_values = self.pending_values[n_commit:]
            self.pending_weights = self.pending_weights[n_commit:]
            self.pending_steps = self.pending_steps[n_commit:]
        
        return stats
    
    def get_stats(self) -> Dict[str, float]:
        """Get current commit engine stats."""
        return {
            "commit/total_commits": float(self.total_commits),
            "commit/total_states_committed": float(self.total_states_committed),
            "commit/pending_size": float(len(self.pending_keys)),
            "commit/window_size": float(len(self.window_keys)),
            "commit/last_commit_score": self.last_commit_score,
            "commit/steps_since_commit": float(
                self.step_counter - self.last_commit_step
            ),
        }
    
    def reset(self) -> None:
        """Reset all buffers and counters."""
        self.pending_keys = []
        self.pending_values = []
        self.pending_weights = []
        self.pending_steps = []
        self.window_keys = []
        self.window_values = []
        self.window_weights = []
        self.episode_rewards = []
        self.episode_states = []
        self.step_counter = 0
        self.last_commit_step = 0
