"""
envs.py

Environment utilities and wrappers for Infinity V3.

Provides:
- Gym environment creation helpers
- Environment wrappers for observation normalization
- Multi-environment support
"""

from typing import List, Tuple, Any, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class DelayedCueEnv(gym.Env):
    def __init__(
        self,
        episode_len: int = 2000,
        cue_time: int = 50,
        delay: int = 1000,
        window: int = 25,
        noise_std: float = 0.1,
        step_penalty: float = -0.001,
        action_mode: str = "discrete",
    ):
        super().__init__()
        assert action_mode == "discrete"
        self.episode_len = int(episode_len)
        self.cue_time = int(cue_time)
        self.delay = int(delay)
        self.window = int(window)
        self.noise_std = float(noise_std)
        self.step_penalty = float(step_penalty)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self.t = 0
        self._cue_seen = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        _ = options
        super().reset(seed=seed)
        self.t = 0
        self._cue_seen = False
        return self._obs(), {}

    def _target_time(self) -> int:
        return int(self.cue_time + self.delay)

    def _in_window(self) -> bool:
        return (
            self._target_time()
            <= self.t
            <= (self._target_time() + self.window)
        )

    def _obs(self) -> np.ndarray:
        cue_bit = 1.0 if self.t == self.cue_time else 0.0
        if cue_bit > 0:
            self._cue_seen = True

        t_norm = float(self.t) / max(float(self.episode_len), 1.0)
        if self._cue_seen:
            ts_norm = float(self.t - self.cue_time) / max(
                float(self.episode_len),
                1.0,
            )
        else:
            ts_norm = 0.0

        target_time_norm = float(self._target_time()) / max(
            float(self.episode_len),
            1.0,
        )
        noise = float(np.random.randn() * self.noise_std)

        obs = np.array(
            [cue_bit, t_norm, ts_norm, target_time_norm, noise],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        action = int(action)
        reward = float(self.step_penalty)

        self.t += 1

        if action == 1 and not self._in_window():
            return self._obs(), -1.0, True, False, {}

        if action == 1 and self._in_window():
            return self._obs(), 10.0, True, False, {}

        if self.t >= self.episode_len:
            return self._obs(), reward, False, True, {}

        return self._obs(), reward, False, False, {}


class DelayedCueRegimeEnv(DelayedCueEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_time = self.episode_len // 2
        self.regime = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        self.shift_time = self.episode_len // 2
        self.regime = 0
        return obs, info

    def step(self, action: int):
        if self.t >= self.shift_time:
            self.regime = 1

        action = int(action)
        reward = float(self.step_penalty)

        self.t += 1

        if self._in_window():
            correct = 1 if self.regime == 0 else 0
            if action == 1:
                rew = 10.0 if correct == 1 else -1.0
                return self._obs(), rew, True, False, {}

        if action == 1 and not self._in_window():
            return self._obs(), -1.0, True, False, {}

        if self.t >= self.episode_len:
            return self._obs(), reward, False, True, {}

        return self._obs(), reward, False, False, {}


def make_env(env_id: str, cfg: Optional[Any] = None) -> Any:
    """
    Create a single environment.

    Args:
        env_id: Gym environment ID (e.g., "CartPole-v1")
    Returns:
        Gym environment instance
    """
    if env_id == "DelayedCue-v0":
        return DelayedCueEnv(
            episode_len=int(getattr(cfg, "delayedcue_episode_len", 2000)),
            cue_time=int(getattr(cfg, "delayedcue_cue_time", 50)),
            delay=int(getattr(cfg, "delayedcue_delay", 1000)),
            window=int(getattr(cfg, "delayedcue_window", 25)),
            noise_std=float(getattr(cfg, "delayedcue_noise_std", 0.1)),
            step_penalty=float(
                getattr(cfg, "delayedcue_step_penalty", -0.001)
            ),
        )
    if env_id == "DelayedCueRegime-v0":
        return DelayedCueRegimeEnv(
            episode_len=int(getattr(cfg, "delayedcue_episode_len", 2000)),
            cue_time=int(getattr(cfg, "delayedcue_cue_time", 50)),
            delay=int(getattr(cfg, "delayedcue_delay", 1000)),
            window=int(getattr(cfg, "delayedcue_window", 25)),
            noise_std=float(getattr(cfg, "delayedcue_noise_std", 0.1)),
            step_penalty=float(
                getattr(cfg, "delayedcue_step_penalty", -0.001)
            ),
        )
    return gym.make(env_id)


def make_envs(
    env_id: str,
    num_envs: int = 1,
    cfg: Optional[Any] = None,
) -> List[Any]:
    """
    Create multiple environments.

    Args:
        env_id: Gym environment ID
        num_envs: Number of environments to create
    Returns:
        List of Gym environment instances
    """
    return [make_env(env_id, cfg=cfg) for _ in range(num_envs)]


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.

    Uses Welford's algorithm for numerically stable updates.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / tot_count
        )
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizedObservationWrapper:
    """
    Wrapper that normalizes observations using running statistics.

    Maintains running mean and std of observations and normalizes
    incoming observations to zero mean and unit variance.
    """

    def __init__(self, env: Any, clip: float = 10.0):
        self.env = env
        self.clip = clip
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        """Reset and normalize observation."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            obs = self._normalize(obs)
            return obs, info
        else:
            return self._normalize(result)

    def step(self, action):
        """Step and normalize observation."""
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            obs = self._normalize(obs)
            return obs, reward, done, truncated, info
        else:
            obs, reward, done, info = result
            obs = self._normalize(obs)
            return obs, reward, done, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        self.obs_rms.update(obs.reshape(1, -1))
        normalized = (obs - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + 1e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def close(self):
        """Close environment."""
        self.env.close()


class RewardScalingWrapper:
    """
    Wrapper that scales rewards using running statistics.
    """

    def __init__(self, env: Any, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.ret_rms = RunningMeanStd(shape=())
        self._ret = 0.0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        """Reset environment."""
        self._ret = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step with reward scaling."""
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done_flag = done or truncated
        else:
            obs, reward, done, info = result
            done_flag = done
            truncated = False

        # Update return estimate
        self._ret = self._ret * self.gamma + reward
        self.ret_rms.update(np.array([self._ret]))

        # Scale reward
        scaled_reward = reward / np.sqrt(self.ret_rms.var + 1e-8)

        if done_flag:
            self._ret = 0.0

        if len(result) == 5:
            return obs, scaled_reward, done, truncated, info
        return obs, scaled_reward, done, info

    def close(self):
        """Close environment."""
        self.env.close()


def wrap_env(
    env: Any,
    normalize_obs: bool = False,
    scale_reward: bool = False,
    gamma: float = 0.99,
) -> Any:
    """
    Apply wrappers to environment.

    Args:
        env: Base environment
        normalize_obs: Apply observation normalization
        scale_reward: Apply reward scaling
        gamma: Discount factor for reward scaling
    Returns:
        Wrapped environment
    """
    if normalize_obs:
        env = NormalizedObservationWrapper(env)
    if scale_reward:
        env = RewardScalingWrapper(env, gamma=gamma)
    return env


def get_env_info(env_id: str, cfg: Optional[Any] = None) -> dict:
    """
    Get environment observation and action dimensions.

    Args:
        env_id: Gym environment ID
    Returns:
        Dict with obs_dim and act_dim
    """
    env = make_env(env_id, cfg=cfg)

    # Get observation dimension
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        obs_dim = (
            obs_space.shape[0]
            if len(obs_space.shape) == 1
            else obs_space.shape
        )
    else:
        obs_dim = obs_space.n

    # Get action dimension
    act_space = env.action_space
    if hasattr(act_space, 'n'):
        act_dim = act_space.n  # Discrete
    elif hasattr(act_space, 'shape'):
        act_dim = act_space.shape[0]  # Continuous
    else:
        act_dim = 1

    env.close()

    return {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "is_discrete": hasattr(act_space, 'n'),
    }
