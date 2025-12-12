"""
envs.py

Environment utilities and wrappers for Infinity V3.

Provides:
- Gym environment creation helpers
- Environment wrappers for observation normalization
- Multi-environment support
"""

from typing import List, Tuple, Any, Optional, Dict
import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    try:
        import gym
        HAS_GYM = True
    except ImportError:
        gym = None
        HAS_GYM = False


if HAS_GYM:

    class DelayedCueEnv(gym.Env):
        def __init__(
            self,
            delay: int = 1000,
            noise_dim: int = 4,
            regime_shift: bool = False,
        ):
            super().__init__()
            self.delay = int(delay)
            self.noise_dim = int(noise_dim)
            self.regime_shift = bool(regime_shift)

            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.noise_dim + 1,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Discrete(2)

            self.t = 0
            self.cue_time = 0
            self.cue = 0

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ):
            _ = options
            super().reset(seed=seed)
            if self.regime_shift:
                self.delay = int(np.random.choice([200, 500, 1000, 2000]))
            self.t = 0
            self.cue_time = int(np.random.randint(10, 50))
            self.cue = int(np.random.randint(0, 2))
            return self._obs(), {}

        def _obs(self) -> np.ndarray:
            cue_value = float(self.cue) if self.t == self.cue_time else 0.0
            noise = np.random.randn(self.noise_dim).astype(np.float32) * 0.1
            obs = np.concatenate(
                [noise, np.array([cue_value], dtype=np.float32)]
            )
            return obs.astype(np.float32)

        def step(self, action: int):
            reward = 0.0
            terminated = False
            truncated = False

            self.t += 1

            if self.t < self.delay:
                if int(action) != 0:
                    reward = -1.0
            else:
                reward = 1.0 if int(action) == self.cue else -1.0
                terminated = True

            return self._obs(), float(reward), terminated, truncated, {}


def make_env(env_id: str) -> Any:
    """
    Create a single environment.

    Args:
        env_id: Gym environment ID (e.g., "CartPole-v1")
    Returns:
        Gym environment instance
    """
    if not HAS_GYM:
        raise ImportError(
            "gym/gymnasium not installed. "
            "Install with: pip install gym or pip install gymnasium"
        )
    if env_id == "DelayedCue-v0":
        return DelayedCueEnv()
    return gym.make(env_id)


def make_envs(env_id: str, num_envs: int = 1) -> List[Any]:
    """
    Create multiple environments.

    Args:
        env_id: Gym environment ID
        num_envs: Number of environments to create
    Returns:
        List of Gym environment instances
    """
    return [make_env(env_id) for _ in range(num_envs)]


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


def get_env_info(env_id: str) -> dict:
    """
    Get environment observation and action dimensions.

    Args:
        env_id: Gym environment ID
    Returns:
        Dict with obs_dim and act_dim
    """
    env = make_env(env_id)

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
