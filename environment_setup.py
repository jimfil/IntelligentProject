"""
Project 1: Safe Mobile Robot Button Navigation
Environment setup, wrappers, and observation normalization.

Environment: SafetyRacecarButton2-v0 (Safety-Gymnasium)
Robot: Racecar - rear-wheel velocity + front-wheel steering control

Rules:
  - Do NOT modify reward function, cost function, termination logic,
    random seeds used for official evaluation, or the scoring script.
  - Allowed: observation normalization, action smoothing, frame stacking,
    recurrent state, memory buffers, learned representations, demonstrations,
    model-based planning.
"""

import dataclasses
import numpy as np
import gymnasium

# Patch dataclasses to allow numpy arrays as default values (compatibility for Python 3.11+)
_orig_get_field = dataclasses._get_field
def _patched_get_field(cls, a_name, a_type, default_kw_only):
    default = getattr(cls, a_name, dataclasses.MISSING)
    if isinstance(default, np.ndarray):
        placeholder = tuple(default.tolist())
        setattr(cls, a_name, placeholder)
        try:
            f = _orig_get_field(cls, a_name, a_type, default_kw_only)
            f.default = default
            setattr(cls, a_name, default)
            return f
        except Exception:
            setattr(cls, a_name, default)
            raise
    return _orig_get_field(cls, a_name, a_type, default_kw_only)

dataclasses._get_field = _patched_get_field

import safety_gymnasium


# ---------------------------------------------------------------------------
# Running statistics for online observation normalization (Welford's method)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Tracks running mean and variance of a vector using Welford's algorithm."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small epsilon to avoid division by zero at start

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -clip, clip)

    def save(self, path: str):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = float(data["count"])


# ---------------------------------------------------------------------------
# Observation Normalizer Wrapper
# ---------------------------------------------------------------------------

class ObsNormWrapper(gymnasium.ObservationWrapper):
    """
    Normalizes observations using a running mean/std.

    ALLOWED by the benchmark rules: observation normalization.
    The running stats are updated only during training (update=True).
    During evaluation pass update=False to freeze the statistics.
    """

    def __init__(self, env, update: bool = True, clip: float = 10.0):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.rms = RunningMeanStd(shape=obs_shape)
        self.update = update
        self.clip = clip

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, cost, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if self.update:
            self.rms.update(obs[np.newaxis])  # treat single obs as batch of 1
        return self.rms.normalize(obs, clip=self.clip).astype(np.float32)

    def freeze(self):
        """Call before evaluation to stop updating running stats."""
        self.update = False

    def save_stats(self, path: str):
        self.rms.save(path)

    def load_stats(self, path: str):
        self.rms.load(path)
        self.update = False


# ---------------------------------------------------------------------------
# Action Smoothing Wrapper
# ---------------------------------------------------------------------------

class ActionSmoothingWrapper(gymnasium.ActionWrapper):
    """
    Applies exponential moving average smoothing to actions.

    ALLOWED by the benchmark rules: action smoothing.
    alpha=1.0 → no smoothing (pass-through).
    alpha→0   → heavy smoothing (slow to change).
    """

    def __init__(self, env, alpha: float = 0.8):
        super().__init__(env)
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = alpha
        self._prev_action = None

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(self.action(action))
        return obs, reward, cost, terminated, truncated, info

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            self._prev_action = action.copy()
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# Frame Stacking Wrapper
# ---------------------------------------------------------------------------

class FrameStackWrapper(gymnasium.ObservationWrapper):
    """
    Stacks the last `n_frames` observations along the last axis.

    ALLOWED by the benchmark rules: frame stacking.
    Useful for giving the controller a short history of sensor readings.
    """

    def __init__(self, env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        obs_shape = env.observation_space.shape
        stacked_shape = (obs_shape[0] * n_frames,)
        low = np.tile(env.observation_space.low, n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self._frames = np.zeros(stacked_shape, dtype=np.float32)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, cost, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = len(obs)
        # Shift old frames left, insert new obs at the end
        self._frames = np.roll(self._frames, -n)
        self._frames[-n:] = obs
        return self._frames.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames[:] = 0.0
        # Fill all frames with the initial observation
        for _ in range(self.n_frames):
            self._frames = np.roll(self._frames, -len(obs))
            self._frames[-len(obs):] = obs
        return self._frames.copy(), info


# ---------------------------------------------------------------------------
# Safe Gymnasium Cost Tracking Wrapper
# ---------------------------------------------------------------------------

class CostTrackingWrapper(gymnasium.Wrapper):
    """
    Accumulates the per-step cost signal returned by Safety-Gymnasium
    and exposes episode statistics.

    Safety-Gymnasium's step() returns:
        obs, reward, cost, terminated, truncated, info

    This wrapper stores the cost in info and accumulates it for logging.
    The cost is NOT modified — benchmark rules forbid that.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0

    def reset(self, **kwargs):
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Safety-Gymnasium returns a 6-tuple including cost
        result = self.env.step(action)
        obs, reward, cost, terminated, truncated, info = result

        self.episode_cost += cost
        self.episode_reward += reward
        self.episode_steps += 1

        info["cost"] = cost
        info["episode_cost"] = self.episode_cost
        info["episode_reward"] = self.episode_reward
        info["episode_steps"] = self.episode_steps

        return obs, reward, cost, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    seed: int = 0,
    normalize_obs: bool = True,
    smooth_actions: bool = True,
    action_alpha: float = 0.8,
    frame_stack: int = 1,
    update_obs_stats: bool = True,
    render_mode: str = None,
) -> gymnasium.Env:
    """
    Creates and wraps the SafetyRacecarButton2-v0 environment.

    Parameters
    ----------
    seed : int
        Random seed for environment initialization.
    normalize_obs : bool
        Whether to apply running-mean observation normalization.
    smooth_actions : bool
        Whether to apply exponential action smoothing.
    action_alpha : float
        EMA coefficient for action smoothing (1.0 = no smoothing).
    frame_stack : int
        Number of observations to stack (1 = no stacking).
    update_obs_stats : bool
        Whether to update the running observation statistics.
        Set to False during evaluation.
    render_mode : str or None
        Passed directly to safety_gymnasium.make().

    Returns
    -------
    gymnasium.Env
        The wrapped environment.
    """
    env = safety_gymnasium.make(
        "SafetyRacecarButton2-v0",
        render_mode=render_mode,
    )

    # Always track costs (does not modify cost values, only logs them)
    env = CostTrackingWrapper(env)

    # Allowed wrappers (benchmark rules permit all of these)
    if frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)

    if normalize_obs:
        env = ObsNormWrapper(env, update=update_obs_stats)

    if smooth_actions:
        env = ActionSmoothingWrapper(env, alpha=action_alpha)

    return env


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Creating SafetyRacecarButton2-v0 with wrappers...")
    env = make_env(seed=0, normalize_obs=True, smooth_actions=True, frame_stack=4)

    obs, info = env.reset(seed=0)
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action space      : {env.action_space}")

    total_reward = 0.0
    total_cost = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += cost

    print(f"  Episode reward    : {total_reward:.4f}")
    print(f"  Episode cost      : {total_cost:.4f}")
    print(f"  Episode steps     : {info['episode_steps']}")
    env.close()
    print("Smoke test passed.")
