# GroomEnvSB3.py  –  part of GroomRL + SB3 integration
"""
Gymnasium-compatible wrappers around the original (gym-based) GroomEnv family.
Stable-Baselines3 >= 1.7 requires *gymnasium* rather than the legacy *gym* API:
  • reset()  must return  (obs, info)
  • step()   must return  (obs, reward, terminated, truncated, info)

We thin-wrap every GroomEnv variant so the rest of the codebase stays untouched.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from groomrl.GroomEnv import GroomEnv, GroomEnvDual, GroomEnvTriple
from groomrl.JetTree import LundCoordinates


# ──────────────────────────────────────────────────────────────────────────────
def _make_gymnasium_spaces(low: np.ndarray, high: np.ndarray):
    """Return Gymnasium Box + Discrete spaces matching the original gym spaces."""
    obs_space = spaces.Box(
        low=low.astype(np.float32),
        high=high.astype(np.float32),
        dtype=np.float32,
    )
    act_space = spaces.Discrete(2)
    return obs_space, act_space


# ──────────────────────────────────────────────────────────────────────────────
class GroomEnvSB3(gymnasium.Env):
    """
    Single-signal gymnasium wrapper around :class:`groomrl.GroomEnv.GroomEnv`.

    Parameters
    ----------
    hps : dict
        Same hyper-parameter dictionary accepted by the original GroomEnv.
    low  : np.ndarray
        Lower bound of the observation space (from LundCoordinates.low).
    high : np.ndarray
        Upper bound of the observation space (from LundCoordinates.high).
    """

    metadata = {"render_modes": []}

    def __init__(self, hps: dict, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self._env = GroomEnv(hps, low, high)
        self.observation_space, self.action_space = _make_gymnasium_spaces(low, high)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action: int):
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated  = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # --------------------------------------------------- pass-through helpers
    @property
    def description(self):
        return self._env.description

    @property
    def massgoal(self):
        return self._env.massgoal

    @property
    def events(self):
        return self._env.events


# ──────────────────────────────────────────────────────────────────────────────
class GroomEnvDualSB3(gymnasium.Env):
    """Gymnasium wrapper around :class:`groomrl.GroomEnv.GroomEnvDual`."""

    metadata = {"render_modes": []}

    def __init__(self, hps: dict, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self._env = GroomEnvDual(hps, low, high)
        self.observation_space, self.action_space = _make_gymnasium_spaces(low, high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action: int):
        obs, reward, done, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ──────────────────────────────────────────────────────────────────────────────
class GroomEnvTripleSB3(gymnasium.Env):
    """Gymnasium wrapper around :class:`groomrl.GroomEnv.GroomEnvTriple`."""

    metadata = {"render_modes": []}

    def __init__(self, hps: dict, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self._env = GroomEnvTriple(hps, low, high)
        self.observation_space, self.action_space = _make_gymnasium_spaces(low, high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action: int):
        obs, reward, done, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ──────────────────────────────────────────────────────────────────────────────
def make_sb3_env(env_setup: dict, low: np.ndarray, high: np.ndarray) -> gymnasium.Env:
    """
    Factory function that selects the correct SB3 environment variant based on
    the ``env_setup`` dictionary, mirroring the logic in ``models.build_and_train_model``.
    """
    if env_setup.get("dual_groomer_env"):
        return GroomEnvDualSB3(env_setup, low, high)
    if env_setup.get("triple_groomer_env"):
        return GroomEnvTripleSB3(env_setup, low, high)
    return GroomEnvSB3(env_setup, low, high)
