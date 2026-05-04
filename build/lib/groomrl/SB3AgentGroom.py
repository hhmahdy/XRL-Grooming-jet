# SB3AgentGroom.py  –  part of GroomRL + SB3 integration
"""
Stable-Baselines3 agent wrapper that presents the same high-level interface as
the original :class:`groomrl.DQNAgentGroom.DQNAgentGroom` (keras-rl based), so
the rest of the pipeline (``models.py``, ``groomer.py``) needs minimal changes.

Supported agent types
---------------------
``"dqn"``   –  Off-policy Deep Q-Network (``stable_baselines3.DQN``)
``"ppo"``   –  On-policy Proximal Policy Optimisation (``stable_baselines3.PPO``)

Both agents use ``MlpPolicy`` which is appropriate for the low-dimensional Lund
coordinate observation vectors produced by :class:`groomrl.GroomEnv.GroomEnv`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium

# SB3 imports – optional so the rest of the codebase still loads without SB3
try:
    from stable_baselines3 import DQN, PPO, MDPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from groomrl.Groomer import AbstractGroomer
from groomrl.JetTree import JetTree, LundCoordinates

SUPPORTED_AGENTS = ("dqn", "ppo", "mdpo")  # Supported agent types (SB3 algorithms)


# ──────────────────────────────────────────────────────────────────────────────
class _SB3Groomer(AbstractGroomer):
    """
    Groomer that drives grooming decisions via a Stable-Baselines3 policy.

    Works identically to :class:`groomrl.Groomer.Groomer` but calls
    ``model.predict`` instead of Keras ``predict_on_batch``.
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    def _groom(self, tree: JetTree):
        if not tree.lundCoord:
            return
        state = tree.state().astype(np.float32)
        action, _states = self.model.predict(state, deterministic=True)
        action = int(action)

        if action == 1:
            tree.remove_soft()
            self._groom(tree)
        else:
            if tree.harder:
                self._groom(tree.harder)
            if tree.softer:
                self._groom(tree.softer)


# ──────────────────────────────────────────────────────────────────────────────
class RewardTrackingCallback(BaseCallback):
    """
    Lightweight SB3 callback that records per-episode rewards and lengths into
    a history dict compatible with keras-rl's ``History.history`` structure, so
    the existing plotting / logging code can consume it unchanged.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.history: dict = {
            "episode_reward": [],
            "nb_episode_steps": [],
            "nb_steps": [],
        }
        self._episode_rewards: list[float] = []
        self._episode_steps: list[int] = []

    # ------------------------------------------------- SB3 callback hooks
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.history["episode_reward"].append(ep_info["r"])
                self.history["nb_episode_steps"].append(ep_info["l"])
                self.history["nb_steps"].append(self.num_timesteps)
        return True


# ──────────────────────────────────────────────────────────────────────────────
class SB3AgentGroom:
    """
    Unified wrapper around ``stable_baselines3.DQN`` and ``stable_baselines3.PPO``
    for use inside the GroomRL training pipeline.

    Parameters
    ----------
    agent_type : str
        ``"dqn"`` or ``"ppo"``.
    env : gymnasium.Env
        A Gymnasium-compatible GroomEnv (see :mod:`groomrl.GroomEnvSB3`).
    hps : dict
        ``groomer_agent`` section of the runcard.  Used to read
        ``learning_rate``, ``nb_units``, ``nb_layers``, ``nstep``, and
        optional ``policy_kwargs`` overrides.
    verbose : int
        Verbosity level forwarded to SB3.
    """

    def __init__(
        self,
        agent_type: str,
        env: gymnasium.Env,
        hps: dict,
        verbose: int = 1,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is not installed.  "
                "Run:  pip install stable-baselines3"
            )
        agent_type = agent_type.lower()
        if agent_type not in SUPPORTED_AGENTS:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'.  "
                f"Choose from: {SUPPORTED_AGENTS}"
            )

        self.agent_type = agent_type
        self.hps = hps
        self._history: Optional[RewardTrackingCallback] = None

        # Build the network architecture from runcard parameters
        policy_kwargs = self._build_policy_kwargs(hps)

        lr = float(hps.get("learning_rate", 1e-3))

        if agent_type == "dqn":
            self.model = DQN(
                policy="MlpPolicy",
                env=Monitor(env),          # Monitor adds episode stats to info
                learning_rate=lr,
                buffer_size=int(hps.get("buffer_size", 500_000)),
                learning_starts=int(hps.get("learning_starts", 500)),
                batch_size=int(hps.get("batch_size", 32)),
                tau=float(hps.get("tau", 1.0)),
                gamma=float(hps.get("gamma", 0.99)),
                target_update_interval=int(hps.get("target_update_interval", 100)),
                exploration_fraction=float(hps.get("exploration_fraction", 0.1)),
                exploration_final_eps=float(hps.get("exploration_final_eps", 0.05)),
                policy_kwargs=policy_kwargs,
                verbose=verbose,
            )
        elif agent_type == "mdpo":
            self.model = MDPO(
                policy="MlpPolicy",
                env=Monitor(env),
                learning_rate=lr,
                method="multistep-SGD",
                n_steps=int(hps.get("n_steps", 2048)),
                batch_size=int(hps.get("batch_size", 64)),
                gamma=float(hps.get("gamma", 0.99)),
                gae_lambda=float(hps.get("gae_lambda", 0.95)),
                clip_range_vf=float(hps.get("clip_range", 0.2)),
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(hps.get("tensorboard_log", "/tensorboard")),
                verbose=verbose,
            )
        else:  # ppo
            self.model = PPO(
                policy="MlpPolicy",
                env=Monitor(env),
                learning_rate=lr,
                n_steps=int(hps.get("n_steps", 2048)),
                batch_size=int(hps.get("batch_size", 64)),
                n_epochs=int(hps.get("n_epochs", 10)),
                gamma=float(hps.get("gamma", 0.99)),
                gae_lambda=float(hps.get("gae_lambda", 0.95)),
                clip_range=float(hps.get("clip_range", 0.2)),
                ent_coef=float(hps.get("ent_coef", 0.0)),
                policy_kwargs=policy_kwargs,
                verbose=verbose,
            )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_policy_kwargs(hps: dict) -> dict:
        """Convert runcard network params into SB3 ``policy_kwargs``."""
        nb_units  = int(hps.get("nb_units", 64))
        nb_layers = int(hps.get("nb_layers", 2))
        net_arch  = [nb_units] * nb_layers
        return {"net_arch": net_arch}

    # ------------------------------------------------------------------ public API (mirrors DQNAgentGroom)
    def fit(
        self,
        total_timesteps: Optional[int] = None,
        callbacks=None,
        verbose: int = 1,
    ) -> "SB3AgentGroom":
        """
        Train the agent.

        Parameters
        ----------
        total_timesteps : int, optional
            Override the number of training steps (defaults to ``hps['nstep']``).
        callbacks : list, optional
            Additional SB3 callbacks.
        """
        n_steps = int(total_timesteps or self.hps.get("nstep", 50_000))

        reward_cb = RewardTrackingCallback(verbose=0)
        all_cbs   = [reward_cb] + (callbacks or [])

        print(f"[+] Training SB3 {self.agent_type.upper()} for {n_steps} steps …")
        self.model.learn(total_timesteps=n_steps, callback=all_cbs)
        self._history = reward_cb
        return self

    @property
    def history(self):
        """Expose training history (compatible with keras-rl History object)."""
        if self._history is None:
            return None
        return self._history

    def groomer(self) -> _SB3Groomer:
        """Return a :class:`_SB3Groomer` that uses the trained policy."""
        return _SB3Groomer(self.model)

    # ------------------------------------------------------------------ persistence
    def save_weights(self, filepath: str, **kwargs):
        """Save the SB3 model. Ignores kwargs for API compat with keras-rl."""
        # SB3 save adds .zip automatically; strip extension if provided
        base = str(filepath).replace(".zip", "").replace(".h5", "")
        self.model.save(base)
        print(f"[+] SB3 model saved → {base}.zip")

    @classmethod
    def load(cls, agent_type: str, path: str, env: gymnasium.Env = None) -> "SB3AgentGroom":
        """
        Load a previously saved SB3 model from *path* (without ``.zip``).

        Returns an ``SB3AgentGroom`` instance with ``hps={}`` and the loaded
        ``model`` already set.
        """
        if not _SB3_AVAILABLE:
            raise ImportError("stable_baselines3 is not installed.")
        obj = object.__new__(cls)
        obj.agent_type = agent_type.lower()
        obj.hps = {}
        obj._history = None

        base = str(path).replace(".zip", "")
        if agent_type == "dqn":
            obj.model = DQN.load(base, env=Monitor(env) if env else None)
        elif agent_type == "mdpo":
            obj.model = MDPO.load(base, env=Monitor(env) if env else None)
        else:
            obj.model = PPO.load(base, env=Monitor(env) if env else None)
        print(f"[+] Loaded SB3 {agent_type.upper()} from {base}.zip")
        return obj

    # ------------------------------------------------------------------ info
    def __repr__(self):
        return f"SB3AgentGroom(agent_type={self.agent_type!r})"
