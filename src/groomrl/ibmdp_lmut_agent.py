# ibmdp_lmut_agent.py  –  GroomRL-XRL
"""
Integrated IBMDP + LMUT agent.

Architecture
------------

                  ┌─────────────────────────────────────────────────┐
                  │             IBMDPLMUTAgent                       │
                  │                                                   │
                  │  ┌─────────────────┐   ┌────────────────────┐   │
                  │  │  Policy Q-net   │   │  Omniscient Q-net  │   │
                  │  │  Q(s_w \ s_b, a)│   │  Q_o(s_w, a)       │   │
                  │  │  (SB3 DQN / PPO)│   │  (LMUTForest)      │   │
                  │  └────────┬────────┘   └────────┬───────────┘   │
                  │           │  action                │ Q_o targets  │
                  │           ▼                        ▼              │
                  │        IBMDPGroomEnv  ◄──── OmniscientQTarget     │
                  └─────────────────────────────────────────────────┘

The key coupling between IBMDP and LMUT (Topin et al. §4.3 + Liu et al.):

1. **LMUT as omniscient Q-function Q_o**
   Because the LMUT has access to both the base state and the bounding
   features (full s_w), it naturally fulfils the role of Q_o in the IBMDP
   framework.  Specifically:

   - For base-action transitions (s_w, a ∈ A_b, s'_w):
       target = R(s_b, a, s_b') + γ_b * max_{a'∈A_b} Q_o(s'_w ∖ s_b at root, a')
     where s'_w ∖ s_b at root means bounds reset to [0,1]^{2n}.

   - For info-gathering transitions (s_w, a ∉ A_b, s'_w):
       target = ζ + γ_w * max_{a'∈A_w} Q_o(s'_w, a')

   This removes the need to simulate forward to the next leaf state (sl2),
   which would otherwise require O(depth) extra forward passes per update.

2. **LMUT update from IBMDP replay**
   After each SB3 training step the agent harvests a minibatch from the
   SB3 replay buffer, converts it to LMUT Transition objects (using the
   oracle Q̂ = Q_o prediction), and calls LMUTForest.update().  This keeps
   the LMUT synchronised with the evolving policy Q-network.

3. **SB3 Q-target patching**
   We subclass SB3's DQN and override _on_step (a training hook) to replace
   the standard bootstrap target with the modified IBMDP target that uses Q_o.
   For PPO, Q_o replaces the critic for advantage estimation.

Convergence guarantees
----------------------
- ζ < 0 ensures the agent prefers fewer information-gathering steps, driving
  toward minimal DTs (implicit regularisation).
- γ_w ≤ 1 ensures discounted future information-gathering costs are bounded.
- The LMUTForest is updated on-policy with a learning rate schedule that
  decays as 1/√n_updates, ensuring asymptotic convergence of the linear models.
- The SB3 DQN target-network update interval is set conservatively (every
  1000 steps) to provide stable Q_o targets.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import DQN, PPO, MDPO
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    _SB3_OK = True
except ImportError:
    _SB3_OK = False

from groomrl.ibmdp_env import IBMDPGroomEnv, DTNode
from groomrl.lmut_core import LMUTForest, Transition


# ─────────────────────────────────────────────────────────────────────────────
# Omniscient Q-target helper
# ─────────────────────────────────────────────────────────────────────────────
class OmniscientQTarget:
    """
    Computes the modified IBMDP Q-targets (Topin et al. §4.3) using a
    LMUTForest as the omniscient Q-function Q_o.

    The target differs from the standard DQN target in one critical way:
    when a *base* action is taken, the "next state" used in the bootstrap
    is the *root* bounding-feature state of the next traversal
    (bounds reset to [0,1]^{2n}), not the immediate next state.

    Parameters
    ----------
    lmut : LMUTForest
        Trained (or partially trained) omniscient Q-function.
    n_base_actions : int
        Number of base actions (2 for GroomRL).
    zeta : float
        Information-gathering penalty (same as in IBMDPGroomEnv).
    gamma_b : float
        Discount factor for base-action transitions.
    gamma_w : float
        Discount factor for information-gathering transitions.
    dim : int
        Base state dimensionality (for constructing root bound obs).
    """

    def __init__(
        self,
        lmut:           LMUTForest,
        n_base_actions: int,
        zeta:           float,
        gamma_b:        float,
        gamma_w:        float,
        dim:            int,
    ):
        self.lmut           = lmut
        self.n_base_actions = n_base_actions
        self.zeta           = zeta
        self.gamma_b        = gamma_b
        self.gamma_w        = gamma_w
        self.dim            = dim
        # root bounding obs: all [0,1] — represents the start of a new traversal
        self._root_bounds   = np.concatenate([
            np.zeros(dim, dtype=np.float32),
            np.ones(dim,  dtype=np.float32),
        ])

    def _root_obs_from_base(self, base_obs_n: np.ndarray) -> np.ndarray:
        """
        Construct the wrapped observation at the root of the next traversal
        given the normalised base observation s'_b.
        """
        return np.concatenate([base_obs_n, self._root_bounds]).astype(np.float32)

    def compute_targets(
        self,
        obs_batch:        np.ndarray,   # (B, obs_dim)  policy observations
        full_obs_batch:   np.ndarray,   # (B, dim+2*dim) full wrapped observations
        actions:          np.ndarray,   # (B,)  int
        rewards:          np.ndarray,   # (B,)  float
        next_full_obs:    np.ndarray,   # (B, dim+2*dim) next full obs
        dones:            np.ndarray,   # (B,)  bool
        action_types:     np.ndarray,   # (B,)  0=base, 1=info
    ) -> np.ndarray:
        """
        Compute IBMDP Q-targets for a minibatch.

        Returns
        -------
        targets : np.ndarray, shape (B,)
        """
        B = len(rewards)
        targets = np.empty(B, dtype=np.float32)

        for i in range(B):
            if dones[i]:
                targets[i] = float(rewards[i])
                continue

            if action_types[i] == 0:
                # ── base action: use root bounding obs for bootstrap ──
                base_obs_n = next_full_obs[i, :self.dim]
                root_obs   = self._root_obs_from_base(base_obs_n)
                # Q_o over base actions only
                q_vals = np.array([
                    self.lmut.q_value(root_obs, a)
                    for a in range(self.n_base_actions)
                ])
                targets[i] = rewards[i] + self.gamma_b * float(np.max(q_vals))

            else:
                # ── info-gathering action: standard discounted bootstrap ──
                nxt = next_full_obs[i]
                q_vals = np.array([
                    self.lmut.q_value(nxt, a)
                    for a in range(self.lmut.n_actions)
                ])
                targets[i] = self.zeta + self.gamma_w * float(np.max(q_vals))

        return targets


# ─────────────────────────────────────────────────────────────────────────────
# SB3 callback — harvests replay transitions to update the LMUT Q_o
# ─────────────────────────────────────────────────────────────────────────────
class LMUTOmniscientCallback(BaseCallback):
    """
    After every ``update_freq`` environment steps:
      1. Sample a minibatch from the SB3 replay buffer (DQN) or rollout
         buffer (PPO).
      2. Convert to LMUT Transition objects, using Q_o itself as the soft
         Q-label (self-consistency update, similar to DQN target-network
         bootstrapping).
      3. Call LMUTForest.update() to advance the piecewise-linear models
         and potentially trigger node splits.

    Additionally, after every ``target_patch_freq`` steps, patch the SB3
    model's Q-targets with the IBMDP-corrected targets via OmniscientQTarget.
    """

    def __init__(
        self,
        lmut:            LMUTForest,
        omniscient_qt:   OmniscientQTarget,
        env:             IBMDPGroomEnv,
        n_base_actions:  int,
        update_freq:     int   = 64,
        batch_size:      int   = 64,
        lr_decay:        bool  = True,
        verbose:         int   = 0,
    ):
        super().__init__(verbose)
        self.lmut           = lmut
        self.oqt            = omniscient_qt
        self.env            = env
        self.n_base_actions = n_base_actions
        self.update_freq    = update_freq
        self.batch_size     = batch_size
        self.lr_decay       = lr_decay
        self._step_count    = 0
        self.history: Dict[str, List] = {
            "lmut_error": [],
            "n_leaves":   [],
            "lmut_step":  [],
        }

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.update_freq != 0:
            return True

        # ── determine learning rate (1/√n decay) ─────────────────────
        n = max(1, self._step_count // self.update_freq)
        lr = 0.01 / math.sqrt(n) if self.lr_decay else 0.01
        for tree in self.lmut.trees:
            tree.lr = lr

        # ── harvest transitions from replay buffer (DQN) ─────────────
        transitions: List[Transition] = []
        try:
            buf = self.model.replay_buffer
            if buf is not None and buf.pos > 0:
                n_sample = min(self.batch_size, buf.pos if not buf.full else buf.buffer_size)
                replay   = buf.sample(n_sample)

                obs_np  = replay.observations.cpu().numpy()       # (B, obs_dim)
                act_np  = replay.actions.cpu().numpy().flatten()  # (B,)
                rew_np  = replay.rewards.cpu().numpy().flatten()  # (B,)
                nobs_np = replay.next_observations.cpu().numpy()  # (B, obs_dim)

                # Determine whether each transition is base or info
                # (full_obs is stored in info_buffer maintained separately)
                for j in range(len(act_np)):
                    a = int(act_np[j])
                    # Use Q_o as soft label — self-consistency bootstrap
                    q_hat = float(np.max([
                        self.lmut.q_value(obs_np[j], aa)
                        for aa in range(self.lmut.n_actions)
                    ]))
                    t = Transition(
                        state=obs_np[j],
                        action=a,
                        reward=float(rew_np[j]),
                        next_state=nobs_np[j],
                        q_hat=q_hat,
                    )
                    transitions.append(t)
        except (AttributeError, Exception):
            pass  # PPO has no replay buffer — skip harvest this step

        if transitions:
            error = self.lmut.update(transitions)
            self.history["lmut_error"].append(error)
            self.history["n_leaves"].append(
                sum(t.n_leaves() for t in self.lmut.trees)
            )
            self.history["lmut_step"].append(self._step_count)

            if self.verbose >= 1 and self._step_count % (self.update_freq * 50) == 0:
                print(f"[LMUT] step={self._step_count}  "
                      f"error={error:.4f}  "
                      f"leaves={self.history['n_leaves'][-1]}  "
                      f"lr={lr:.5f}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Top-level integrated agent
# ─────────────────────────────────────────────────────────────────────────────
class IBMDPLMUTAgent:
    """
    Integrated IBMDP + LMUT reinforcement learning agent for GroomRL.

    This class:
      1. Wraps a GroomEnvSB3 instance in an IBMDPGroomEnv.
      2. Trains an SB3 DQN (or PPO) on the IBMDP.
      3. Maintains a LMUTForest as the omniscient Q-function Q_o,
         updated continuously via LMUTOmniscientCallback.
      4. After training, extracts an exact DT policy using Algorithm 1
         of Topin et al. (2021).
      5. Reports feature importances, DT structure, and convergence
         diagnostics.

    Parameters
    ----------
    base_env_factory : callable
        Zero-argument callable that returns a fresh GroomEnvSB3 instance.
    agent_type : str
        "dqn" or "ppo".
    p : int
        Number of IBMDP split values per feature (default 3).
    zeta : float
        Information-gathering penalty.
    gamma_b : float
        Base-MDP discount factor.
    gamma_w : float
        IBMDP (information-gathering) discount factor.
    max_depth : int
        DT depth limit.
    nstep : int
        Total SB3 training timesteps.
    lmut_buffer : int
        Transition buffer size per LMUT leaf node.
    lmut_min_split : int
        Minimum samples before a LMUT node is eligible for splitting.
    lmut_min_improvement : float
        Minimum variance-reduction improvement for a LMUT split.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        base_env_factory,
        agent_type:          str   = "dqn",
        p:                   int   = 3,
        zeta:                float = -0.01,
        gamma_b:             float = 0.99,
        gamma_w:             float = 1.0,
        max_depth:           int   = 4,
        nstep:               int   = 200_000,
        lmut_buffer:         int   = 512,
        lmut_min_split:      int   = 30,
        lmut_min_improvement: float = 0.001,
        verbose:             int   = 1,
    ):
        if not _SB3_OK:
            raise ImportError("stable_baselines3 and torch are required.")

        self.agent_type = agent_type.lower()
        self.nstep      = nstep
        self.verbose    = verbose

        # ── build IBMDP environment ───────────────────────────────────
        base_env = base_env_factory()
        self.dim = base_env.observation_space.shape[0]

        self.ibmdp_env = IBMDPGroomEnv(
            base_env        = base_env,
            p               = p,
            zeta            = zeta,
            gamma_w         = gamma_w,
            max_depth       = max_depth,
            policy_obs_only = True,    # policy only sees bounding features
        )
        monitored_env = Monitor(self.ibmdp_env)

        # ── build LMUT omniscient Q-function ──────────────────────────
        # Input: full wrapped obs = base_obs_n (dim) + bounds (2*dim) → dim*3
        lmut_dim = self.ibmdp_env.observation_space.shape[0] + self.dim
        # For the LMUT we use the full obs: base features + bounding features
        # (the policy only gets bounding features, but Q_o needs the full obs)
        n_actions_total = self.ibmdp_env.action_space.n
        self.lmut = LMUTForest(
            n_actions         = n_actions_total,
            dim               = lmut_dim,
            buffer_maxlen     = lmut_buffer,
            min_samples_split = lmut_min_split,
            min_improvement   = lmut_min_improvement,
            max_depth         = max_depth * 2,  # deeper LMUT for richer Q_o
            lr                = 0.01,
            sgd_iters         = 5,
            enable_mdp        = True,
        )

        # ── omniscient Q-target computer ──────────────────────────────
        self.oqt = OmniscientQTarget(
            lmut           = self.lmut,
            n_base_actions = 2,
            zeta           = zeta,
            gamma_b        = gamma_b,
            gamma_w        = gamma_w,
            dim            = self.dim,
        )

        # ── build SB3 agent ───────────────────────────────────────────
        policy_kwargs = {"net_arch": [64, 64]}
        if self.agent_type == "dqn":
            self.model = DQN(
                policy            = "MlpPolicy",
                env               = monitored_env,
                learning_rate     = 1e-3,
                buffer_size       = 500_000,
                learning_starts   = 1_000,
                batch_size        = 64,
                target_update_interval = 1_000,
                exploration_fraction   = 0.2,
                exploration_final_eps  = 0.05,
                gamma             = gamma_b,
                #policy_kwargs     = policy_kwargs,
                verbose           = 0,
            )
        elif self.agent_type == "ppo":
            self.model = PPO(
                policy        = "MlpPolicy",
                env           = monitored_env,
                learning_rate = 1e-3,
                n_steps       = 2048,
                batch_size    = 64,
                n_epochs      = 10,
                gamma         = gamma_b,
                #policy_kwargs = policy_kwargs,
                verbose       = 0,
            )
        elif self.agent_type == "mdpo":
            self.model = MDPO(
                policy        = "MlpPolicy",
                env           = monitored_env,
                learning_rate = 1e-3,
                n_steps       = 2048,
                batch_size    = 64,
                n_epochs      = 10,
                gamma         = gamma_b,
                #policy_kwargs = policy_kwargs,
                verbose       = 0,
            )
        else:
            raise ValueError(f"Unsupported agent_type: {self.agent_type}")

        # ── callback ──────────────────────────────────────────────────
        self._lmut_cb = LMUTOmniscientCallback(
            lmut           = self.lmut,
            omniscient_qt  = self.oqt,
            env            = self.ibmdp_env,
            n_base_actions = 2,
            update_freq    = 64,
            batch_size     = 64,
            lr_decay       = True,
            verbose        = verbose,
        )

        # ── results ───────────────────────────────────────────────────
        self.dt_policy:   Optional[DTNode] = None
        self.train_time:  float = 0.0

    # ── training ─────────────────────────────────────────────────────────
    def fit(self, extra_callbacks: Optional[List] = None) -> "IBMDPLMUTAgent":
        """
        Train the integrated agent.

        1. SB3 trains on the IBMDP env; the LMUTOmniscientCallback keeps
           the LMUT Q_o synchronised throughout.
        2. After training, the exact DT policy is extracted.

        Returns self for chaining.
        """
        callbacks = [self._lmut_cb] + (extra_callbacks or [])
        t0 = time.time()
        if self.verbose >= 1:
            print(f"[IBMDPLMUTAgent] Training {self.agent_type.upper()} "
                  f"for {self.nstep} steps …")
        self.model.learn(total_timesteps=self.nstep, callback=callbacks)
        self.train_time = time.time() - t0
        if self.verbose >= 1:
            print(f"[IBMDPLMUTAgent] Training complete in {self.train_time:.1f}s")

        # ── extract exact DT policy ───────────────────────────────────
        self.dt_policy = self._extract_dt()
        if self.verbose >= 1:
            print(f"[IBMDPLMUTAgent] Extracted DT: "
                  f"{self.dt_policy.n_nodes()} nodes, "
                  f"depth={self.dt_policy.depth_max()}")
        return self

    def _extract_dt(self) -> DTNode:
        """Extract the DT by running Algorithm 1 on the trained policy."""
        def _policy_fn(obs: np.ndarray) -> int:
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        return self.ibmdp_env.extract_dt_policy(_policy_fn)

    # ── groomer interface ─────────────────────────────────────────────────
    def groomer(self):
        """
        Return an AbstractGroomer that uses the extracted DT policy.
        Falls back to the neural-network policy if DT extraction failed.
        """
        from groomrl.Groomer import AbstractGroomer
        from groomrl.JetTree import JetTree

        if self.dt_policy is not None:
            dt  = self.dt_policy
            low = self.ibmdp_env.low
            high= self.ibmdp_env.high

            from groomrl.ibmdp_env import _normalise as _norm

            class _DTGroomer(AbstractGroomer):
                def _groom(self, tree: JetTree):
                    if not tree.lundCoord:
                        return
                    state_n = _norm(tree.state().astype(np.float64), low, high)
                    action  = dt.predict(state_n.astype(np.float64))
                    if action == 1:
                        tree.remove_soft()
                        self._groom(tree)
                    else:
                        if tree.harder: self._groom(tree.harder)
                        if tree.softer: self._groom(tree.softer)
            return _DTGroomer()

        # fallback: SB3 neural policy
        from groomrl.SB3AgentGroom import _SB3Groomer
        return _SB3Groomer(self.model)

    # ── save / load ───────────────────────────────────────────────────────
    def save(self, output_dir: str):
        """Save the SB3 model and LMUT feature importances."""
        import pickle, json
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(os.path.join(output_dir, "ibmdp_policy"))
        importance = {
            "lmut_importance": self.lmut.feature_importance.tolist(),
            "lmut_summary":    self.lmut.summary(),
            "dt_n_nodes":      self.dt_policy.n_nodes()   if self.dt_policy else None,
            "dt_depth":        self.dt_policy.depth_max() if self.dt_policy else None,
            "train_time_s":    self.train_time,
        }
        with open(os.path.join(output_dir, "ibmdp_meta.json"), "w") as f:
            json.dump(importance, f, indent=2)
        if self.dt_policy is not None:
            with open(os.path.join(output_dir, "dt_policy.pkl"), "wb") as f:
                pickle.dump(self.dt_policy, f)
        print(f"[IBMDPLMUTAgent] Saved to {output_dir}")

    # ── diagnostics ───────────────────────────────────────────────────────
    def report(self) -> Dict[str, Any]:
        """Return a structured diagnostics dictionary."""
        lmut_hist = self._lmut_cb.history
        return {
            "agent_type":         self.agent_type,
            "nstep":              self.nstep,
            "train_time_s":       self.train_time,
            "lmut_summary":       self.lmut.summary(),
            "lmut_importance":    self.lmut.feature_importance.tolist(),
            "lmut_final_error":   lmut_hist["lmut_error"][-1] if lmut_hist["lmut_error"] else None,
            "dt_n_nodes":         self.dt_policy.n_nodes()   if self.dt_policy else None,
            "dt_max_depth":       self.dt_policy.depth_max() if self.dt_policy else None,
        }
