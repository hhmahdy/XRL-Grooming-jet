# ibmdp_env.py  –  GroomRL-XRL  |  Iterative Bounding MDP wrapper
"""
Iterative Bounding MDP (IBMDP) environment for GroomRL.

Theory recap (Topin et al., 2021)
----------------------------------
A base MDP ⟨S_b, A_b, T_b, R_b, γ_b⟩ is wrapped into an IBMDP by:

  1. **State augmentation**
     Each base state s_b ∈ ℝ^n is paired with 2n bounding features
     (f₁ˡ, …, fₙˡ, f₁ʰ, …, fₙʰ) ∈ [0,1]^{2n}, representing the current
     lower and upper bounds on each feature value known from prior
     information-gathering actions during this tree traversal.

     s_w = (s_b, f₁ˡ, …, fₙˡ, f₁ʰ, …, fₙʰ)

  2. **Action augmentation**
     A_w = A_b ∪ A_I, where each information-gathering action hc, vᵢ⟩
     compares feature c to a split value vᵢ ∈ {1/(p+1), …, p/(p+1)} and
     tightens the corresponding bound.

  3. **Modified reward**
     Information-gathering actions receive a fixed penalty ζ.
     Base actions receive R_b(s_b, a, s_b').

  4. **Policy masking**
     During training, the policy only observes s_w ∖ s_b (the bounding
     features alone), guaranteeing that any learned policy is equivalent
     to a DT for the base MDP.

  5. **Modified Q-target**
     When a base action is taken, the bootstrap target uses s_{l2}
     (the next state at which a base action is taken) rather than the
     immediate next state, corrected via the omniscient Q-function Q_o.

GroomRL specifics
-----------------
- Base state: s_b = (ln z, ln Δ)  ∈ ℝ²  (or up to 5 Lund coordinates)
- Base actions: A_b = {0=keep, 1=groom}
- Features are normalised to [0,1] using known Lund-coordinate bounds
- p = 3 → 3 × 2 = 6 information-gathering actions (total |A_w| = 8)
- The depth limit is enforced via action masking: once the bounding-path
  depth reaches max_depth, information-gathering actions are disabled.

The class IBMDPGroomEnv is a Gymnasium-compatible environment that wraps any
GroomEnvSB3 instance.  It can be used directly as the env argument to SB3.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Default Lund-coordinate normalisation bounds
# (must match LundCoordinates.__low_full / __high_full in JetTree.py)
# ─────────────────────────────────────────────────────────────────────────────
_LUND_LOW_FULL  = np.array([-10.0, -8.0, -4.0, -1.5708, 0.0])
_LUND_HIGH_FULL = np.array([  0.0,  0.0,  8.0,  1.5708, 8.0])


def _normalise(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map x ∈ [low, high] to [0, 1] (clip for numerical safety)."""
    denom = high - low
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    return np.clip((x - low) / denom, 0.0, 1.0)


def _unnormalise(x_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of _normalise."""
    return x_norm * (high - low) + low


# ─────────────────────────────────────────────────────────────────────────────
# Action index layout
# ─────────────────────────────────────────────────────────────────────────────
def _build_action_map(n_features: int, p: int) -> Tuple[Dict, int]:
    """
    Build a mapping from flat action index → (type, feature, value).

    Layout
    ------
    Indices 0 … |A_b|-1 : base actions
    Indices |A_b| …     : information-gathering actions
                          ordered as (feature 0, split 0), (feature 0, split 1), …

    Returns
    -------
    action_map : dict  index → ("base", action_idx) | ("info", feat, val)
    n_actions  : int   total number of wrapped actions
    """
    action_map: Dict[int, tuple] = {}
    idx = 0
    # base actions: keep (0) and groom (1)
    for a in range(2):
        action_map[idx] = ("base", a)
        idx += 1
    # information-gathering actions
    split_vals = [(i + 1) / (p + 1) for i in range(p)]
    for f in range(n_features):
        for v in split_vals:
            action_map[idx] = ("info", f, v)
            idx += 1
    return action_map, idx


# ─────────────────────────────────────────────────────────────────────────────
# Main IBMDP environment
# ─────────────────────────────────────────────────────────────────────────────
class IBMDPGroomEnv(gymnasium.Env):
    """
    Gymnasium wrapper that turns any GroomEnvSB3 instance into an IBMDP.

    Parameters
    ----------
    base_env : gymnasium.Env
        The wrapped GroomEnvSB3 (or any Gymnasium env with Box obs / Discrete act).
    p : int
        Number of candidate split values per feature (default 3 → values 0.25, 0.5, 0.75).
    zeta : float
        Penalty per information-gathering action (default −0.01).
    gamma_w : float
        Discount factor for information-gathering transitions (default 1.0).
    max_depth : int
        Maximum DT depth (i.e. max number of information-gathering actions per
        base action).  −1 = unlimited.
    policy_obs_only : bool
        If True, the ``observation`` returned to the *policy* contains only
        the bounding features (s_w ∖ s_b).  The full s_w is accessible via
        ``env.full_obs`` for the omniscient Q-function.
    lund_low  : np.ndarray, optional
        Override feature lower bounds (default: first ``dim`` entries of
        _LUND_LOW_FULL).
    lund_high : np.ndarray, optional
        Override feature upper bounds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env:         gymnasium.Env,
        p:                int   = 3,
        zeta:             float = -0.01,
        gamma_w:          float = 1.0,
        max_depth:        int   = -1,
        policy_obs_only:  bool  = True,
        lund_low:         Optional[np.ndarray] = None,
        lund_high:        Optional[np.ndarray] = None,
    ):
        super().__init__()
        self._base_env       = base_env
        self.p               = p
        self.zeta            = float(zeta)
        self.gamma_w         = float(gamma_w)
        self.max_depth       = max_depth
        self.policy_obs_only = policy_obs_only

        # ── feature bounds ────────────────────────────────────────────
        self.dim = base_env.observation_space.shape[0]
        self.low  = lund_low  if lund_low  is not None else _LUND_LOW_FULL[:self.dim]
        self.high = lund_high if lund_high is not None else _LUND_HIGH_FULL[:self.dim]

        # ── action space ──────────────────────────────────────────────
        self._action_map, n_actions_total = _build_action_map(self.dim, p)
        self.n_base_actions = 2
        self.n_info_actions = n_actions_total - self.n_base_actions
        self.action_space   = spaces.Discrete(n_actions_total)

        # ── observation spaces ────────────────────────────────────────
        # Bounding features: 2*dim values in [0,1]
        bound_low  = np.zeros(2 * self.dim, dtype=np.float32)
        bound_high = np.ones(2 * self.dim,  dtype=np.float32)

        if policy_obs_only:
            # Policy sees only bounding features
            self.observation_space = spaces.Box(bound_low, bound_high, dtype=np.float32)
        else:
            # Policy sees full wrapped state
            full_low  = np.concatenate([np.zeros(self.dim, dtype=np.float32), bound_low])
            full_high = np.concatenate([np.ones(self.dim, dtype=np.float32), bound_high])
            self.observation_space = spaces.Box(full_low, full_high, dtype=np.float32)

        # ── internal state ────────────────────────────────────────────
        self._base_obs:    np.ndarray = np.zeros(self.dim,         dtype=np.float32)
        self._base_obs_n:  np.ndarray = np.zeros(self.dim,         dtype=np.float32)
        self._bounds_l:    np.ndarray = np.zeros(self.dim,         dtype=np.float32)
        self._bounds_h:    np.ndarray = np.ones(self.dim,          dtype=np.float32)
        self._tree_depth:  int        = 0
        self._done:        bool       = False

        # Full obs for omniscient Q-function (always maintained regardless of mode)
        self.full_obs: np.ndarray = np.zeros(self.dim + 2 * self.dim, dtype=np.float32)

    # ── helpers ───────────────────────────────────────────────────────────
    def _make_policy_obs(self) -> np.ndarray:
        bounds = np.concatenate([self._bounds_l, self._bounds_h]).astype(np.float32)
        if self.policy_obs_only:
            return bounds
        return np.concatenate([self._base_obs_n, bounds]).astype(np.float32)

    def _update_full_obs(self):
        bounds = np.concatenate([self._bounds_l, self._bounds_h])
        self.full_obs = np.concatenate([self._base_obs_n, bounds]).astype(np.float32)

    def _reset_bounds(self):
        """Reset bounding features to [0, 1] (root of a new DT traversal)."""
        self._bounds_l    = np.zeros(self.dim, dtype=np.float32)
        self._bounds_h    = np.ones(self.dim,  dtype=np.float32)
        self._tree_depth  = 0

    def _action_mask(self) -> np.ndarray:
        """
        Binary mask over the action space.
        Info-gathering actions are disabled when max_depth is reached.
        Base actions are always enabled (unless the base env is done).
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        if self.max_depth >= 0 and self._tree_depth >= self.max_depth:
            # disable all information-gathering actions
            for idx, adef in self._action_map.items():
                if adef[0] == "info":
                    mask[idx] = False
        return mask

    # ── Gymnasium API ─────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        base_obs, info = self._base_env.reset(seed=seed, options=options)
        self._base_obs   = base_obs.astype(np.float32)
        self._base_obs_n = _normalise(base_obs, self.low, self.high).astype(np.float32)
        self._reset_bounds()
        self._done = False
        self._update_full_obs()
        obs = self._make_policy_obs()
        info["action_mask"] = self._action_mask()
        info["full_obs"]    = self.full_obs.copy()
        return obs, info

    def step(self, action: int):
        adef = self._action_map[action]

        if adef[0] == "info":
            # ── information-gathering action ──────────────────────────
            _, feat, v_norm = adef
            # un-normalise v_norm relative to current bounds
            v_proj = v_norm * (self._bounds_h[feat] - self._bounds_l[feat]) + self._bounds_l[feat]
            # compare to normalised base feature
            if self._base_obs_n[feat] <= v_proj:
                self._bounds_h[feat] = min(self._bounds_h[feat], v_proj)
            else:
                self._bounds_l[feat] = max(self._bounds_l[feat], v_proj)
            self._tree_depth += 1
            self._update_full_obs()

            obs     = self._make_policy_obs()
            reward  = self.zeta
            term    = False
            trunc   = False
            info    = {
                "action_mask":  self._action_mask(),
                "full_obs":     self.full_obs.copy(),
                "action_type":  "info",
                "split_feat":   feat,
                "split_val":    v_proj,
            }
            return obs, reward, term, trunc, info

        else:
            # ── base action ───────────────────────────────────────────
            _, base_a = adef
            base_obs_new, base_reward, terminated, truncated, base_info = \
                self._base_env.step(base_a)

            # reset bounding features for next traversal
            self._base_obs   = base_obs_new.astype(np.float32)
            self._base_obs_n = _normalise(base_obs_new, self.low, self.high).astype(np.float32)
            self._reset_bounds()
            self._update_full_obs()
            self._done = terminated or truncated

            obs    = self._make_policy_obs()
            info   = {
                **base_info,
                "action_mask":  self._action_mask(),
                "full_obs":     self.full_obs.copy(),
                "action_type":  "base",
                "base_reward":  base_reward,
            }
            return obs, base_reward, terminated, truncated, info

    def render(self):
        return self._base_env.render()

    def close(self):
        self._base_env.close()

    # ── DT extraction helper ──────────────────────────────────────────────
    def extract_dt_policy(self, policy_fn) -> "DTNode":
        """
        Run Algorithm 1 from Topin et al. (2021):
        recursively call the policy on bounding-feature observations to build
        the full DT for the base MDP.

        Parameters
        ----------
        policy_fn : callable
            Maps a bounding-feature obs (np.ndarray) → action index (int).

        Returns
        -------
        DTNode : root of the extracted decision tree.
        """
        root_bounds_l = np.zeros(self.dim, dtype=np.float64)
        root_bounds_h = np.ones(self.dim,  dtype=np.float64)
        return _subtree_from_policy(
            policy_fn, self._action_map, root_bounds_l, root_bounds_h,
            self.dim, max_depth=self.max_depth, depth=0
        )


# ─────────────────────────────────────────────────────────────────────────────
# DT extraction (Algorithm 1, Topin et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────
class DTNode:
    """Node of an extracted Decision Tree policy."""
    __slots__ = ("action", "feat", "val", "left", "right", "depth")

    def __init__(self, action=None, feat=None, val=None, left=None, right=None, depth=0):
        self.action = action   # int if leaf, None if internal
        self.feat   = feat     # split feature index (internal only)
        self.val    = val      # split value in normalised space (internal only)
        self.left   = left     # DTNode (feature < val)
        self.right  = right    # DTNode (feature ≥ val)
        self.depth  = depth

    @property
    def is_leaf(self) -> bool:
        return self.action is not None

    def predict(self, state_normalised: np.ndarray) -> int:
        """Traverse the DT and return the base action for the given (normalised) state."""
        node = self
        while not node.is_leaf:
            if state_normalised[node.feat] < node.val:
                node = node.left
            else:
                node = node.right
        return node.action

    def n_nodes(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + self.left.n_nodes() + self.right.n_nodes()

    def depth_max(self) -> int:
        if self.is_leaf:
            return self.depth
        return max(self.left.depth_max(), self.right.depth_max())

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf(action={self.action}, depth={self.depth})"
        return (f"InternalNode(feat={self.feat}, val={self.val:.4f}, "
                f"depth={self.depth}, n={self.n_nodes()})")


def _subtree_from_policy(
    policy_fn,
    action_map: Dict[int, tuple],
    bounds_l:   np.ndarray,
    bounds_h:   np.ndarray,
    dim:        int,
    max_depth:  int,
    depth:      int,
) -> DTNode:
    """Recursive Algorithm 1 implementation."""
    obs = np.concatenate([bounds_l, bounds_h]).astype(np.float32)

    # mask info-gathering actions if depth limit reached
    if max_depth >= 0 and depth >= max_depth:
        # force a base action
        # query policy but only accept base actions
        raw = policy_fn(obs)
        adef = action_map.get(int(raw), ("base", 0))
        base_a = adef[1] if adef[0] == "base" else 0
        return DTNode(action=base_a, depth=depth)

    raw  = int(policy_fn(obs))
    adef = action_map[raw]

    if adef[0] == "base":
        return DTNode(action=adef[1], depth=depth)

    # information-gathering action: recurse into both children
    _, feat, v_norm = adef
    v_proj = v_norm * (bounds_h[feat] - bounds_l[feat]) + bounds_l[feat]

    bounds_l_left  = bounds_l.copy()
    bounds_h_left  = bounds_h.copy()
    bounds_h_left[feat] = min(bounds_h[feat], v_proj)

    bounds_l_right = bounds_l.copy()
    bounds_h_right = bounds_h.copy()
    bounds_l_right[feat] = max(bounds_l[feat], v_proj)

    left  = _subtree_from_policy(policy_fn, action_map, bounds_l_left,  bounds_h_left,  dim, max_depth, depth + 1)
    right = _subtree_from_policy(policy_fn, action_map, bounds_l_right, bounds_h_right, dim, max_depth, depth + 1)

    return DTNode(feat=feat, val=v_proj, left=left, right=right, depth=depth)
