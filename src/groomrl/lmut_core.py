# lmut_core.py  –  GroomRL-XRL  |  Linear Model U-Tree data structures
"""
Low-level data structures for Linear Model U-Trees (LMUTs).

Design principles
-----------------
* Each ``LMUTNode`` owns its own weight vector (one weight per input feature
  plus a bias), a circular buffer of recent transitions, and optional MDP
  bookkeeping (transition counts, average rewards).
* Splitting is triggered by the *variance-reduction* criterion of Liu et al.
  (2018), weighted by the linear-model weight magnitudes (Equation 3 of that
  paper) to produce the composite importance measure used in the paper.
* The tree is stored as a flat array of nodes with left/right child indices so
  it can be serialised trivially and traversed without Python recursion.
* A ``LMUTForest`` wraps one tree per action, matching the per-action LMUT
  structure of the original paper.

Notation
--------
For a leaf node N with feature weights w_N ∈ ℝ^J:

    Q^UT(s | w_N, a) = sum_j s_j * w_{Nj} + w_{N0}        (linear model)

    Inf_f^N = (1 + |w_{Nf}|² / sum_j |w_{Nj}|²)
              * (var_N - sum_c (Num_c / sum_i Num_i) * var_c)
                                                            (Eq. 3, Liu 2018)

    Inf_f = sum_{N splits on f} Inf_f^N                     (global importance)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_UNSET = -1          # sentinel for absent child/parent indices
_LEAF_WEIGHT_INIT_SCALE = 0.01   # small random init for leaf linear models


# ─────────────────────────────────────────────────────────────────────────────
# Transition tuple
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Transition:
    """One (s, a, r, s', Q̂) experience tuple."""
    state:      np.ndarray     # shape (dim,)
    action:     int
    reward:     float
    next_state: np.ndarray     # shape (dim,)
    q_hat:      float          # soft Q-label from the neural-network oracle


# ─────────────────────────────────────────────────────────────────────────────
# Single tree node
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LMUTNode:
    """
    One node of a Linear Model U-Tree.

    Leaf nodes hold:
      - ``weights``: linear model coefficients (size = dim + 1, bias last)
      - ``buffer``:  circular buffer of recent transitions
      - Optional MDP bookkeeping: transition counts, reward sums

    Internal nodes hold:
      - ``split_feature``: which input feature is compared
      - ``split_value``:   the threshold for the comparison
      - ``left`` / ``right``: indices into the owning LMUTTree's node list
        (left ↔ feature < threshold,  right ↔ feature ≥ threshold)
    """
    node_id:        int
    dim:            int                     # state dimension
    is_leaf:        bool        = True
    depth:          int         = 0

    # ── linear model (leaf only) ──────────────────────────────────────────
    weights:        np.ndarray  = field(default=None)    # (dim+1,) bias last
    train_error:    float       = float("inf")

    # ── transition buffer (leaf only) ─────────────────────────────────────
    buffer:         deque       = field(default_factory=lambda: deque(maxlen=512))
    n_updates:      int         = 0

    # ── MDP bookkeeping (leaf only, optional) ─────────────────────────────
    # counts[(s_disc, a, s'_disc)] = int
    # rewards[(s_disc, a, s'_disc)] = float (running mean)
    # q_avg[(s_disc, a)]            = float (running mean of Q̂)
    counts:         Dict        = field(default_factory=dict)
    rewards_sum:    Dict        = field(default_factory=dict)
    q_avg:          Dict        = field(default_factory=dict)

    # ── split info (internal only) ────────────────────────────────────────
    split_feature:  int         = _UNSET
    split_value:    float       = 0.0
    left:           int         = _UNSET   # index into LMUTTree.nodes
    right:          int         = _UNSET

    # ── importance accumulation ──────────────────────────────────────────
    importance_contribution: float = 0.0  # sum of Inf_f^N across splits on this node

    def __post_init__(self):
        if self.weights is None:
            rng = np.random.default_rng()
            self.weights = rng.normal(0, _LEAF_WEIGHT_INIT_SCALE, self.dim + 1)

    # ── prediction ───────────────────────────────────────────────────────
    def predict(self, state: np.ndarray) -> float:
        """Linear Q-value prediction:  Q = w·s + bias."""
        return float(np.dot(state, self.weights[:-1]) + self.weights[-1])

    # ── SGD update ───────────────────────────────────────────────────────
    def sgd_update(self, state: np.ndarray, target: float, lr: float = 0.01) -> float:
        """One SGD step on the squared loss; returns the residual squared error."""
        pred  = self.predict(state)
        error = target - pred
        # gradient: d/dw (0.5*(target-pred)²) = -(target-pred)*[s; 1]
        grad  = np.append(state, 1.0)
        self.weights += lr * error * grad
        return error ** 2

    # ── buffer management ────────────────────────────────────────────────
    def add_transition(self, t: Transition):
        self.buffer.append(t)

    def get_q_values(self) -> np.ndarray:
        """Return Q̂ values for all buffered transitions (used by splitting)."""
        return np.array([t.q_hat for t in self.buffer])

    # ── MDP bookkeeping ──────────────────────────────────────────────────
    def update_mdp(self, s_key, a: int, sp_key, reward: float, q_hat: float):
        """Incrementally update MDP statistics on this leaf."""
        k = (s_key, a, sp_key)
        self.counts[k]      = self.counts.get(k, 0) + 1
        n                   = self.counts[k]
        prev_r              = self.rewards_sum.get(k, 0.0)
        self.rewards_sum[k] = prev_r + (reward - prev_r) / n

        qa_key = (s_key, a)
        prev_q = self.q_avg.get(qa_key, 0.0)
        cnt_qa = sum(v for (s, aa, _), v in self.counts.items() if s == s_key and aa == a)
        self.q_avg[qa_key] = prev_q + (q_hat - prev_q) / max(cnt_qa, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Tree for one action
# ─────────────────────────────────────────────────────────────────────────────
class LMUTTree:
    """
    Linear Model U-Tree for a single action ``a``.

    The tree is stored as a flat list ``self.nodes`` indexed by integer node IDs.
    Node 0 is always the root.

    Parameters
    ----------
    dim : int
        Dimensionality of the state / feature vector.
    buffer_maxlen : int
        Maximum number of transitions stored per leaf node.
    min_samples_split : int
        Minimum number of buffered samples required before a split is attempted.
    min_improvement : float
        Minimum variance-reduction improvement required to accept a split.
    max_depth : int
        Maximum allowable tree depth.  −1 = unlimited.
    lr : float
        SGD learning rate for weight updates.
    sgd_iters : int
        Number of SGD passes over the minibatch per update call.
    enable_mdp : bool
        Whether to maintain per-leaf MDP statistics.
    """

    def __init__(
        self,
        dim:               int,
        buffer_maxlen:     int   = 512,
        min_samples_split: int   = 30,
        min_improvement:   float = 0.001,
        max_depth:         int   = -1,
        lr:                float = 0.01,
        sgd_iters:         int   = 5,
        enable_mdp:        bool  = False,
    ):
        self.dim               = dim
        self.buffer_maxlen     = buffer_maxlen
        self.min_samples_split = min_samples_split
        self.min_improvement   = min_improvement
        self.max_depth         = max_depth
        self.lr                = lr
        self.sgd_iters         = sgd_iters
        self.enable_mdp        = enable_mdp

        # Flat node store
        self.nodes: List[LMUTNode] = [LMUTNode(node_id=0, dim=dim)]
        self._feature_importance = np.zeros(dim)   # accumulated across all splits

    # ── routing ──────────────────────────────────────────────────────────
    def route(self, state: np.ndarray) -> LMUTNode:
        """Descend the tree and return the leaf node for ``state``."""
        node = self.nodes[0]
        while not node.is_leaf:
            if state[node.split_feature] < node.split_value:
                node = self.nodes[node.left]
            else:
                node = self.nodes[node.right]
        return node

    def route_path(self, state: np.ndarray) -> List[int]:
        """Return the list of node IDs from root to the leaf for ``state``."""
        path = []
        node = self.nodes[0]
        while True:
            path.append(node.node_id)
            if node.is_leaf:
                break
            if state[node.split_feature] < node.split_value:
                node = self.nodes[node.left]
            else:
                node = self.nodes[node.right]
        return path

    # ── prediction ───────────────────────────────────────────────────────
    def predict(self, state: np.ndarray) -> float:
        return self.route(state).predict(state)

    # ── data gathering ───────────────────────────────────────────────────
    def add_transition(self, t: Transition):
        """Route transition to the correct leaf and store it."""
        node = self.route(t.state)
        node.add_transition(t)
        if self.enable_mdp:
            s_key  = self._discretise(t.state,  node)
            sp_key = self._discretise(t.next_state, self.route(t.next_state))
            node.update_mdp(s_key, t.action, sp_key, t.reward, t.q_hat)

    def _discretise(self, state: np.ndarray, leaf: LMUTNode) -> tuple:
        """Return a hashable key for state inside leaf (centroid bucket)."""
        buf = list(leaf.buffer)
        if len(buf) == 0:
            return tuple(np.round(state, 2))
        # assign to nearest centroid among buffered states
        states_arr = np.array([b.state for b in buf])
        dists = np.linalg.norm(states_arr - state, axis=1)
        return int(dists.argmin())

    # ── SGD weight update ────────────────────────────────────────────────
    def sgd_update_leaf(self, node: LMUTNode) -> float:
        """
        Run ``sgd_iters`` passes of SGD over all transitions buffered on ``node``.
        Returns final mean squared error.
        """
        buf = list(node.buffer)
        if not buf:
            return float("inf")
        errors = []
        for _ in range(self.sgd_iters):
            np.random.shuffle(buf)
            for t in buf:
                sq = node.sgd_update(t.state, t.q_hat, self.lr)
                errors.append(sq)
        node.train_error = float(np.mean(errors[-len(buf):]))
        node.n_updates  += 1
        return node.train_error

    # ── splitting ────────────────────────────────────────────────────────
    def _variance_reduction(
        self,
        parent_qs: np.ndarray,
        left_qs:   np.ndarray,
        right_qs:  np.ndarray,
    ) -> float:
        """
        Variance-reduction splitting criterion (Liu 2018).
        Returns the reduction in Q-value variance achieved by the split.
        """
        if len(left_qs) == 0 or len(right_qs) == 0:
            return -float("inf")
        n_total = len(parent_qs)
        var_p   = float(np.var(parent_qs))
        var_l   = float(np.var(left_qs))
        var_r   = float(np.var(right_qs))
        weighted_child_var = (len(left_qs) * var_l + len(right_qs) * var_r) / n_total
        return var_p - weighted_child_var

    def _importance_weight(self, node: LMUTNode, feat: int) -> float:
        """
        Weight-based importance multiplier (first factor of Eq. 3, Liu 2018).
        """
        w_sq   = node.weights[:-1] ** 2
        total  = w_sq.sum()
        if total < 1e-12:
            return 1.0
        return 1.0 + w_sq[feat] / total

    def try_split(self, node: LMUTNode) -> bool:
        """
        Attempt to split a leaf node.  Returns True if a split was performed.

        Scans all (feature, value) candidate pairs using the transitions in
        ``node.buffer`` and applies the weighted variance-reduction criterion.
        """
        buf = list(node.buffer)
        if len(buf) < self.min_samples_split:
            return False
        if self.max_depth >= 0 and node.depth >= self.max_depth:
            return False

        q_vals   = np.array([t.q_hat for t in buf])
        states   = np.array([t.state  for t in buf])
        var_node = float(np.var(q_vals))
        if var_node < 1e-8:
            return False   # no variance to reduce

        best_score   = self.min_improvement
        best_feat    = _UNSET
        best_val     = 0.0
        best_left_ix: List[int] = []
        best_right_ix: List[int] = []

        for f in range(self.dim):
            col     = states[:, f]
            vals    = np.unique(col)
            # use midpoints between consecutive unique values as candidate thresholds
            candidates = (vals[:-1] + vals[1:]) / 2.0 if len(vals) > 1 else []
            for v in candidates:
                left_ix  = np.where(col < v)[0]
                right_ix = np.where(col >= v)[0]
                vr = self._variance_reduction(q_vals, q_vals[left_ix], q_vals[right_ix])
                score = self._importance_weight(node, f) * vr
                if score > best_score:
                    best_score    = score
                    best_feat     = f
                    best_val      = v
                    best_left_ix  = left_ix.tolist()
                    best_right_ix = right_ix.tolist()

        if best_feat == _UNSET:
            return False   # no improvement found

        # ── perform the split ────────────────────────────────────────────
        left_id  = len(self.nodes)
        right_id = left_id + 1

        left_node  = LMUTNode(node_id=left_id,  dim=self.dim,
                               depth=node.depth + 1,
                               buffer=deque(maxlen=self.buffer_maxlen))
        right_node = LMUTNode(node_id=right_id, dim=self.dim,
                               depth=node.depth + 1,
                               buffer=deque(maxlen=self.buffer_maxlen))

        # inherit parent weights (stagewise approach, Liu 2018 §4.2)
        left_node.weights  = node.weights.copy()
        right_node.weights = node.weights.copy()

        # distribute buffered transitions to children
        for idx, t in enumerate(buf):
            if idx in best_left_ix:
                left_node.buffer.append(t)
            else:
                right_node.buffer.append(t)

        # run initial SGD on children
        self.sgd_update_leaf(left_node)
        self.sgd_update_leaf(right_node)

        self.nodes.append(left_node)
        self.nodes.append(right_node)

        # convert parent to internal node
        node.is_leaf       = False
        node.split_feature = best_feat
        node.split_value   = best_val
        node.left          = left_id
        node.right         = right_id
        node.buffer        = deque(maxlen=0)   # free memory

        # accumulate global feature importance
        self._feature_importance[best_feat] += best_score

        return True

    # ── batch update (one minibatch) ────────────────────────────────────
    def update(self, transitions: List[Transition]) -> float:
        """
        Algorithm 1 of Liu et al. (2018):
          Part I  – route each transition to a leaf, add to buffer
          Part II – for each leaf: SGD update; attempt split if needed

        Returns mean leaf training error across all updated leaves.
        """
        # Part I: data gathering
        for t in transitions:
            self.add_transition(t)

        # Part II: node splitting phase
        errors = []
        leaves = [n for n in self.nodes if n.is_leaf]
        for node in leaves:
            if len(node.buffer) == 0:
                continue
            err = self.sgd_update_leaf(node)
            if err <= self.min_improvement * 2 or node.n_updates % 20 == 0:
                self.try_split(node)
            errors.append(err)

        return float(np.mean(errors)) if errors else float("inf")

    # ── feature importance ───────────────────────────────────────────────
    @property
    def feature_importance(self) -> np.ndarray:
        """Normalised feature-importance scores (sum to 1)."""
        total = self._feature_importance.sum()
        if total < 1e-12:
            return np.ones(self.dim) / self.dim
        return self._feature_importance / total

    # ── serialisation ────────────────────────────────────────────────────
    def n_leaves(self) -> int:
        return sum(1 for n in self.nodes if n.is_leaf)

    def depth(self) -> int:
        return max((n.depth for n in self.nodes), default=0)

    def __repr__(self) -> str:
        return (f"LMUTTree(dim={self.dim}, nodes={len(self.nodes)}, "
                f"leaves={self.n_leaves()}, depth={self.depth()})")


# ─────────────────────────────────────────────────────────────────────────────
# Per-action forest
# ─────────────────────────────────────────────────────────────────────────────
class LMUTForest:
    """
    One ``LMUTTree`` per action.  This is the public API for Q-function
    approximation used by the rest of the GroomRL-XRL system.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions (2 for GroomRL: keep / groom).
    dim : int
        State-vector dimensionality.
    **tree_kwargs :
        Forwarded to each ``LMUTTree`` constructor.
    """

    def __init__(self, n_actions: int, dim: int, **tree_kwargs):
        self.n_actions = n_actions
        self.dim       = dim
        self.trees: List[LMUTTree] = [
            LMUTTree(dim=dim, **tree_kwargs) for _ in range(n_actions)
        ]

    # ── Q-value interface ────────────────────────────────────────────────
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for all actions as a (n_actions,) array."""
        return np.array([t.predict(state) for t in self.trees])

    def predict_action(self, state: np.ndarray) -> int:
        """Greedy action selection."""
        return int(np.argmax(self.q_values(state)))

    def q_value(self, state: np.ndarray, action: int) -> float:
        return self.trees[action].predict(state)

    # ── training ────────────────────────────────────────────────────────
    def update(self, transitions: List[Transition]) -> float:
        """Route each transition to the correct per-action tree and update."""
        by_action: List[List[Transition]] = [[] for _ in range(self.n_actions)]
        for t in transitions:
            if 0 <= t.action < self.n_actions:
                by_action[t.action].append(t)
        errors = [self.trees[a].update(by_action[a]) for a in range(self.n_actions)]
        return float(np.mean([e for e in errors if math.isfinite(e)]))

    # ── diagnostics ─────────────────────────────────────────────────────
    @property
    def feature_importance(self) -> np.ndarray:
        """Mean feature importance across all per-action trees."""
        return np.mean([t.feature_importance for t in self.trees], axis=0)

    def summary(self) -> Dict:
        return {
            f"action_{a}": {
                "n_nodes":  len(tree.nodes),
                "n_leaves": tree.n_leaves(),
                "depth":    tree.depth(),
            }
            for a, tree in enumerate(self.trees)
        }

    def __repr__(self) -> str:
        return f"LMUTForest(n_actions={self.n_actions}, dim={self.dim})"
