# convergence_metrics.py  –  GroomRL-XRL
"""
Convergence monitoring, stability safeguards, and evaluation metrics
for the integrated IBMDP + LMUT system.

Module structure
----------------
StabilityGuard       – Early stopping / LR reduction on Q-value divergence
ConvergenceMonitor   – Windowed statistics for policy Q-values and LMUT error
EvaluationSuite      – Physics-motivated metrics for GroomRL
MetricsLogger        – Unified logger; writes CSV + JSON summaries
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_OK = True
except ImportError:
    _SB3_OK = False


# ═══════════════════════════════════════════════════════════════════════════
#  1. Stability Guard
# ═══════════════════════════════════════════════════════════════════════════
class StabilityGuard:
    """
    Detects and mitigates Q-value divergence during IBMDP training.

    Three complementary mechanisms:

    1. **Q-value clipping** – absolute cap on |Q̂| to prevent gradient
       explosions when the omniscient LMUT target temporarily overshoots.

    2. **Rolling-variance alarm** – if the per-step variance of the max-Q
       over a sliding window exceeds ``var_threshold``, trigger a learning
       rate reduction by factor ``lr_decay_factor``.

    3. **Gradient clipping** – applied inside the SB3 policy_kwargs via the
       ``max_grad_norm`` parameter; value here is used for reporting only.

    Parameters
    ----------
    q_clip : float
        Maximum absolute Q-value.  Default 100.
    var_window : int
        Window size for rolling variance.  Default 200.
    var_threshold : float
        Variance threshold that triggers LR reduction.  Default 50.
    lr_decay_factor : float
        Multiplicative LR reduction factor.  Default 0.5.
    min_lr : float
        Minimum learning rate.  Default 1e-5.
    max_grad_norm : float
        Gradient clipping norm (reported; set in SB3 policy_kwargs).
    """

    def __init__(
        self,
        q_clip:          float = 100.0,
        var_window:      int   = 200,
        var_threshold:   float = 50.0,
        lr_decay_factor: float = 0.5,
        min_lr:          float = 1e-5,
        max_grad_norm:   float = 10.0,
    ):
        self.q_clip          = q_clip
        self.var_window      = var_window
        self.var_threshold   = var_threshold
        self.lr_decay_factor = lr_decay_factor
        self.min_lr          = min_lr
        self.max_grad_norm   = max_grad_norm
        self._q_window       = deque(maxlen=var_window)
        self.n_triggers      = 0

    def clip_q(self, q: float) -> float:
        return float(np.clip(q, -self.q_clip, self.q_clip))

    def record_q(self, max_q: float) -> bool:
        """Record a max-Q observation.  Returns True if alarm was triggered."""
        self._q_window.append(float(max_q))
        if len(self._q_window) < self.var_window:
            return False
        variance = float(np.var(list(self._q_window)))
        if variance > self.var_threshold:
            self.n_triggers += 1
            return True
        return False

    def reduce_lr(self, current_lr: float) -> float:
        """Return the reduced learning rate (respecting minimum)."""
        return max(self.min_lr, current_lr * self.lr_decay_factor)

    def policy_kwargs_additions(self) -> dict:
        """Return policy_kwargs additions to pass to SB3."""
        return {"optimizer_kwargs": {"eps": 1e-5}}


# ═══════════════════════════════════════════════════════════════════════════
#  2. Convergence Monitor
# ═══════════════════════════════════════════════════════════════════════════
class ConvergenceMonitor:
    """
    Tracks and analyses convergence signals during joint IBMDP + LMUT training.

    Convergence criteria (all must be met for ``has_converged`` to return True):

    a. **Policy stability** – the absolute change in greedy action probabilities
       over consecutive evaluation windows falls below ``action_change_threshold``.
    b. **LMUT error plateau** – the smoothed LMUT training error has not improved
       by more than ``lmut_tol`` for ``patience`` consecutive evaluation windows.
    c. **DT structure stability** – the number of DT nodes has not changed in
       ``dt_stability_window`` consecutive extractions.

    Parameters
    ----------
    eval_env : gymnasium.Env
        Environment used for policy evaluation.
    eval_states : np.ndarray, shape (N, obs_dim)
        Fixed set of bounding-feature observations for action-stability checks.
    eval_freq : int
        Frequency (in environment steps) of convergence checks.
    action_change_threshold : float
        Max allowed fraction of states whose greedy action changes.
    lmut_tol : float
        Minimum relative improvement in LMUT error to count as progress.
    patience : int
        Number of evaluation windows without LMUT improvement before declaring plateau.
    dt_stability_window : int
        Number of windows of stable DT node count.
    """

    def __init__(
        self,
        eval_env,
        eval_states:              np.ndarray,
        eval_freq:                int   = 2000,
        action_change_threshold:  float = 0.02,
        lmut_tol:                 float = 1e-3,
        patience:                 int   = 5,
        dt_stability_window:      int   = 3,
    ):
        self.eval_env                 = eval_env
        self.eval_states              = eval_states
        self.eval_freq                = eval_freq
        self.action_change_threshold  = action_change_threshold
        self.lmut_tol                 = lmut_tol
        self.patience                 = patience
        self.dt_stability_window      = dt_stability_window

        self._prev_actions:  Optional[np.ndarray] = None
        self._best_lmut_err: float                = float("inf")
        self._lmut_plateau:  int                  = 0
        self._dt_node_hist:  deque                = deque(maxlen=dt_stability_window)

        self.history: Dict[str, List] = {
            "step":               [],
            "action_change_frac": [],
            "lmut_error":         [],
            "dt_n_nodes":         [],
            "converged":          [],
        }

    def check(
        self,
        step:       int,
        policy_fn:  Callable,
        lmut_error: float,
        dt_node_count: Optional[int] = None,
    ) -> bool:
        """
        Run a convergence check.  Returns True if all criteria are met.

        Parameters
        ----------
        step : int
            Current environment step.
        policy_fn : callable
            Maps obs → action (deterministic).
        lmut_error : float
            Current LMUT training error.
        dt_node_count : int, optional
            Current extracted DT node count.
        """
        # ── a. policy stability ──────────────────────────────────────
        actions = np.array([policy_fn(s) for s in self.eval_states])
        if self._prev_actions is not None:
            changed = float(np.mean(actions != self._prev_actions))
        else:
            changed = 1.0
        self._prev_actions = actions.copy()

        # ── b. LMUT error plateau ────────────────────────────────────
        if math.isfinite(lmut_error):
            if self._best_lmut_err - lmut_error > self.lmut_tol * self._best_lmut_err:
                self._best_lmut_err = lmut_error
                self._lmut_plateau  = 0
            else:
                self._lmut_plateau += 1
        lmut_converged = self._lmut_plateau >= self.patience

        # ── c. DT structure stability ────────────────────────────────
        if dt_node_count is not None:
            self._dt_node_hist.append(dt_node_count)
        dt_stable = (
            len(self._dt_node_hist) == self.dt_stability_window
            and len(set(self._dt_node_hist)) == 1
        )

        converged = (
            changed          <= self.action_change_threshold
            and lmut_converged
            and dt_stable
        )

        self.history["step"].append(step)
        self.history["action_change_frac"].append(changed)
        self.history["lmut_error"].append(lmut_error)
        self.history["dt_n_nodes"].append(dt_node_count)
        self.history["converged"].append(converged)
        return converged


# ═══════════════════════════════════════════════════════════════════════════
#  3. Evaluation Suite
# ═══════════════════════════════════════════════════════════════════════════
class EvaluationSuite:
    """
    Physics-motivated and RL-standard metrics for GroomRL-XRL.

    Metrics
    -------
    reward_metrics        – mean/std of per-episode return
    mass_metrics          – groomed jet mass peak position and width
    physics_alignment     – Spearman ρ between policy grooming probability
                            and Soft Drop indicator function
    dt_metrics            – DT node count, depth, exact-fidelity flag
    lmut_metrics          – feature importances, leaf count, MAE
    efficiency_metrics    – training time, steps/second, memory usage
    """

    def __init__(
        self,
        feature_names:  List[str],
        zcut:           float = 0.05,
        beta:           float = 1.0,
        R0:             float = 1.0,
    ):
        self.feature_names = feature_names
        self.zcut          = zcut
        self.beta          = beta
        self.R0            = R0

    # ── physics alignment ─────────────────────────────────────────────────
    def soft_drop_indicator(
        self,
        states: np.ndarray,
        lnz_idx: int = 0,
        lnDelta_idx: int = 1,
    ) -> np.ndarray:
        """
        Binary indicator I[z < z_cut * (Δ/R₀)^β] for each state row.
        States are assumed to contain raw (un-normalised) Lund coordinates.
        """
        z     = np.exp(states[:, lnz_idx])
        delta = np.exp(states[:, lnDelta_idx])
        return (z < self.zcut * (delta / self.R0) ** self.beta).astype(float)

    def physics_alignment(
        self,
        states:  np.ndarray,
        actions: np.ndarray,
        lnz_idx: int = 0,
        lnDelta_idx: int = 1,
    ) -> float:
        """
        Spearman rank correlation between policy actions (0/1) and the
        Soft Drop indicator.  A score of 1 means perfect alignment.
        """
        from scipy.stats import spearmanr
        sd_ind = self.soft_drop_indicator(states, lnz_idx, lnDelta_idx)
        rho, _ = spearmanr(actions.astype(float), sd_ind)
        return float(rho) if math.isfinite(rho) else 0.0

    # ── mass metrics ──────────────────────────────────────────────────────
    @staticmethod
    def mass_peak_stats(
        masses: np.ndarray,
        target_mass: float,
        lower_frac: float = 20,
        upper_frac: float = 80,
    ) -> Dict[str, float]:
        """
        Compute mass peak statistics used in the GroomRL reward:
        peak position (median), width (IQR-based), and offset from target.
        """
        masses = np.asarray(masses)
        lower  = float(np.nanpercentile(masses, lower_frac))
        upper  = float(np.nanpercentile(masses, upper_frac))
        inwin  = masses[(masses > lower) & (masses < upper)]
        median = float(np.median(inwin)) if len(inwin) > 0 else float("nan")
        return {
            "mass_peak":   median,
            "mass_lower":  lower,
            "mass_upper":  upper,
            "mass_width":  upper - lower,
            "mass_offset": abs(median - target_mass),
        }

    # ── surrogate DT fidelity ─────────────────────────────────────────────
    @staticmethod
    def dt_fidelity(
        dt_policy,              # DTNode (exact → 1.0) or sklearn DT
        states:  np.ndarray,
        actions: np.ndarray,
        normalise_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Fidelity of a DT policy against collected (state, action) pairs.
        """
        preds = np.array([
            dt_policy.predict(normalise_fn(s) if normalise_fn else s)
            for s in states
        ])
        acc = float(np.mean(preds == actions))
        return {
            "fidelity": acc,
            "n_states": len(states),
            "n_agree":  int(np.sum(preds == actions)),
        }

    # ── LMUT diagnostics ──────────────────────────────────────────────────
    @staticmethod
    def lmut_diagnostics(
        lmut_forest,          # LMUTForest
        states:  np.ndarray,
        q_hat:   np.ndarray,  # oracle Q-values (B, n_actions)
    ) -> Dict[str, Any]:
        """
        LMUT-specific metrics: per-action MAE, number of leaves, and
        feature-importance vector.
        """
        n_actions = lmut_forest.n_actions
        maes = []
        for a in range(n_actions):
            preds = np.array([lmut_forest.q_value(s, a) for s in states])
            maes.append(float(np.mean(np.abs(preds - q_hat[:, a]))))
        return {
            "lmut_mae":         maes,
            "lmut_mean_mae":    float(np.mean(maes)),
            "lmut_summary":     lmut_forest.summary(),
            "feature_importance": {
                name: float(imp)
                for name, imp in zip(
                    ["f" + str(i) for i in range(lmut_forest.dim)],
                    lmut_forest.feature_importance
                )
            },
        }

    # ── efficiency ────────────────────────────────────────────────────────
    @staticmethod
    def efficiency_metrics(
        total_steps:    int,
        train_time_s:   float,
        lmut_forest,
        dt_policy=None,
    ) -> Dict[str, Any]:
        import tracemalloc
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        top_stats = snapshot.statistics("lineno")
        mem_mb = sum(s.size for s in top_stats) / 1e6

        return {
            "total_steps":    total_steps,
            "train_time_s":   train_time_s,
            "steps_per_sec":  total_steps / max(train_time_s, 1e-6),
            "lmut_n_leaves":  sum(t.n_leaves() for t in lmut_forest.trees),
            "dt_n_nodes":     dt_policy.n_nodes()   if dt_policy else None,
            "dt_depth":       dt_policy.depth_max() if dt_policy else None,
            "mem_approx_mb":  round(mem_mb, 2),
        }

    # ── comprehensive report ──────────────────────────────────────────────
    def full_report(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        masses:      np.ndarray,
        target_mass: float,
        lmut_forest,
        q_hat:       np.ndarray,
        dt_policy=None,
        normalise_fn: Optional[Callable] = None,
        train_time_s: float = 0.0,
        total_steps:  int   = 0,
    ) -> Dict[str, Any]:
        """Run all metrics and return a single nested dictionary."""
        report: Dict[str, Any] = {}

        report["physics"]    = {
            "physics_alignment": self.physics_alignment(states, actions),
            **self.mass_peak_stats(masses, target_mass),
        }
        report["lmut"]       = self.lmut_diagnostics(lmut_forest, states, q_hat)
        report["efficiency"] = self.efficiency_metrics(
            total_steps, train_time_s, lmut_forest, dt_policy
        )
        if dt_policy is not None:
            report["dt"] = self.dt_fidelity(dt_policy, states, actions, normalise_fn)

        return report


# ═══════════════════════════════════════════════════════════════════════════
#  4. Metrics Logger
# ═══════════════════════════════════════════════════════════════════════════
class MetricsLogger:
    """
    Writes per-step training metrics to CSV and a final JSON summary.

    Parameters
    ----------
    output_dir : str
        Directory where logs are written.
    run_name : str
        Prefix for file names.
    """

    def __init__(self, output_dir: str, run_name: str = "ibmdp_lmut"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name   = run_name
        self._csv_path  = self.output_dir / f"{run_name}_metrics.csv"
        self._rows: List[Dict] = []
        self._fieldnames: Optional[List[str]] = None

    def log_step(self, step: int, **kwargs):
        """Record a row of per-step metrics."""
        row = {"step": step, "timestamp": time.time(), **kwargs}
        self._rows.append(row)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

    def flush(self):
        """Write all buffered rows to the CSV file."""
        if not self._rows:
            return
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def save_summary(self, report: Dict[str, Any]):
        """Write a final JSON summary."""
        path = self.output_dir / f"{self.run_name}_summary.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[MetricsLogger] Summary → {path}")
        self.flush()
        print(f"[MetricsLogger] CSV     → {self._csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. SB3 convergence callback
# ═══════════════════════════════════════════════════════════════════════════
if _SB3_OK:
    class ConvergenceCallback(BaseCallback):
        """
        SB3 callback that runs the ConvergenceMonitor at regular intervals
        and applies stability guard LR reductions when divergence is detected.
        """

        def __init__(
            self,
            monitor:         ConvergenceMonitor,
            guard:           StabilityGuard,
            lmut_cb,         # LMUTOmniscientCallback reference
            ibmdp_agent,     # IBMDPLMUTAgent reference
            check_freq:      int  = 2000,
            verbose:         int  = 1,
        ):
            super().__init__(verbose)
            self.monitor   = monitor
            self.guard     = guard
            self.lmut_cb   = lmut_cb
            self.agent     = ibmdp_agent
            self.check_freq = check_freq

        def _on_step(self) -> bool:
            if self.num_timesteps % self.check_freq != 0:
                return True

            # record Q-value stability
            try:
                q_arr = self.model.q_net(
                    self.model.q_net.obs_to_tensor(
                        np.random.randn(1, self.model.observation_space.shape[0]).astype(np.float32)
                    )[0]
                ).detach().cpu().numpy()
                max_q = float(np.max(np.abs(q_arr)))
                triggered = self.guard.record_q(max_q)
                if triggered:
                    old_lr = self.model.learning_rate
                    if callable(old_lr):
                        old_lr = float(old_lr(1.0))
                    new_lr = self.guard.reduce_lr(old_lr)
                    self.model.learning_rate = new_lr
                    if self.verbose >= 1:
                        print(f"[StabilityGuard] LR reduced {old_lr:.2e} → {new_lr:.2e} "
                              f"(trigger #{self.guard.n_triggers})")
            except Exception:
                pass

            # run convergence check
            lmut_hist = self.lmut_cb.history
            err = lmut_hist["lmut_error"][-1] if lmut_hist["lmut_error"] else float("inf")

            def _pf(obs):
                a, _ = self.model.predict(obs, deterministic=True)
                return int(a)

            dt = self.agent._extract_dt()
            converged = self.monitor.check(
                step=self.num_timesteps,
                policy_fn=_pf,
                lmut_error=err,
                dt_node_count=dt.n_nodes(),
            )

            if self.verbose >= 1 and self.num_timesteps % (self.check_freq * 5) == 0:
                hist = self.monitor.history
                print(f"[Convergence] step={self.num_timesteps}  "
                      f"action_change={hist['action_change_frac'][-1]:.3f}  "
                      f"lmut_err={err:.4f}  "
                      f"dt_nodes={dt.n_nodes()}  "
                      f"converged={converged}")

            return not converged   # return False → stop training early
