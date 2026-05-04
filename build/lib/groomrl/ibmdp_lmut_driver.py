# ibmdp_lmut_driver.py  –  GroomRL-XRL
"""
High-level driver that wires IBMDP + LMUT into the existing GroomRL pipeline.

Call ``run_ibmdp_lmut()`` from groomer.py (or directly) to:
  1. Build the IBMDP-wrapped environment.
  2. Train an IBMDPLMUTAgent.
  3. Run the full EvaluationSuite.
  4. Write all artefacts (DT policy, LMUT model, metrics CSV/JSON, plots).

Typical usage in groomer.py
----------------------------
    from groomrl.ibmdp_lmut_driver import run_ibmdp_lmut
    results = run_ibmdp_lmut(setup, output_dir="my_run", nstep=200_000)
"""

from __future__ import annotations

import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── optional heavy deps ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from scipy.stats import spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ── GroomRL-XRL internals ──────────────────────────────────────────────────
from groomrl.ibmdp_lmut_agent   import IBMDPLMUTAgent
from groomrl.ibmdp_env          import IBMDPGroomEnv, _normalise
from groomrl.lmut_core          import LMUTForest, Transition
from groomrl.convergence_metrics import (
    EvaluationSuite, MetricsLogger, ConvergenceMonitor,
    StabilityGuard, ConvergenceCallback,
)
from groomrl.read_data          import Jets
from groomrl.JetTree            import JetTree, LundCoordinates
from groomrl.tools              import mass


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect (state, action, mass) from a groomer on a jet sample
# ─────────────────────────────────────────────────────────────────────────────
def _collect_eval_data(groomer, events: list, max_events: int = -1):
    """
    Walk each jet through the groomer, collecting Lund states, actions, and
    groomed momenta.  Returns (states, actions, groomed_jets).
    """
    states:  List[np.ndarray] = []
    actions: List[int]        = []
    groomed: list             = []

    n = len(events) if max_events < 0 else min(max_events, len(events))
    for jet in events[:n]:
        tree = JetTree(jet)
        _walk(groomer, tree, states, actions)
        groomed.append(deepcopy(groomer(jet)))

    return (
        np.array(states,  dtype=np.float32) if states  else np.zeros((0, LundCoordinates.dimension)),
        np.array(actions, dtype=int),
        groomed,
    )


def _walk(groomer, tree: JetTree, states: list, actions: list):
    if not tree.lundCoord:
        return
    state = tree.state().astype(np.float32)

    if hasattr(groomer, "model") and hasattr(groomer.model, "predict"):
        a_arr, _ = groomer.model.predict(state, deterministic=True)
        action = int(a_arr)
    elif hasattr(groomer, "dt_policy") and groomer.dt_policy is not None:
        action = groomer.dt_policy.predict(state.astype(np.float64))
    else:
        action = 0

    states.append(state)
    actions.append(action)

    if action == 1:
        tree.remove_soft()
        _walk(groomer, tree, states, actions)
    else:
        if tree.harder: _walk(groomer, tree.harder, states, actions)
        if tree.softer: _walk(groomer, tree.softer, states, actions)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_feature_importance(importance: np.ndarray, names: List[str], path: str):
    if not _MPL:
        return
    # Pad names with auto-generated labels if the importance array is longer
    # (e.g. IBMDP augments the state to 2×lund_dim but names covers only lund_dim).
    names = list(names)
    if len(names) < len(importance):
        names += [f"f{i}" for i in range(len(names), len(importance))]
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.argsort(importance)[::-1]
    ax.bar(range(len(importance)), importance[idx], color="#2E86C1", alpha=0.85)
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([names[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Normalised importance")
    ax.set_title("LMUT feature importance (grooming decisions)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_lmut_convergence(lmut_cb_history: Dict, path: str):
    if not _MPL or not lmut_cb_history["lmut_error"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lmut_cb_history["lmut_step"], lmut_cb_history["lmut_error"],
                 color="#C0392B", lw=1.5, alpha=0.8)
    axes[0].set_xlabel("Training step"); axes[0].set_ylabel("LMUT MSE")
    axes[0].set_title("LMUT training error")
    axes[1].plot(lmut_cb_history["lmut_step"], lmut_cb_history["n_leaves"],
                 color="#1B4F72", lw=1.5, alpha=0.8)
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Total leaf nodes")
    axes[1].set_title("LMUT model complexity")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_action_scatter(states: np.ndarray, actions: np.ndarray, path: str,
                         feat_names: List[str]):
    if not _MPL or len(states) == 0:
        return
    g = actions == 1
    k = ~g
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(states[k, 0], states[k, 1],
               c="#2980B9", s=4, alpha=0.3, label="keep (0)")
    ax.scatter(states[g, 0], states[g, 1],
               c="#E74C3C", s=4, alpha=0.3, label="groom (1)")
    ax.set_xlabel(feat_names[0] if len(feat_names) > 0 else "f0")
    ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else "f1")
    ax.set_title("Grooming decisions (IBMDP+LMUT policy)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_dt_structure(dt_node, path: str):
    """Simple ASCII-to-file DT print."""
    lines = []

    def _print(node, prefix="", is_left=True):
        if node is None:
            return
        connector = "├── " if is_left else "└── "
        if node.is_leaf:
            lines.append(f"{prefix}{connector}LEAF → action={node.action}")
        else:
            lines.append(f"{prefix}{connector}SPLIT feat={node.feat} val={node.val:.4f}")
            ext = "│   " if is_left else "    "
            _print(node.left,  prefix + ext, is_left=True)
            _print(node.right, prefix + ext, is_left=False)

    lines.append(f"DT Policy ({dt_node.n_nodes()} nodes, depth={dt_node.depth_max()})")
    _print(dt_node, "", is_left=False)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main driver function
# ─────────────────────────────────────────────────────────────────────────────
def run_ibmdp_lmut(
    setup:              Dict[str, Any],
    output_dir:         str,
    nstep:              int   = 200_000,
    agent_type:         str   = "dqn",
    p:                  int   = 3,
    zeta:               float = -0.01,
    gamma_b:            float = 0.99,
    gamma_w:            float = 1.0,
    max_depth:          int   = 4,
    nev_eval:           int   = -1,
    enable_convergence: bool  = True,
    verbose:            int   = 1,
) -> Dict[str, Any]:
    """
    Full IBMDP + LMUT training and evaluation pipeline.

    Parameters
    ----------
    setup : dict
        GroomRL runcard dictionary (as loaded by load_runcard).
    output_dir : str
        Output folder (must exist or be creatable).
    nstep : int
        Total SB3 training timesteps.
    agent_type : str
        "dqn" or "ppo".
    p : int
        IBMDP split granularity.
    zeta : float
        Information-gathering penalty.
    gamma_b / gamma_w : float
        Discount factors for base / info-gathering transitions.
    max_depth : int
        DT depth limit.
    nev_eval : int
        Number of test events for evaluation (−1 = all).
    enable_convergence : bool
        Whether to attach ConvergenceCallback (early stopping).
    verbose : int
        Verbosity.

    Returns
    -------
    dict : Full evaluation report + paths to saved artefacts.
    """
    out = Path(output_dir)
    (out / "ibmdp_lmut").mkdir(parents=True, exist_ok=True)
    xrl_dir = out / "ibmdp_lmut"

    env_setup   = setup["groomer_env"]
    lund_dim    = int(env_setup.get("state_dim", 2))
    _base_names = ["lnz", "lnDelta", "psi", "lnm", "lnKt"][:lund_dim]
    # IBMDP augments the state to 2×lund_dim; provide names for both halves.
    feat_names  = _base_names + [f"{n}_b" for n in _base_names]
    target_mass = float(env_setup["mass"])

    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"  IBMDP + LMUT Integration  |  {agent_type.upper()}  |  dim={lund_dim}")
        print(f"{'='*60}\n")

    # ── environment factory ───────────────────────────────────────────────
    from groomrl.GroomEnvSB3 import make_sb3_env

    def _make_env():
        return make_sb3_env(env_setup, LundCoordinates.low, LundCoordinates.high)

    # ── build and train agent ────────────────────────────────────────────
    agent = IBMDPLMUTAgent(
        base_env_factory     = _make_env,
        agent_type           = agent_type,
        p                    = p,
        zeta                 = zeta,
        gamma_b              = gamma_b,
        gamma_w              = gamma_w,
        max_depth            = max_depth,
        nstep                = nstep,
        lmut_buffer          = 512,
        lmut_min_split       = 30,
        lmut_min_improvement = 1e-3,
        verbose              = verbose,
    )

    extra_cbs = []
    if enable_convergence:
        # build a small set of random bounding-feature observations for stability checks
        rng        = np.random.default_rng(42)
        eval_obs   = rng.uniform(0, 1, (200, 2 * lund_dim)).astype(np.float32)
        guard      = StabilityGuard(q_clip=200.0, var_threshold=80.0)
        monitor    = ConvergenceMonitor(
            eval_env=None,
            eval_states=eval_obs,
            eval_freq=2000,
            patience=5,
        )
        conv_cb = ConvergenceCallback(
            monitor    = monitor,
            guard      = guard,
            lmut_cb    = agent._lmut_cb,
            ibmdp_agent= agent,
            check_freq = 2000,
            verbose    = verbose,
        )
        extra_cbs.append(conv_cb)

    agent.fit(extra_callbacks=extra_cbs)

    # ── save artefacts ────────────────────────────────────────────────────
    agent.save(str(xrl_dir))

    if agent.dt_policy is not None:
        _plot_dt_structure(agent.dt_policy, str(xrl_dir / "dt_policy.txt"))

    _plot_feature_importance(
        agent.lmut.feature_importance, feat_names,
        str(xrl_dir / "lmut_feature_importance.pdf")
    )
    _plot_lmut_convergence(
        agent._lmut_cb.history,
        str(xrl_dir / "lmut_convergence.pdf")
    )

    # ── evaluate on test set ──────────────────────────────────────────────
    groomer = agent.groomer()
    reader  = Jets(env_setup["fn"], nev_eval)
    events  = reader.values()

    if verbose >= 1:
        print("[Driver] Collecting evaluation trajectories …")
    states_eval, actions_eval, groomed_jets = _collect_eval_data(groomer, events, nev_eval)

    _plot_action_scatter(states_eval, actions_eval,
                         str(xrl_dir / "action_scatter.pdf"), feat_names)

    # mass metrics
    groomed_4vecs = [np.array([j[0], j[1], j[2], j[3]]) if not hasattr(j, '__len__') else j
                     for j in groomed_jets]
    masses_eval = np.array(mass(groomed_jets)) if groomed_jets else np.array([])

    # oracle Q-values for LMUT diagnostics (use LMUT itself as oracle)
    if len(states_eval) > 0:
        q_hat_eval = np.array([
            agent.lmut.q_values(s) for s in states_eval
        ])
    else:
        q_hat_eval = np.zeros((0, agent.lmut.n_actions))

    suite = EvaluationSuite(
        feature_names = feat_names,
        zcut          = float(env_setup.get("zcut", 0.05)),
        beta          = float(env_setup.get("beta", 1.0)),
    )

    lund_low  = LundCoordinates.low
    lund_high = LundCoordinates.high
    norm_fn   = lambda s: _normalise(s, lund_low, lund_high)

    report = suite.full_report(
        states      = states_eval,
        actions     = actions_eval,
        masses      = masses_eval,
        target_mass = target_mass,
        lmut_forest = agent.lmut,
        q_hat       = q_hat_eval,
        dt_policy   = agent.dt_policy,
        normalise_fn= norm_fn,
        train_time_s= agent.train_time,
        total_steps = nstep,
    )

    # add agent diagnostics
    report["agent"] = agent.report()

    # ── log and save ──────────────────────────────────────────────────────
    logger = MetricsLogger(str(xrl_dir), run_name=f"ibmdp_{agent_type}")
    for i, (step, err) in enumerate(
        zip(agent._lmut_cb.history["lmut_step"],
            agent._lmut_cb.history["lmut_error"])
    ):
        logger.log_step(
            step,
            lmut_error = err,
            n_leaves   = agent._lmut_cb.history["n_leaves"][i],
        )
    logger.save_summary(report)

    if verbose >= 1:
        pa  = report.get("physics", {}).get("physics_alignment", float("nan"))
        mw  = report.get("physics", {}).get("mass_width",        float("nan"))
        fi  = report.get("lmut",    {}).get("lmut_mean_mae",     float("nan"))
        nn  = report.get("agent",   {}).get("dt_n_nodes",        "—")
        print(f"\n{'─'*50}")
        print(f"  Physics alignment (ρ)  : {pa:.3f}")
        print(f"  Mass peak width (GeV)  : {mw:.2f}")
        print(f"  LMUT mean MAE          : {fi:.4f}")
        print(f"  DT nodes               : {nn}")
        print(f"  Outputs → {xrl_dir}")
        print(f"{'─'*50}\n")

    return report
