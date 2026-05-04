# xrl_integration.py  –  part of GroomRL + SB3 + LogicXGNN integration
"""
Explainable-RL (XRL) analysis module for GroomRL.

Approach
--------
We fuse two complementary explanation strategies drawn from the LogicXGNN
framework and classical surrogate-model XAI:

1. **Surrogate Decision-Tree (DT)**
   Collect all (Lund-state, action) pairs produced by the trained policy on a
   test sample.  Fit a shallow Decision Tree on these pairs.  Extract
   human-readable grooming rules expressed in Lund coordinates
   (lnz, lnDelta, psi, lnm, lnKt).

2. **Jet-Tree Graph Neural Network + LogicXGNN-style predicate mining**
   Represent each jet's declustering tree as a PyTorch-Geometric graph
   (nodes = declustering nodes, node features = Lund coordinates, edge = soft /
   harder child relationship).  Train a lightweight 2-layer GCN to predict the
   groomed/kept label at every node.  Then apply the LogicXGNN predicate
   pipeline: Weisfeiler-Lehman subgraph hashing → binary predicates → decision
   tree rule extraction → grounding.

Both outputs are written to ``<output_dir>/xrl/``.
"""

from __future__ import annotations

import math
import os
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# sklearn
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# GroomRL internals
from groomrl.JetTree import JetTree, LundCoordinates
from groomrl.Groomer import AbstractGroomer

# Optional PyG / torch for GNN path
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader as PyGLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    import networkx as nx
    from torch_geometric.utils import to_networkx
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

FEATURE_NAMES_2D = ["lnz", "lnDelta"]
FEATURE_NAMES_5D = ["lnz", "lnDelta", "psi", "lnm", "lnKt"]


# ══════════════════════════════════════════════════════════════════════════════
#  Part 1 – Trajectory collection
# ══════════════════════════════════════════════════════════════════════════════

def _collect_node_decisions(
    groomer: AbstractGroomer,
    jet,
    states: list,
    actions: list,
):
    """
    Walk the declustering tree of *jet* under *groomer* and record each
    (state, action) pair at every declustering node.

    We replicate the groomer's traversal logic so we can capture what action
    was taken at each node without modifying the Groomer classes.
    """
    tree = JetTree(jet)
    _walk_tree(groomer, tree, states, actions)


def _walk_tree(groomer, tree: JetTree, states: list, actions: list):
    """Recursive traversal mirroring Groomer._groom but recording decisions."""
    if not tree.lundCoord:
        return

    state = tree.state().astype(np.float32)

    # Determine action the policy would take
    if hasattr(groomer, "model") and hasattr(groomer.model, "predict"):
        # SB3-style predict
        try:
            action_arr, _ = groomer.model.predict(state, deterministic=True)
            action = int(action_arr)
        except Exception:
            action = 0
    elif hasattr(groomer, "model") and hasattr(groomer.model, "predict_on_batch"):
        # keras-rl style
        q_values = groomer.model.predict_on_batch(np.array([[state]])).flatten()
        action = int(np.argmax(q_values))
    else:
        action = 0

    states.append(state)
    actions.append(action)

    # Mirror groomer traversal
    if action == 1:
        tree.remove_soft()
        _walk_tree(groomer, tree, states, actions)
    else:
        if tree.harder:
            _walk_tree(groomer, tree.harder, states, actions)
        if tree.softer:
            _walk_tree(groomer, tree.softer, states, actions)


def collect_trajectories(
    groomer: AbstractGroomer,
    events: list,
    max_events: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect all (Lund-state, action) pairs from *groomer* applied to *events*.

    Returns
    -------
    states  : np.ndarray, shape (N, dim)
    actions : np.ndarray, shape (N,)   dtype int (0=keep, 1=groom)
    """
    all_states:  List[np.ndarray] = []
    all_actions: List[int]        = []

    n = len(events) if max_events < 0 else min(max_events, len(events))
    for jet in events[:n]:
        _collect_node_decisions(groomer, jet, all_states, all_actions)

    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
#  Part 2 – Surrogate Decision-Tree explanation
# ══════════════════════════════════════════════════════════════════════════════

def _feature_names(dim: int) -> List[str]:
    full = FEATURE_NAMES_5D
    return full[:dim]


def fit_surrogate_dt(
    states: np.ndarray,
    actions: np.ndarray,
    max_depth: int = 4,
) -> DecisionTreeClassifier:
    """Fit a shallow Decision Tree on (state → action) pairs."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(states, actions)
    return clf


def extract_dt_rules(clf: DecisionTreeClassifier, feature_names: List[str]) -> str:
    """Return a human-readable text representation of the DT rules."""
    return export_text(clf, feature_names=feature_names, decimals=4)


def plot_dt(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    output_path: str,
):
    """Save a matplotlib decision-tree visualisation."""
    fig, ax = plt.subplots(figsize=(max(12, clf.get_depth() * 4), 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["keep (0)", "groom (1)"],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[XRL] Decision-tree plot  → {output_path}")


def plot_feature_importance(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    output_path: str,
):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Gini importance")
    ax.set_title("Surrogate DT – Feature Importance for Grooming Decisions")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[XRL] Feature-importance plot → {output_path}")


def plot_action_distribution(
    states: np.ndarray,
    actions: np.ndarray,
    feature_names: List[str],
    output_path: str,
):
    """2-D scatter of groom vs. keep decisions in (lnz, lnDelta) space."""
    groom_mask = actions == 1
    keep_mask  = ~groom_mask

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        states[keep_mask,  0], states[keep_mask,  1],
        c="steelblue", alpha=0.3, s=5, label="keep (0)"
    )
    ax.scatter(
        states[groom_mask, 0], states[groom_mask, 1],
        c="firebrick", alpha=0.3, s=5, label="groom (1)"
    )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1] if len(feature_names) > 1 else "")
    ax.set_title("Grooming decisions in Lund coordinate space")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[XRL] Action-distribution plot → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 3 – Jet-Tree Graph construction (for GNN path)
# ══════════════════════════════════════════════════════════════════════════════

def _build_jet_graph(jet, groomer: AbstractGroomer) -> Optional["Data"]:
    """
    Convert a fastjet PseudoJet into a PyTorch-Geometric Data object.

    Nodes  : each declustering node in the Cambridge/Aachen tree.
    Features: Lund coordinates at that node (zero-padded to 5 dims).
    Edges  : directed from child → parent (i.e. following the harder branch up).
    Labels : 0 = keep, 1 = groom  (action taken by the groomer at that node).
    """
    if not _TORCH_AVAILABLE:
        return None

    node_features: List[np.ndarray] = []
    node_labels:   List[int]        = []
    edge_src:      List[int]        = []
    edge_dst:      List[int]        = []
    node_idx_map:  Dict[int, int]   = {}   # id(JetTree) → graph node index

    def visit(tree: JetTree, parent_idx: Optional[int]):
        if not tree.lundCoord:
            return

        idx = len(node_features)
        node_idx_map[id(tree)] = idx

        # Lund state, always padded to 5 dims
        raw = tree.state()
        if len(raw) < 5:
            padded = np.zeros(5, dtype=np.float32)
            padded[:len(raw)] = raw
        else:
            padded = raw[:5].astype(np.float32)
        node_features.append(padded)

        # Action label
        if hasattr(groomer, "model") and hasattr(groomer.model, "predict"):
            try:
                action_arr, _ = groomer.model.predict(raw.astype(np.float32), deterministic=True)
                label = int(action_arr)
            except Exception:
                label = 0
        elif hasattr(groomer, "model") and hasattr(groomer.model, "predict_on_batch"):
            q = groomer.model.predict_on_batch(np.array([[raw]])).flatten()
            label = int(np.argmax(q))
        else:
            label = 0
        node_labels.append(label)

        if parent_idx is not None:
            edge_src.append(idx)
            edge_dst.append(parent_idx)

        if tree.harder:
            visit(tree.harder, idx)
        if tree.softer:
            visit(tree.softer, idx)

    tree = JetTree(jet)
    visit(tree, None)

    if len(node_features) == 0:
        return None

    x    = torch.tensor(np.array(node_features), dtype=torch.float)
    y    = torch.tensor(node_labels, dtype=torch.long)
    if len(edge_src) > 0:
        ei = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=ei, y=y, num_nodes=len(node_features))


# ══════════════════════════════════════════════════════════════════════════════
#  Part 4 – Lightweight GCN for node-level grooming classification
# ══════════════════════════════════════════════════════════════════════════════

class _NodeGCN(nn.Module if _TORCH_AVAILABLE else object):
    """2-layer GCN for node-level binary classification (groom / keep)."""

    def __init__(self, in_channels: int = 5, hidden: int = 32):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch / torch_geometric required for GNN path")
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc    = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        logits = self.fc(x)
        return logits


def _train_node_gnn(
    graphs: List["Data"],
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: int = 32,
) -> "_NodeGCN":
    """Train the node GCN on the labelled jet-tree graphs."""
    device = torch.device("cpu")
    model = _NodeGCN(in_channels=5, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = PyGLoader(graphs, batch_size=32, shuffle=True)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"[XRL GNN] epoch {ep+1:3d}/{epochs}  loss={total_loss/len(loader):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Part 5 – LogicXGNN-style predicate extraction (adapted for jet trees)
# ══════════════════════════════════════════════════════════════════════════════

def _wl_hash_subgraph(graph: "Data", center_node: int, hops: int = 1) -> str:
    """
    Compute a Weisfeiler-Lehman hash for the *hops*-hop neighbourhood of
    *center_node* in *graph*.  This mirrors the predicate construction in the
    original LogicXGNN ``build_logicGNN.py``.
    """
    from torch_geometric.utils import k_hop_subgraph
    node_tensor = torch.tensor([center_node], dtype=torch.long)
    node_idx, sub_ei, _, _ = k_hop_subgraph(
        node_idx=node_tensor,
        num_hops=hops,
        edge_index=graph.edge_index,
        relabel_nodes=True,
    )
    sub_data = Data(edge_index=sub_ei, num_nodes=len(node_idx))
    nx_g = to_networkx(sub_data, to_undirected=True)
    return nx.weisfeiler_lehman_graph_hash(nx_g)


def extract_predicates_from_graphs(
    graphs: List["Data"],
    gnn_model: "_NodeGCN",
    threshold: float = 0.0,
    k_hops: int = 1,
) -> Dict:
    """
    For each graph, for each node, compute a (WL_hash, activation_bin) predicate
    and record which graphs / nodes carry it, following the LogicXGNN methodology.

    Returns a dict with keys:
        predicates       : list of unique (wl_hash, act_bin) tuples
        predicate_to_idx : dict predicate → int index
        predicate_node   : dict predicate → list of (graph_idx, node_idx)
        rules_by_class   : dict {0: matrix, 1: matrix}
    """
    from collections import defaultdict

    device = torch.device("cpu")
    gnn_model.eval()

    # For each graph, collect per-node: (predicate, label)
    predicate_node: Dict = defaultdict(list)
    predicate_graph_by_class = {0: {}, 1: {}}

    for g_idx, graph in enumerate(graphs):
        graph = graph.to(device)
        with torch.no_grad():
            logits = gnn_model(graph.x, graph.edge_index)
            preds  = logits.argmax(dim=1).cpu().numpy()
            acts   = logits[:, 1].cpu().numpy()   # activation for "groom"

        # Representative label for this graph = majority vote
        graph_label = int(np.bincount(preds).argmax())

        node_predicates = set()
        for n in range(graph.num_nodes):
            try:
                wl_h = _wl_hash_subgraph(graph, n, hops=k_hops)
            except Exception:
                wl_h = "none"
            act_bin = int(acts[n] > threshold)
            pred = (wl_h, act_bin)
            predicate_node[pred].append((g_idx, n))
            node_predicates.add(pred)

        predicate_graph_by_class[graph_label][g_idx] = node_predicates

    predicates      = list(predicate_node.keys())
    predicate_to_idx = {p: i for i, p in enumerate(predicates)}

    # Build binary graph × predicate matrices for each class
    def _build_matrix(class_graphs: dict) -> np.ndarray:
        graph_ids = sorted(class_graphs.keys())
        mat = np.zeros((len(predicates), len(graph_ids)), dtype=np.float32)
        for col, g_idx in enumerate(graph_ids):
            for p in class_graphs[g_idx]:
                if p in predicate_to_idx:
                    mat[predicate_to_idx[p], col] = 1.0
        return mat

    mat0 = _build_matrix(predicate_graph_by_class[0])
    mat1 = _build_matrix(predicate_graph_by_class[1])

    return {
        "predicates":         predicates,
        "predicate_to_idx":   predicate_to_idx,
        "predicate_node":     dict(predicate_node),
        "rules_matrix_keep":  mat0,
        "rules_matrix_groom": mat1,
    }


def learn_logical_rules(predicate_data: dict, max_depth: int = 3) -> dict:
    """
    Fit a Decision Tree over the predicate matrices (LogicXGNN Step 4).
    Returns the classifier and a human-readable rule summary.
    """
    mat0 = predicate_data["rules_matrix_keep"]
    mat1 = predicate_data["rules_matrix_groom"]

    if mat0.shape[1] == 0 or mat1.shape[1] == 0:
        return {"rules_keep": [], "rules_groom": [], "accuracy": 0.0}

    X = np.concatenate([mat0, mat1], axis=1).T
    y = np.concatenate([np.zeros(mat0.shape[1]), np.ones(mat1.shape[1])])

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"[XRL Logic] Predicate-DT fidelity: {acc:.4f}")

    rules_text = export_text(
        clf,
        feature_names=[f"pred_{i}" for i in range(X.shape[1])],
        decimals=3,
    )
    return {"clf": clf, "accuracy": acc, "rules_text": rules_text}


# ══════════════════════════════════════════════════════════════════════════════
#  Part 6 – Top-level orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class XRLAnalyzer:
    """
    Orchestrator for the full XRL pipeline.

    Parameters
    ----------
    groomer : AbstractGroomer
        Trained groomer (keras-rl or SB3-based).
    events : list
        List of fastjet PseudoJets (test sample).
    output_dir : str
        Root output directory; an ``xrl/`` subfolder is created automatically.
    max_events : int
        Maximum number of events to analyse (−1 = all).
    dt_max_depth : int
        Maximum depth of the surrogate decision tree.
    run_gnn : bool
        Whether to run the GNN + LogicXGNN pipeline (requires PyG / torch).
    gnn_epochs : int
        Number of epochs for the node GCN.
    lund_dim : int
        Dimensionality of the Lund state vector used during training.
    """

    def __init__(
        self,
        groomer: AbstractGroomer,
        events: list,
        output_dir: str,
        max_events: int = -1,
        dt_max_depth: int = 4,
        run_gnn: bool = True,
        gnn_epochs: int = 30,
        lund_dim: int = 2,
    ):
        self.groomer     = groomer
        self.events      = events
        self.output_dir  = Path(output_dir) / "xrl"
        self.max_events  = max_events
        self.dt_max_depth = dt_max_depth
        self.run_gnn     = run_gnn and _TORCH_AVAILABLE
        self.gnn_epochs  = gnn_epochs
        self.feat_names  = _feature_names(lund_dim)

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ run
    def run(self):
        """Execute the full XRL pipeline and save all outputs."""
        print("\n" + "=" * 60)
        print("  XRL Analysis")
        print("=" * 60)

        # ── 1. Collect trajectories ──────────────────────────────────────
        print("[XRL] Collecting (state, action) trajectories …")
        states, actions = collect_trajectories(
            self.groomer, self.events, self.max_events
        )
        n_nodes = len(states)
        n_groom = int(actions.sum())
        print(
            f"[XRL] Collected {n_nodes} declustering nodes; "
            f"{n_groom} groomed ({100*n_groom/n_nodes:.1f}%)"
        )
        np.save(self.output_dir / "states.npy",  states)
        np.save(self.output_dir / "actions.npy", actions)

        # ── 2. Surrogate DT ──────────────────────────────────────────────
        print("[XRL] Fitting surrogate Decision Tree …")
        clf = fit_surrogate_dt(states, actions, max_depth=self.dt_max_depth)
        dt_acc = accuracy_score(actions, clf.predict(states))
        print(f"[XRL] Surrogate DT train accuracy: {dt_acc:.4f}")
        print("[XRL] Classification report:\n",
              classification_report(actions, clf.predict(states),
                                    target_names=["keep", "groom"], zero_division=0))

        rules_text = extract_dt_rules(clf, self.feat_names)
        rules_file = self.output_dir / "dt_rules.txt"
        rules_file.write_text(
            f"Surrogate Decision Tree Rules (depth={self.dt_max_depth})\n"
            f"Train accuracy: {dt_acc:.4f}\n\n"
            + rules_text
        )
        print(f"[XRL] Rules written → {rules_file}")

        # Plots
        plot_dt(clf, self.feat_names, str(self.output_dir / "dt_tree.pdf"))
        plot_feature_importance(
            clf, self.feat_names,
            str(self.output_dir / "feature_importance.pdf")
        )
        plot_action_distribution(
            states, actions, self.feat_names,
            str(self.output_dir / "action_distribution.pdf")
        )

        # ── 3. (Optional) GNN + LogicXGNN ────────────────────────────────
        gnn_results = {}
        if self.run_gnn:
            gnn_results = self._run_gnn_pipeline()
        else:
            print("[XRL] Skipping GNN path (torch/torch_geometric not available "
                  "or run_gnn=False).")

        # ── 4. Summary JSON ──────────────────────────────────────────────
        summary = {
            "n_nodes":         n_nodes,
            "n_groomed":       n_groom,
            "groom_fraction":  float(n_groom / n_nodes),
            "dt_max_depth":    self.dt_max_depth,
            "dt_train_acc":    float(dt_acc),
            "feature_names":   self.feat_names,
            "dt_rules":        rules_text,
            "gnn_available":   bool(self.run_gnn),
        }
        if gnn_results:
            summary["gnn_node_acc"]      = gnn_results.get("node_acc", None)
            summary["logic_dt_fidelity"] = gnn_results.get("logic_acc", None)

        summary_file = self.output_dir / "xrl_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"[XRL] Summary written → {summary_file}")
        print("=" * 60 + "\n")
        return summary

    # ------------------------------------------------------------------ GNN path
    def _run_gnn_pipeline(self) -> dict:
        print("[XRL GNN] Building jet-tree graphs …")
        n = len(self.events) if self.max_events < 0 else min(self.max_events, len(self.events))
        graphs = []
        for jet in self.events[:n]:
            g = _build_jet_graph(jet, self.groomer)
            if g is not None and g.num_nodes > 1:
                graphs.append(g)
        print(f"[XRL GNN] Built {len(graphs)} valid graphs.")
        if len(graphs) < 10:
            print("[XRL GNN] Too few graphs – skipping GNN path.")
            return {}

        # Train node GCN
        gnn_model = _train_node_gnn(
            graphs, epochs=self.gnn_epochs, lr=1e-3, hidden=32
        )

        # Evaluate node-level accuracy
        gnn_model.eval()
        all_preds, all_labels = [], []
        for g in graphs:
            with torch.no_grad():
                out   = gnn_model(g.x, g.edge_index)
                preds = out.argmax(dim=1).numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(g.y.numpy().tolist())
        node_acc = accuracy_score(all_labels, all_preds)
        print(f"[XRL GNN] Node-level accuracy: {node_acc:.4f}")

        # Save GNN
        torch.save(gnn_model.state_dict(), str(self.output_dir / "node_gcn.pt"))

        # LogicXGNN predicate extraction
        print("[XRL Logic] Extracting WL predicates …")
        predicate_data = extract_predicates_from_graphs(graphs, gnn_model)
        n_pred = len(predicate_data["predicates"])
        print(f"[XRL Logic] Found {n_pred} unique predicates.")

        logic_result = learn_logical_rules(predicate_data, max_depth=3)
        logic_acc    = logic_result.get("accuracy", 0.0)

        rules_file = self.output_dir / "logic_rules.txt"
        rules_file.write_text(
            f"LogicXGNN Predicate Rules\n"
            f"Fidelity: {logic_acc:.4f}\n\n"
            + logic_result.get("rules_text", "")
        )
        print(f"[XRL Logic] Rules written → {rules_file}")

        return {"node_acc": node_acc, "logic_acc": logic_acc}
