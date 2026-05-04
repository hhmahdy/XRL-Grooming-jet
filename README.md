# GroomRL-XRL

**Stable-Baselines3 agents + Explainable-RL for jet grooming with GroomRL**

This project extends the original [GroomRL](https://github.com/groomrl/groomrl)
codebase with two major capabilities:

1. **`--agent {dqn,ppo}`** — drop-in replacement for the keras-rl DQN using
   Stable-Baselines3, giving access to both off-policy DQN and on-policy PPO
   without altering the existing environment or Groomer infrastructure.

2. **`--xrl`** — a full Explainable-RL pipeline inspired by the
   [LogicXGNN](https://github.com/LogicXGNN/LogicXGNN) framework, producing
   human-readable grooming rules from the trained policy.

---

## Table of Contents

- [Background](#background)
- [New Files Overview](#new-files-overview)
- [Installation](#installation)
- [Quick-Start Examples](#quick-start-examples)
- [CLI Reference](#cli-reference)
- [Runcard Configuration](#runcard-configuration)
- [XRL Output Guide](#xrl-output-guide)
- [Architecture Notes](#architecture-notes)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Background

### GroomRL

GroomRL trains a reinforcement learning agent to decide, at each node of a
Cambridge/Aachen jet declustering tree, whether to *groom* (remove) the softer
branch or to *keep* it.  The state is expressed in Lund coordinates
(ln z, ln Δ, …), and the reward is a peaked function of the groomed jet mass
plus a Soft Drop regularisation term.

### SB3 Integration

The original codebase used **keras-rl**, which depends on an old Keras/TF stack
and does not support modern Gymnasium environments.  We wrap
`GroomEnv` in a thin [Gymnasium](https://gymnasium.farama.org/) adapter
(`GroomEnvSB3`) and introduce `SB3AgentGroom`, a wrapper around
`stable_baselines3.DQN` and `stable_baselines3.PPO` that exposes the same
`fit() / groomer() / save_weights()` API as the original `DQNAgentGroom`.

### XRL Integration

The XRL pipeline (`xrl_integration.py`) is a two-tier explanation system:

| Tier | What it does |
|------|-------------|
| **Surrogate Decision Tree** | Collects all (Lund-state, action) pairs from policy rollouts, fits a shallow sklearn `DecisionTreeClassifier`, and exports readable rules such as *"if lnz ≤ −2.84 and lnDelta ≤ −0.63 → groom"*. |
| **GNN + LogicXGNN predicates** | Represents each jet's declustering tree as a PyG graph, trains a 2-layer node-level GCN, then applies Weisfeiler-Lehman subgraph hashing to extract binary structural predicates.  A second decision tree over these predicates produces logical rules mirroring the LogicXGNN methodology. |

---

## New Files Overview

```
src/groomrl/
├── GroomEnvSB3.py        ← Gymnasium wrappers for GroomEnv / Dual / Triple
├── SB3AgentGroom.py      ← SB3 DQN & PPO agent wrapper
├── xrl_integration.py    ← Full XRL pipeline (trajectories → DT → GNN → rules)
├── models.py             ← MODIFIED: routes to SB3 or legacy based on --agent
└── scripts/
    └── groomer.py        ← MODIFIED: adds --agent, --xrl, --xrl-events, etc.

runcards/
└── sb3_groomer.json      ← Example runcard for SB3 agents
```

Everything else (`GroomEnv.py`, `JetTree.py`, `Groomer.py`, `DQNAgentGroom.py`,
`diagnostics.py`, `read_data.py`, `tools.py`, `keras_to_cpp.py`) is **unchanged**
so the original training path still works with `--agent legacy`.

---

## Installation

### 1. Clone / copy the repository

```bash
git clone <your-fork>
cd GroomRL_Final
```

### 2. Install GroomRL itself

```bash
pip install -e .
```

### 3. Install SB3 (required for `--agent dqn` / `--agent ppo`)

```bash
pip install stable-baselines3>=2.0
```

### 4. Install XRL dependencies (required for `--xrl`)

**Minimal (Surrogate DT only):**
```bash
pip install scikit-learn matplotlib
```

**Full (includes GNN / LogicXGNN sub-pipeline):**
```bash
pip install torch torchvision
pip install torch_geometric networkx
```

### 5. (Optional) Keep the legacy keras-rl path

```bash
pip install keras-rl tensorflow==2.x
```

### Conda environment (all-in-one)

```yaml
# environment_sb3.yml
name: groomrl_sb3
channels: [conda-forge, defaults]
dependencies:
  - python=3.10
  - pip
  - pip:
    - fastjet
    - stable-baselines3>=2.0
    - gymnasium
    - scikit-learn
    - matplotlib
    - torch
    - torch_geometric
    - networkx
```

```bash
conda env create -f environment_sb3.yml
conda activate groomrl_sb3
pip install -e .
```

---

## Quick-Start Examples

### Train with SB3 DQN

```bash
groomer runcards/sb3_groomer.json --agent dqn -o run_dqn
```

### Train with SB3 PPO

```bash
groomer runcards/sb3_groomer.json --agent ppo -o run_ppo
```

### Train + run XRL analysis in one command

```bash
groomer runcards/sb3_groomer.json \
    --agent dqn \
    --xrl \
    --xrl-depth 4 \
    --xrl-events 2000 \
    -o run_dqn_xrl
```

### Run XRL on an already-trained SB3 model

```bash
groomer --model run_dqn --xrl --agent dqn --xrl-depth 5
```

### Run XRL on an already-trained **legacy** model

```bash
groomer --model results/groomerW_final --xrl --agent legacy
```

### Surrogate DT only (no GNN, faster)

```bash
groomer runcards/sb3_groomer.json \
    --agent ppo \
    --xrl \
    --no-gnn \
    -o run_ppo_xrl_fast
```

### Plot + XRL together

```bash
groomer --model run_dqn --plot --xrl --agent dqn
```

### Original legacy DQN (backward-compatible, no changes)

```bash
groomer runcards/default_dense.json -o run_legacy
```

---

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `runcard` | — | Path to a JSON training runcard |
| `--model MODEL` | — | Load an existing model folder (mutually exclusive with `runcard`) |
| `--output / -o` | runcard name | Output directory |
| `--agent` | `legacy` | RL backend: `legacy` \| `dqn` \| `ppo` |
| `--xrl` | off | Enable XRL analysis after training/loading |
| `--xrl-events N` | −1 (all) | Max events for XRL trajectory collection |
| `--xrl-depth D` | 4 | Max depth of surrogate Decision Tree |
| `--no-gnn` | off | Skip GNN/LogicXGNN sub-pipeline |
| `--plot` | off | Generate mass and Lund-plane plots |
| `--cpp` | off | Export to C++ (legacy only) |
| `--data FILE` | — | Extra data file for plotting |
| `--nev / -n` | −1 | Max events for test grooming |
| `--force / -f` | off | Overwrite existing output folder |

---

## Runcard Configuration

The `groomer_agent` section is shared between all three agent types.  Fields
that are irrelevant to a given agent are silently ignored.

```jsonc
{
  "groomer_env": {
    // ── same as original ──────────────────────────────────────────
    "fn":        "path/to/train.json.gz",
    "nev":       10000,
    "mass":      80.385,
    "width":     10.0,
    "reward":    "cauchy",   // cauchy | gaussian | exponential | inverse
    "state_dim": 2,          // 2 to 5 Lund coordinates
    "SD_groom":  "exp_add",
    "SD_keep":   "exp_add",
    "alpha1": 0.5, "beta1": 0.5,
    "alpha2": 0.5, "beta2": 0.5,
    "SD_norm": 0.1,
    "lnzRef1": -2.0, "lnzRef2": -2.0
  },

  "groomer_agent": {
    // ── network (all agents) ──────────────────────────────────────
    "nb_units":  64,      // neurons per hidden layer
    "nb_layers": 2,       // number of hidden layers

    // ── general training ─────────────────────────────────────────
    "nstep":         50000,   // total training timesteps
    "learning_rate": 1e-3,
    "gamma":         0.99,    // discount factor

    // ── SB3 DQN-specific ─────────────────────────────────────────
    "buffer_size":              500000,
    "learning_starts":          500,
    "batch_size":               32,
    "tau":                      1.0,
    "target_update_interval":   100,
    "exploration_fraction":     0.1,
    "exploration_final_eps":    0.05,

    // ── SB3 PPO-specific ─────────────────────────────────────────
    "n_steps":    2048,   // rollout buffer length
    "n_epochs":   10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef":   0.0,

    // ── legacy (keras-rl) only ────────────────────────────────────
    "architecture":           "Dense",   // Dense | LSTM
    "dropout":                0.0,
    "policy":                 "epsgreedyq",
    "enable_dueling_network": false,
    "enable_double_dqn":      false,
    "optimizer":              "Adam"
  },

  "test": {
    "fn": "path/to/test.json.gz"
  }
}
```

---

## XRL Output Guide

All XRL outputs are written to `<output_dir>/xrl/`.

```
<output_dir>/xrl/
├── states.npy                 # (N, dim) Lund-coordinate array
├── actions.npy                # (N,) int array: 0=keep, 1=groom
│
├── dt_rules.txt               # Human-readable DT grooming rules
├── dt_tree.pdf                # Decision-tree visualisation
├── feature_importance.pdf     # Bar chart of Gini importances
├── action_distribution.pdf    # Scatter: groom/keep in (lnz, lnDelta) space
│
├── node_gcn.pt                # (if --no-gnn not set) Trained node-GCN weights
├── logic_rules.txt            # LogicXGNN-style predicate rules
│
└── xrl_summary.json           # Machine-readable summary of all results
```

### Reading `dt_rules.txt`

```
Surrogate Decision Tree Rules (depth=4)
Train accuracy: 0.9142

|--- lnz <= -2.8400
|   |--- lnDelta <= -0.6300
|   |   |--- class: groom (1)    ← remove soft branch
|   |--- lnDelta >  -0.6300
|   |   |--- class: keep (0)     ← keep soft branch
|--- lnz >  -2.8400
|   |--- class: keep (0)
```

**Interpreting the rules:**
- `lnz = ln(z)` where `z = p_T(soft) / (p_T(hard) + p_T(soft))`.  
  Small `lnz` (very negative) means a very soft emission → likely to be groomed.
- `lnDelta = ln(Δ)` where `Δ = ΔR(j1, j2)` is the angular separation.  
  Large `lnDelta` (less negative) means wide-angle emission.

The tree recovers a learned approximation of Soft Drop: groom when the
emission is both soft **and** at wide angle.

### Reading `xrl_summary.json`

```json
{
  "n_nodes":          12847,
  "n_groomed":        3201,
  "groom_fraction":   0.249,
  "dt_max_depth":     4,
  "dt_train_acc":     0.914,
  "feature_names":    ["lnz", "lnDelta"],
  "gnn_available":    true,
  "gnn_node_acc":     0.891,
  "logic_dt_fidelity": 0.873
}
```

---

## Architecture Notes

### Gymnasium wrapper (`GroomEnvSB3`)

SB3 ≥ 1.7 requires the [Gymnasium](https://gymnasium.farama.org/) API:
- `reset()` → `(obs, info)`
- `step()` → `(obs, reward, terminated, truncated, info)`

`GroomEnvSB3` is a thin adapter that wraps the original `GroomEnv` (which uses
the legacy OpenAI Gym API) without modifying it, preserving backward
compatibility.

### SB3AgentGroom interface

| Method | Purpose |
|--------|---------|
| `fit(total_timesteps)` | Train the agent |
| `groomer()` | Return an `_SB3Groomer` compatible with all downstream code |
| `save_weights(path)` | Save as `path.zip` |
| `SB3AgentGroom.load(agent_type, path)` | Class-method to restore a saved agent |

The returned `_SB3Groomer` is a subclass of `AbstractGroomer`, so it works
transparently with `plot_mass`, `plot_lund`, and any other diagnostic code.

### XRL trajectory collection

We replicate the groomer's recursive tree traversal during analysis without
modifying `Groomer._groom`, by re-walking the same `JetTree` structure and
querying the policy at each node.  This is deterministic and reproducible.

### LogicXGNN adaptation

The original LogicXGNN framework operates on **graph classification** with
Pytorch-Geometric GNNs.  We adapt it for **node-level** grooming decisions:

1. Each jet's declustering tree → a directed PyG `Data` graph.
2. Node features = Lund coordinates (zero-padded to 5 dims).
3. Node labels = policy action at that node.
4. A 2-layer `GCNConv` model is trained for node classification.
5. Per-node k-hop subgraphs are hashed with Weisfeiler-Lehman to form
   structural predicates.
6. A final decision tree over predicate presence/absence yields logical rules.

---

## Dependencies

| Package | Version | Required for |
|---------|---------|--------------|
| `fastjet` | ≥ 3.4 | Core GroomRL |
| `numpy` | ≥ 1.21 | Core |
| `gymnasium` | ≥ 0.26 | SB3 agents |
| `stable-baselines3` | ≥ 2.0 | `--agent dqn/ppo` |
| `scikit-learn` | ≥ 1.0 | `--xrl` (DT) |
| `matplotlib` | ≥ 3.5 | `--xrl` (plots) |
| `torch` | ≥ 2.0 | `--xrl` GNN path |
| `torch_geometric` | ≥ 2.3 | `--xrl` GNN path |
| `networkx` | ≥ 3.0 | `--xrl` WL hash |
| `keras` (TF 2.x) | optional | `--agent legacy` |
| `keras-rl` | optional | `--agent legacy` |
| `hyperopt` | optional | hyperparameter scan |

---

## Troubleshooting

**`ImportError: stable_baselines3 is not installed`**  
→ `pip install stable-baselines3`

**`ImportError: gymnasium is not installed`**  
→ `pip install gymnasium`

**`--xrl` runs but "Skipping GNN path"**  
→ Install `torch` and `torch_geometric`.  Or pass `--no-gnn` to use only the
surrogate DT (no torch required).

**Loading a legacy model with `--agent dqn`**  
→ If the folder contains `weights.h5` (legacy), the loader detects this and
uses the Keras path regardless of `--agent`.

**Training is slow with SB3 PPO on small runs**  
→ PPO collects `n_steps` transitions before each update.  For short runs
(`nstep < 4096`), use DQN or reduce `n_steps` in the runcard.

**`gym.utils.seeding` ImportError after upgrading gym**  
→ The original `GroomEnv.py` uses the old `gym` API.  Do not upgrade `gym`
beyond 0.26 unless you also migrate `GroomEnv.py` itself.  The SB3 path wraps
the legacy env with `GroomEnvSB3` to avoid this issue.
