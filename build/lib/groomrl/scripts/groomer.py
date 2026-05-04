#!/usr/bin/env python
# groomer.py  –  part of GroomRL (modified for SB3 + XRL integration)
#
# Original authors: S. Carrazza and F. A. Dreyer
# SB3 / XRL additions: GroomRL-XRL project
"""
Entry point for training, evaluating, and explaining GroomRL agents.

New CLI arguments
-----------------
--agent {legacy,dqn,ppo}
    Select the RL backend:
      legacy  – original keras-rl DQN (default; requires keras-rl)
      dqn     – Stable-Baselines3 Deep Q-Network
      ppo     – Stable-Baselines3 Proximal Policy Optimisation

--xrl
    After training (or after loading an existing model), run the full
    Explainable-RL analysis pipeline and write outputs to
    ``<output>/xrl/``.

--xrl-events N
    Number of test events to use for XRL analysis (default: −1 = all).

--xrl-depth D
    Maximum depth of the surrogate Decision Tree (default: 4).

--no-gnn
    Disable the GNN / LogicXGNN sub-pipeline in XRL (useful when
    torch_geometric is not installed).

Usage examples
--------------
Train with SB3 DQN and run XRL::

    groomer runcards/default_dense.json --agent dqn --xrl -o my_run

Train with SB3 PPO::

    groomer runcards/default_dense.json --agent ppo -o my_ppo_run

Re-run XRL on an existing model folder::

    groomer --model results/groomerW_final --xrl --agent dqn

Plot from an existing model (original behaviour)::

    groomer --model results/groomerW_final --plot
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import pprint
import shutil
from copy import deepcopy
from shutil import copyfile
from time import time

import argparse

# ── GroomRL internals ──────────────────────────────────────────────────────
from groomrl.read_data import Jets
from groomrl.models import build_and_train_model, load_runcard
from groomrl.diagnostics import plot_mass, plot_lund, plot_decision_boundary, plot_reward_decomposition
from groomrl.JetTree import LundCoordinates

# Legacy Groomer (always importable for --plot / --cpp on existing models)
try:
    from groomrl.Groomer import Groomer
    from groomrl.keras_to_cpp import keras_to_cpp, check_model
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

# SB3 agent (optional)
try:
    from groomrl.SB3AgentGroom import SB3AgentGroom
    from groomrl.GroomEnvSB3 import make_sb3_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# XRL (optional – graceful degradation)
try:
    from groomrl.xrl_integration import XRLAnalyzer
    _XRL_AVAILABLE = True
except ImportError:
    _XRL_AVAILABLE = False

# Hyperopt (optional)
try:
    from hyperopt import fmin, tpe, hp, Trials, space_eval
    from hyperopt.mongoexp import MongoTrials
    _HYPEROPT_AVAILABLE = True
except ImportError:
    _HYPEROPT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
def run_hyperparameter_scan(search_space: dict) -> dict:
    """Run a hyperopt scan (legacy path only)."""
    if not _HYPEROPT_AVAILABLE:
        raise ImportError("hyperopt is not installed.")
    print('[+] Performing hyperparameter scan…')
    if search_space['cluster']['enable']:
        trials = MongoTrials(
            search_space['cluster']['url'],
            exp_key=search_space['cluster']['exp_key'],
        )
    else:
        trials = Trials()

    max_evals = search_space['cluster']['max_evals']
    best = fmin(
        build_and_train_model, search_space,
        algo=tpe.suggest, max_evals=max_evals, trials=trials,
    )
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = '%s/hyperopt_log_{}.pickle'.format(time()) % search_space['output']
    with open(log, 'wb') as wfp:
        print(f'[+] Saving trials → {log}')
        pickle.dump(trials.trials, wfp)

    best_setup['scan'] = False
    return best_setup


# ─────────────────────────────────────────────────────────────────────────────
def load_json(runcard_file: str, agent_type: str = 'legacy') -> dict:
    """Load + parse a JSON runcard, hoisting hyperopt expressions."""
    runcard = load_runcard(runcard_file)
    runcard['scan']  = False
    runcard['agent'] = agent_type   # inject CLI choice
    for key, value in runcard.get('groomer_env', {}).items():
        if 'hp.' in str(value):
            runcard['groomer_env'][key] = eval(value)
            runcard['scan'] = True
    for key, value in runcard.get('groomer_agent', {}).items():
        if 'hp.' in str(value):
            runcard['groomer_agent'][key] = eval(value)
            runcard['scan'] = True
    return runcard


# ─────────────────────────────────────────────────────────────────────────────
def makedir(folder: str):
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise Exception(f'Output folder {folder} already exists.')


# ─────────────────────────────────────────────────────────────────────────────
def _load_groomer_from_folder(folder: str, agent_type: str):
    """
    Load a trained groomer from an output folder.
    Detects whether the folder contains a legacy Keras model or an SB3 model.
    """
    sb3_meta_path = os.path.join(folder, 'sb3_meta.json')
    weights_zip   = os.path.join(folder, 'weights.zip')
    weights_h5    = os.path.join(folder, 'weights.h5')

    if os.path.exists(sb3_meta_path):
        # SB3 model
        if not _SB3_AVAILABLE:
            raise ImportError("SB3 model detected but stable_baselines3 not installed.")
        with open(sb3_meta_path) as f:
            meta = json.load(f)
        actual_agent_type = meta.get('agent_type', agent_type)
        agent = SB3AgentGroom.load(
            actual_agent_type,
            os.path.join(folder, 'weights'),
        )
        return agent.groomer(), actual_agent_type

    elif os.path.exists(weights_h5):
        # Legacy Keras model
        if not _LEGACY_AVAILABLE:
            raise ImportError("Legacy .h5 weights found but keras-rl not installed.")
        groomer = Groomer()
        groomer.load_with_json(
            os.path.join(folder, 'model.json'),
            weights_h5,
        )
        return groomer, 'legacy'

    else:
        raise FileNotFoundError(
            f"No recognised model weights found in {folder}. "
            "Expected 'weights.h5' (legacy) or 'weights.zip' + 'sb3_meta.json' (SB3)."
        )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Train / evaluate / explain a GroomRL agent.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── positional / original flags ──────────────────────────────────────────
    parser.add_argument('runcard', nargs='?', default=None,
                        help='JSON runcard file for a new training run.')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to an existing model folder.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output folder (overrides runcard basename).')
    parser.add_argument('--plot', action='store_true', dest='plot',
                        help='Generate mass and Lund-plane plots.')
    parser.add_argument('--force', '-f', action='store_true', dest='force',
                        help='Overwrite existing output folder.')
    parser.add_argument('--cpp', action='store_true', dest='cpp',
                        help='Export model to C++ (legacy only).')
    parser.add_argument('--data', type=str, default=None,
                        help='Additional data file for plotting.')
    parser.add_argument('--nev', '-n', type=float, default=-1,
                        help='Number of events (−1 = all).')

    # ── new flags ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--agent', type=str, default='legacy',
        choices=['legacy', 'dqn', 'ppo', 'mdpo'],
        help=(
            'RL backend: "legacy" = original keras-rl DQN (default), '
            '"dqn" = SB3 DQN, "ppo" = SB3 PPO, "mdpo" = SB3 MDPO.'
        ),
    )
    parser.add_argument(
        '--xrl', action='store_true', dest='xrl',
        help='Run the Explainable-RL (XRL) analysis pipeline after training.',
    )
    parser.add_argument(
        '--xrl-events', type=int, default=-1, metavar='N',
        help='Max events for XRL analysis (−1 = all; default: −1).',
    )
    parser.add_argument(
        '--xrl-depth', type=int, default=4, metavar='D',
        help='Max depth of surrogate Decision Tree (default: 4).',
    )
    parser.add_argument(
        '--no-gnn', action='store_true', dest='no_gnn',
        help='Disable GNN / LogicXGNN sub-pipeline in XRL.',
    )

    parser.add_argument(
        '--ibmdp-lmut', action='store_true', dest='ibmdp_lmut',
        help='Run the integrated IBMDP + LMUT training pipeline.',
    )
    parser.add_argument(
        '--ibmdp-depth', type=int, default=4, metavar='D',
        help='IBMDP max DT depth (default: 4).',
    )
    parser.add_argument(
        '--ibmdp-p', type=int, default=3, metavar='P',
        help='IBMDP split granularity p (default: 3).',
    )
    parser.add_argument(
        '--ibmdp-zeta', type=float, default=-0.01, metavar='Z',
        help='IBMDP info-gathering penalty zeta (default: -0.01).',
    )

    args = parser.parse_args()

    # ── validate inputs ──────────────────────────────────────────────────────
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        parser.error('Provide either a runcard OR --model, not both.')
    if args.runcard and not os.path.isfile(args.runcard):
        parser.error(f'Runcard not found: {args.runcard}')
    if args.model and not (args.plot or args.cpp or args.data or args.xrl):
        parser.error('--model requires at least one action: --plot, --cpp, --data, or --xrl.')
    if args.agent in ('dqn', 'ppo', 'mdpo') and not _SB3_AVAILABLE:
        parser.error(
            f'--agent {args.agent} requires stable_baselines3.  '
            'Install it with:  pip install stable-baselines3'
        )
    if args.xrl and not _XRL_AVAILABLE:
        parser.error(
            '--xrl requires additional dependencies (sklearn, matplotlib).  '
            'Check your installation.'
        )
    if args.force:
        print('WARNING: --force will overwrite existing model.')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TRAINING path
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.runcard:
        setup = load_json(args.runcard, agent_type=args.agent)

        base = os.path.splitext(os.path.basename(args.runcard))[0]
        out  = args.output if args.output else base
        out  = f'./{out}/{args.agent}_groomer/'

        try:
            makedir(out)
        except Exception as err:
            if args.force:
                print(f'WARNING: Overwriting {out}')
                shutil.rmtree(out)
                makedir(out)
            else:
                print(err)
                print('Delete the folder or use --force to overwrite.')
                raise SystemExit(1)

        setup['output'] = out
        setup['groomer_agent']['tensorboard_log'] = f'./{out}/tensorboard/'
        copyfile(args.runcard, f'{out}/runcard.json')

        # Hyperopt scan (legacy only)
        if setup.get('scan'):
            if args.agent != 'legacy':
                print('WARNING: hyperopt scan is only supported with --agent legacy.  '
                      'Disabling scan.')
                setup['scan'] = False
                groomer_agent_setup = setup
            else:
                groomer_agent_setup = run_hyperparameter_scan(setup)
        else:
            groomer_agent_setup = setup

        # ── Train ────────────────────────────────────────────────────────────
        print('[+] Training model …')
        trained_agent = build_and_train_model(groomer_agent_setup)

        # ── Save runcard ─────────────────────────────────────────────────────
        runcard_out = {k: v for k, v in groomer_agent_setup.items()
                       if k != 'scan'}
        with open(f'{out}/runcard.json', 'w') as f:
            json.dump(runcard_out, f, indent=4)

        # ── Test grooming ────────────────────────────────────────────────────
        fnres = f'{out}/test_predictions.pickle'
        print('[+] Testing on sample set …')
        if os.path.exists(fnres):
            os.remove(fnres)

        groomer = trained_agent.groomer()
        reader  = Jets(setup['test']['fn'], args.nev)
        groomed_jets = [deepcopy(groomer(jet)) for jet in reader.values()]
        with open(fnres, 'wb') as wfp:
            pickle.dump(groomed_jets, wfp)

            # ── Auto-generate reward decomposition plot for SB3 agents ───────────
        if args.agent != 'legacy' and hasattr(trained_agent, 'history') \
                and trained_agent.history is not None:
            plotdir = f'{out}/plots'
            os.makedirs(plotdir, exist_ok=True)
            print('[+] Generating reward decomposition plot …')
            plot_reward_decomposition(
                trained_agent.history.history,
                output_folder=plotdir,
                agent_label=args.agent.upper(),
            )

        folder = out

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LOAD EXISTING MODEL path
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif args.model:
        folder = args.model.rstrip('/')
        setup  = load_runcard(f'{folder}/runcard.json')
        groomer_agent_setup = setup

        groomer, _detected_agent = _load_groomer_from_folder(folder, args.agent)

        trained_agent = None   # no training object available

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  POST-TRAINING ACTIONS
    # ━━━━━━━━━━━━━━━━━━
    # ── XRL ──────────────────────────────────────────────────────────────────
    if args.xrl:
        print('\n[+] Running XRL analysis …')
        # Determine the groomer to use
        if 'groomer' not in dir():
            groomer = trained_agent.groomer() if trained_agent else None
        if groomer is None:
            print('[!] XRL: could not obtain groomer – skipping.')
        else:
            reader = Jets(setup['test']['fn'], args.xrl_events)
            events = reader.values()
            analyzer = XRLAnalyzer(
                groomer=groomer,
                events=events,
                output_dir=folder,
                max_events=args.xrl_events,
                dt_max_depth=args.xrl_depth,
                run_gnn=not args.no_gnn,
                lund_dim=setup['groomer_env'].get('state_dim', 2),
            )
            xrl_summary = analyzer.run()
            print(f"[+] XRL outputs written to:  {folder}/xrl/")

    # ── IBMDP + LMUT ─────────────────────────────────────────────────────────
    if getattr(args, 'ibmdp_lmut', False):
        try:
            from groomrl.ibmdp_lmut_driver import run_ibmdp_lmut
            print('\n[+] Running IBMDP + LMUT integration pipeline ...')
            run_ibmdp_lmut(
                setup       = setup,
                output_dir  = folder,
                nstep       = setup['groomer_agent'].get('nstep', 200_000),
                agent_type  = args.agent if args.agent in ('dqn', 'ppo') else 'dqn',
                p           = args.ibmdp_p,
                zeta        = args.ibmdp_zeta,
                gamma_b     = setup['groomer_agent'].get('gamma', 0.99),
                max_depth   = args.ibmdp_depth,
                nev_eval    = int(args.nev),
                verbose     = 1,
            )
            print(f'[+] IBMDP+LMUT outputs written to:  {folder}/ibmdp_lmut/')
        except ImportError as e:
            print(f'[!] IBMDP+LMUT skipped: {e}')

    # ── Plotting ─────────────────────────────────────────────────────────────
    if args.plot:
        plotdir = f'{folder}/plots'
        try:
            makedir(plotdir)
        except Exception:
            print(f'[+] Skipping plot: {plotdir} already exists.')
        else:
            print(f'[+] Creating test plots in {plotdir}')
            if 'groomer' not in dir():
                groomer = trained_agent.groomer()
            plot_mass(groomer, setup['test']['fn'],
                      mass_ref=setup['groomer_env']['mass'],
                      output_folder=plotdir, nev=args.nev)
            plot_lund(groomer, setup['test']['fn'],
                      output_folder=plotdir, nev=args.nev)
            
            if 'fn_bkg' in setup['test']:
                plot_mass(groomer, setup['test']['fn_bkg'],
                          mass_ref=setup['groomer_env']['mass'],
                          output_folder=plotdir, nev=args.nev, background=True)
                plot_lund(groomer, setup['test']['fn_bkg'],
                          output_folder=plotdir, nev=args.nev, background=True)

    # ── Extra data ────────────────────────────────────────────────────────────
    if args.data:
        fn_base = os.path.splitext(os.path.basename(args.data))[0]
        plotdir = f'{folder}/{fn_base}'
        try:
            makedir(plotdir)
        except Exception:
            print(f'[+] Skipping data: {plotdir} already exists.')
        else:
            if 'groomer' not in dir():
                groomer = trained_agent.groomer()
            plot_mass(groomer, args.data,
                      mass_ref=setup['groomer_env']['mass'],
                      output_folder=plotdir, nev=args.nev)
            plot_lund(groomer, args.data, output_folder=plotdir, nev=args.nev)

    # ── C++ export ─────────────────────────────────────────────────────────────
    if args.cpp:
        if args.agent != 'legacy':
            print('[!] --cpp is only supported for legacy (keras-rl) models.  Skipping.')
        elif not _LEGACY_AVAILABLE:
            print('[!] --cpp requires keras-rl.  Skipping.')
        else:
            check_model(groomer_agent_setup['groomer_agent'])
            cppdir = f'{folder}/cpp'
            try:
                makedir(cppdir)
            except Exception:
                print(f'[+] Skipping cpp: {cppdir} already exists.')
            else:
                print(f'[+] Adding C++ model to {cppdir}')
                if 'groomer' not in dir():
                    groomer = trained_agent.groomer()
                cpp_fn = f'{cppdir}/model.nnet'
                arch_dic = ast.literal_eval(
                    groomer.model.to_json()
                    .replace('true', 'True')
                    .replace('null', 'None')
                )
                keras_to_cpp(groomer.model, arch_dic['config']['layers'], cpp_fn)


if __name__ == '__main__':
    main()
