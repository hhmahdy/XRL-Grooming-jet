# models.py  –  part of GroomRL (modified for SB3 + XRL integration)
#
# Original authors: S. Carrazza and F. A. Dreyer
# SB3 / XRL additions: GroomRL-XRL project
"""
Model construction and training for GroomRL.

This module exposes a single top-level function ``build_and_train_model`` that
routes to either the *legacy* keras-rl DQN path or one of the new
Stable-Baselines3 paths (DQN / PPO) depending on the ``agent`` key in the
``groomer_agent_setup`` dictionary.

Routing table
-------------
setup['agent'] == 'legacy'  (default)  →  original DQNAgentGroom (keras-rl)
setup['agent'] == 'dqn'                →  SB3AgentGroom with SB3 DQN
setup['agent'] == 'ppo'                →  SB3AgentGroom with SB3 PPO
"""

from __future__ import annotations

import json
import os
import pprint
from time import time
from typing import Optional

import numpy as np

# ── GroomRL internals ──────────────────────────────────────────────────────
from groomrl.JetTree import LundCoordinates
from groomrl.read_data import Jets
from groomrl.tools import get_window_width, mass

# ── Legacy (keras-rl) path ─────────────────────────────────────────────────
try:
    from groomrl.GroomEnv import GroomEnv, GroomEnvDual, GroomEnvTriple
    from groomrl.DQNAgentGroom import DQNAgentGroom
    from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
    from rl.memory import SequentialMemory
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
    from keras.optimizers import Adam, SGD, RMSprop, Adagrad
    from keras.callbacks import TensorBoard
    import keras.backend as K
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False

# ── SB3 path ───────────────────────────────────────────────────────────────
try:
    from groomrl.GroomEnvSB3 import make_sb3_env
    from groomrl.SB3AgentGroom import SB3AgentGroom
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    from hyperopt import STATUS_OK
except ImportError:
    STATUS_OK = "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy (keras-rl) helpers – unchanged from original
# ══════════════════════════════════════════════════════════════════════════════

def build_model(hps, input_dim):
    """Construct the Keras model used by the legacy DQN."""
    if not _KERAS_AVAILABLE:
        raise ImportError("Keras / keras-rl is not installed.")
    K.clear_session()
    model = Sequential()
    if hps['architecture'] == 'Dense':
        model.add(Flatten(input_shape=(1,) + input_dim))
        for _ in range(hps['nb_layers']):
            model.add(Dense(hps['nb_units']))
            model.add(Activation('relu'))
        if hps['dropout'] > 0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(2))
        model.add(Activation('linear'))
    elif hps['architecture'] == 'LSTM':
        model.add(LSTM(hps['nb_units'],
                       input_shape=(1, max(input_dim)),
                       return_sequences=not (hps['nb_layers'] == 1)))
        for i in range(hps['nb_layers'] - 1):
            model.add(LSTM(hps['nb_units'],
                           return_sequences=not (i + 2 == hps['nb_layers'])))
        if hps['dropout'] > 0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(2))
        model.add(Activation('linear'))
    print(model.summary())
    return model


def build_dqn(hps, input_dim):
    """Create a legacy keras-rl DQN agent."""
    if not _KERAS_AVAILABLE:
        raise ImportError("Keras / keras-rl is not installed.")
    print('[+] Constructing DQN agent, model setup:')
    pprint.pprint(hps)

    model  = build_model(hps, input_dim)
    memory = SequentialMemory(limit=500_000, window_length=1)

    if hps["policy"] == "boltzmann":
        policy = BoltzmannQPolicy()
    elif hps["policy"] == "epsgreedyq":
        policy = EpsGreedyQPolicy()
    else:
        raise ValueError("Invalid policy: %s" % hps["policy"])

    agent = DQNAgentGroom(
        model=model, nb_actions=2,
        enable_dueling_network=hps["enable_dueling_network"],
        enable_double_dqn=hps["enable_double_dqn"],
        memory=memory, nb_steps_warmup=500,
        target_model_update=1e-2, policy=policy,
    )

    optimizers = {"Adam": Adam, "SGD": SGD, "RMSprop": RMSprop, "Adagrad": Adagrad}
    opt_cls = optimizers.get(hps['optimizer'], Adam)
    agent.compile(opt_cls(lr=hps['learning_rate']), metrics=['mae'])
    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_runcard(runcard: str) -> dict:
    """Read a JSON runcard and initialise LundCoordinates dimensions."""
    with open(runcard, 'r') as f:
        res = json.load(f)
    env_setup = res.get("groomer_env")
    if not isinstance(env_setup["state_dim"], str):
        LundCoordinates.change_dimension(env_setup["state_dim"])
    return res


def loss_calc(dqn, fn_sig, fn_bkg, nev, massref):
    """Compute the hyperparameter scan loss (legacy path only)."""
    reader_sig = Jets(fn_sig, nev)
    reader_bkg = Jets(fn_bkg, nev)
    groomed_sig = [dqn.groomer()(jet) for jet in reader_sig.values()]
    masses_sig  = np.array(mass(groomed_sig))
    lower, upper, median = get_window_width(masses_sig)
    groomed_bkg = [dqn.groomer()(jet) for jet in reader_bkg.values()]
    masses_bkg  = np.array(mass(groomed_bkg))
    count_bkg = ((masses_bkg > lower) & (masses_bkg < upper)).sum()
    frac_bkg  = count_bkg / float(len(masses_bkg))
    loss = abs(upper - lower) / 5 + abs(median - massref) + frac_bkg * 20
    return loss, (lower, upper, median)


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train_model(groomer_agent_setup: dict):
    """
    Build and train a GroomRL agent.

    The agent type is controlled by ``groomer_agent_setup['agent']``:

    * ``"legacy"`` (default) – original keras-rl DQN.
    * ``"dqn"``              – Stable-Baselines3 DQN.
    * ``"ppo"``              – Stable-Baselines3 PPO.

    Returns the trained agent (DQNAgentGroom or SB3AgentGroom).
    """
    env_setup    = groomer_agent_setup.get('groomer_env')
    agent_setup  = groomer_agent_setup.get('groomer_agent')
    agent_type   = groomer_agent_setup.get('agent', 'legacy').lower()

    # Ensure LundCoordinates dimension is in sync
    if env_setup["state_dim"] != LundCoordinates.dimension:
        LundCoordinates.change_dimension(env_setup["state_dim"])

    print(f"[+] Agent type: {agent_type}")

    # ── SB3 path ─────────────────────────────────────────────────────────────
    if agent_type in ("dqn", "ppo", "mdpo"):
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is not installed – "
                "cannot use agent_type='%s'." % agent_type
            )

        groomer_env = make_sb3_env(env_setup, LundCoordinates.low, LundCoordinates.high)
        agent = SB3AgentGroom(agent_type, groomer_env, agent_setup)
        agent.fit(total_timesteps=agent_setup.get('nstep', 50_000))

        # Persist
        if not groomer_agent_setup.get('scan', False):
            out = groomer_agent_setup['output']
            weight_file = f'{out}/weights'   # SB3 adds .zip
            print(f'[+] Saving SB3 model to {weight_file}.zip')
            agent.save_weights(weight_file)

            # Save a small metadata JSON so the load path knows the agent type
            meta = {
                "agent_type":  agent_type,
                "state_dim":   env_setup["state_dim"],
                "architecture": "SB3-MlpPolicy",
            }
            with open(f'{out}/sb3_meta.json', 'w') as f:
                json.dump(meta, f, indent=4)

        return agent

    # ── Legacy (keras-rl) path ────────────────────────────────────────────────
    if not _KERAS_AVAILABLE:
        raise ImportError(
            "Keras / keras-rl is not installed – "
            "cannot use agent_type='legacy'."
        )

    if env_setup.get("dual_groomer_env"):
        groomer_env = GroomEnvDual(env_setup,
                                   low=LundCoordinates.low,
                                   high=LundCoordinates.high)
    elif env_setup.get("triple_groomer_env"):
        groomer_env = GroomEnvTriple(env_setup,
                                     low=LundCoordinates.low,
                                     high=LundCoordinates.high)
    else:
        groomer_env = GroomEnv(env_setup,
                               low=LundCoordinates.low,
                               high=LundCoordinates.high)

    dqn = build_dqn(agent_setup, groomer_env.observation_space.shape)

    logdir = '%s/logs/{}'.format(time()) % groomer_agent_setup['output']
    print(f'[+] TensorBoard log → {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('[+] Fitting legacy DQN agent …')
    r = dqn.fit(groomer_env,
                nb_steps=agent_setup['nstep'],
                visualize=False,
                verbose=1,
                callbacks=[tensorboard])

    median_reward = np.median(r.history['episode_reward'])
    print(f'[+] Median reward: {median_reward}')

    if not groomer_agent_setup.get('scan', False):
        out = groomer_agent_setup['output']
        weight_file = f'{out}/weights.h5'
        model_file  = f'{out}/model.json'
        print(f'[+] Saving weights → {weight_file}')
        dqn.save_weights(weight_file, overwrite=True)
        print(f'[+] Saving model  → {model_file}')
        with open(model_file, 'w') as f:
            json.dump(dqn.model.to_json(), f)

    if groomer_agent_setup.get('scan', False):
        loss, window = loss_calc(
            dqn,
            env_setup['val'], env_setup['val_bkg'],
            env_setup['nev_val'], env_setup['mass'],
        )
        print(f'Loss function for scan = {loss}')
        return {
            'loss': loss,
            'reward': median_reward,
            'window': window,
            'status': STATUS_OK,
        }

    return dqn
