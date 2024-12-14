import datetime
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


def prepare_training_data(trajectory):    
        mb_states, mb_actions, mb_rewards, mb_extras = trajectory
        mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
        mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])

        mb_values = np.concatenate([mb_values, [[0]] ])
        mb_values = mb_values.reshape(mb_values.shape[0], 1)

        mb_deltas = mb_rewards + 0.99 * mb_values[1:]  - mb_values[:-1]
        nsteps = len(mb_states)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            mb_advs[t] = lastgaelam = mb_deltas[t] + 0.99 * 0.95 * lastgaelam

        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            return np.squeeze(arr, axis = 1)
            # s = arr.shape
            # return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        mb_returns = mb_advs + mb_values[:-1]
        data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp]]
        name = ['state', 'return', 'action', 'value', 'neglogp']
        return dict(zip(name, data))

def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id


def load_yaml_config(args, role_type: str) -> None:
    if role_type not in {'actor', 'learner'}:
        raise ValueError('Invalid role type')

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = None

    if config is not None and isinstance(config, dict):
        if role_type in config:
            for k, v in config[role_type].items():
                if k in args:
                    setattr(args, k, v)
                else:
                    warnings.warn(f"Invalid config item '{k}' ignored", RuntimeWarning)
        args.model_config = {'model': config['model']} if 'model' in config else None
    else:
        args.model_config = None


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()
