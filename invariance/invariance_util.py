import ast
import argparse
import json
import os

import tqdm

from domainbed import datasets
from domainbed import algorithms


def load_data_dir(path):
    train_info = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                            ncols=80,
                            leave=False):
        dir_path = os.path.join(path, subdir)
        result_path = os.path.join(path, subdir, "results.jsonl")
        try: 
            with open(result_path, 'r') as f:
                for line in f:
                    r = json.loads(line[:-1])
                    break
                dir_path = os.path.join(path, subdir, 'model.pkl')
                args = r['args']
                hparms = r['hparams']
                train_info.append((dir_path, args, hparms))
        except IOError:
            pass
        
    return train_info

def parse_dict(args_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    arguments = []
    for arg, value in args_dict.items():
        if arg == 'skip_model_save' or arg == 'save_model_every_checkpoint':
            if value is False:
                continue
        else:
            if value is None:
                continue
            else:
                arguments.append('--{}'.format(arg))
                if arg == 'test_envs':
                    test_envs = ast.literal_eval(str(value))
                    for t in test_envs:
                        arguments.append(str(t))
                else:
                    arguments.append(str(value))
        
    args = parser.parse_args(arguments)
    return args

def load_dataset(args, hparams):
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError
    return dataset

def load_model(device, args, dataset, hparams, algorithm_dict):
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    algorithm.to(device)
    if args.algorithm == 'Fish':
        algorithm.create_clone(device)

    algorithm.load_state_dict(algorithm_dict)
    return algorithm
