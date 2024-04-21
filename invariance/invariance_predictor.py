import argparse
import datetime
import json
import os
import random
import sys
import time

import math
import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, '../')

from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

from invariance import invariance_util

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)

    args = parser.parse_args()

    results_file = args.results_file 

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = invariance_util.load_data_dir(args.input_dir)
    print("Total dirs:", len(records))

    device = "cpu"

    result_by_env = {}
    result_by_seed = {}
    for r in records:
        dir_path, r_args, hparms = r
        r_args = invariance_util.parse_dict(r_args) 
        # load dataset
        dataset = invariance_util.load_dataset(r_args, hparms)

        in_splits = []
        out_splits = []
        test_splits = []

        eval_loader_names = []
        val_loader_names = []
        test_loader_names = []

        for env_i, env in enumerate(dataset):
            if env_i in r_args.test_envs:
                test_splits.append(env)
                test_loader_names.append('env{}'.format(env_i))
            else:
                out, in_ = misc.split_dataset(env, int(len(env)*r_args.holdout_fraction), misc.seed_hash(r_args.trial_seed, env_i))
                in_splits.append(in_)
                out_splits.append(out)
                eval_loader_names.append('env{}_in'.format(env_i))
                val_loader_names.append('env{}_out'.format(env_i))
        
        train_num = [len(env) for env in in_splits]
        val_num = [len(env) for env in out_splits]
        test_num = [len(env) for env in test_splits] 

        val_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparms['batch_size'],
            num_workers=1)
            for env in out_splits]
        val_weights = [None for env in out_splits]

        # load model
        algorithm_dict = torch.load(dir_path)
        algorithm = invariance_util.load_model(device, r_args, dataset, hparms, algorithm_dict['model_dict'])
        
        algo = algorithm.network

        steps_per_epoch = min([len(env)/hparms['batch_size'] for env in out_splits])
        train_minibatches_iterator = zip(*val_loaders)
        
        loss = torch.nn.CrossEntropyLoss()

        invariance_result = [[] for i in range(len(dataset) - len(r_args.test_envs))]
            
        for l in range(int(steps_per_epoch)+1):
            step_start_time = time.time()
            minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)]
            uda_device = None


            for j in range(len(dataset) - len(r_args.test_envs)):
                x, y = minibatches_device[j]
                y_hat = algo(x)
                invariance_result[j].append(loss(y_hat, y).item())

        final_result = []

        for i in range(len(dataset) - len(r_args.test_envs)):
            if len(invariance_result[i]) != 0:
                final_result.append(torch.mean(torch.tensor(invariance_result[i])).item())
        
        print(r_args.algorithm, torch.var(torch.tensor(final_result)).item())