import argparse
import json
import random
import sys


import numpy as np
import torch
import torch.utils.data

sys.path.insert(1, '../')

from domainbed import hparams_registry, datasets, algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

from feature_importance.feature_importance import FeatureImportance


def get_prediction(algorithm, feature, y):
    correct = 0
    total = 0

    feature = algorithm.network.avgpool(feature)
    feature = torch.flatten(feature, 1)
    feature = algorithm.network.fc(feature)
    correct += (feature.argmax(1).eq(y).float()).sum().item()
    total += feature.size(0)

    return correct, total

def get_prediction_domain(algorithm, feature, y):
    correct = 0
    total = 0

    feature = algorithm.domain_classifier(feature)
    correct += (feature.argmax(1).eq(y).float()).sum().item()
    total += feature.size(0)

    return correct, total

def test_ablation(dataloader, model, optimiser, abla_percentage, ablation_type, device):
    correct = 0
    total = 0 
    total_change = 0
    count = 0
    for d in dataloader:
        for i, batch in enumerate(d):
            x, y = batch[0].to(device), batch[1].to(device)
            class_gradient = None
            feature = model.get_feature(x).detach()
            feature.requires_grad_(True)
            y_c = model.network.avgpool(feature)
            y_c = torch.flatten(y_c, 1)
            y_c = model.network.fc(y_c)
            loss_c = sum([y_c[i][y[i].item()] for i in range(y.size(0))])
            optimiser.zero_grad()
            loss_c.backward(retain_graph=True)
            class_gradient = feature.grad

            feature = feature.detach()
            feature_import = FeatureImportance(feature, class_gradient, abla_percentage)

            if ablation_type == 'random':
                map, change = feature_import.random()
            elif ablation_type == 'gradient_norm':
                map, change = feature_import.gradient_norm()
            elif ablation_type == 'gradient_feature':
                map, change = feature_import.gradient_and_feature_pool()

            total_change += change
            updated_features = feature * map.to(device)

            current_correct, current_total = get_prediction(model, updated_features, y)
            correct += current_correct
            total += current_total
            count += 1

    print('{} class accuracy {} change percentage {}'.format(ablation_type, correct/total, total_change/count))
            
def test_ablation_domain(dataloader, model, optimiser, abla_percentage, ablation_type, device):
    correct = 0
    total = 0 
    total_change = 0
    count = 0
    count_d = -1
    for d in dataloader:
        count_d += 1
        for i, batch in enumerate(d):
            x, y = batch[0].to(device), batch[1].to(device)
            g = torch.zeros(x.size(0)).to(device) + count_d
            g = g.long()   
            domain_gradient = None
            feature = model.get_feature(x).detach()
            feature.requires_grad_(True)
            g_hat_result = model.domain_classifier(feature)
            loss_d = sum([g_hat_result[i][g[i].item()] for i in range(g.size(0))])
            optimiser.zero_grad()
            loss_d.backward(retain_graph=True)
            domain_gradient = feature.grad

            feature = feature.detach()
            feature_import = FeatureImportance(feature, domain_gradient, abla_percentage)

            if ablation_type == 'random':
                map, change = feature_import.random()
            elif ablation_type == 'gradient_norm':
                map, change = feature_import.gradient_norm()
            elif ablation_type == 'gradient_feature':
                map, change = feature_import.gradient_and_feature_pool()

            total_change += change
            updated_features = feature * map.to(device)

            current_correct, current_total = get_prediction_domain(model, updated_features, g)
            correct += current_correct
            total += current_total
            count += 1

    print('{} domain accuracy {} change percentage {}'.format(ablation_type, correct/total, total_change/count))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')

    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    
    args = parser.parse_args()


    '''load hyper parameters for the algorithm'''
    if args.hparams_seed == 0:
        print('Load default hyper parameters')
        hparams = hparams_registry.default_hparams('xdomain_mix', args.dataset)
    else:
        print('Generate hyper parameters randomly')
        hparams = hparams_registry.random_hparams('xdomain_mix', args.dataset,
            misc.seed_hash(args.hparams_seed))
    if args.hparams:
        print('Load hyper parameters from', args.hparams)
        hparams.update(json.loads(args.hparams))


    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))


    '''set seeds and backend attribute'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    '''set device and dataset'''
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    '''create dataset loader'''
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    in_splits = []
    out_splits = []
    test_splits = []

    eval_loader_names = []
    val_loader_names = []
    test_loader_names = []

    for env_i, env in enumerate(dataset):
        if env_i in args.test_envs:
            test_splits.append(env)
            test_loader_names.append('env{}'.format(env_i))
        else:
            out, in_ = misc.split_dataset(env, int(len(env)*args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
            in_splits.append(in_)
            out_splits.append(out)
            eval_loader_names.append('env{}_in'.format(env_i))
            val_loader_names.append('env{}_out'.format(env_i))
    
    train_num = [len(env) for env in in_splits]
    eval_num = train_num
    val_num = [len(env) for env in out_splits]
    test_num = [len(env) for env in test_splits] 
  
    val_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=4)
        for env in out_splits]

    network = algorithms.xdomain_mix(dataset.input_shape, dataset.num_classes,len(dataset) - len(args.test_envs), hparams)
    network.to(device)
    optimizer = torch.optim.Adam(
            network.network.network.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams['weight_decay']
        )
    domain_optimizer = torch.optim.Adam(
            network.network.domain_classifier.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams['weight_decay']
        )

    network.load_state_dict(torch.load(args.output_dir + '/model.pkl')['model_dict'])
    for r in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
        test_ablation(val_loaders, network.network, optimizer, r, 'random', device)
        test_ablation(val_loaders, network.network, optimizer, r, 'gradient_norm', device)
        test_ablation(val_loaders, network.network, optimizer, r, 'gradient_feature', device)

    for r in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
        test_ablation_domain(val_loaders, network.network, domain_optimizer, r, 'random', device)
        test_ablation_domain(val_loaders, network.network, domain_optimizer, r, 'gradient_norm', device)
        test_ablation_domain(val_loaders, network.network, domain_optimizer, r, 'gradient_feature', device)















       

