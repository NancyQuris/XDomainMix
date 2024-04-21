import argparse
import datetime
import copy 
import time
import json
import os
import pdb
import sys
import csv
import tqdm
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import models

from config import dataset_defaults
from hparams_registry import default_hparams, random_hparams
from utils import unpack_data, sample_domains, save_best_model, \
    Logger, return_predict_fn, return_criterion, fish_step

from pytorch_transformers import AdamW, WarmupLinearSchedule
from domainclassmix_module import DomainLearner, DomainClassMixAugmentation


runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='WILDS DG test bed')
# General
parser.add_argument('--dataset', type=str, default='camelyon',
                    help="Name of dataset, choose from iwildcam, camelyon, rxrx")
parser.add_argument('--algorithm', type=str, default='xdomain_mix',
                    help='training scheme')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data-dir', type=str, default='../../data/wilds',
                    help='path to data dir')
parser.add_argument('--stratified', action='store_true', default=False,
                    help='whether to use stratified sampling for classes')
parser.add_argument('--num-domains', type=int, default=15,
                    help='Number of domains, only specify for cdsprites')
# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')
parser.add_argument("--n_groups_per_batch", default=4, type=int)
parser.add_argument("--cut_mix", default=True, type=int)
parser.add_argument("--mix_alpha", default=2, type=float)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--group_by_label", default=False, action='store_true')
parser.add_argument("--power", default=0, type=float)
parser.add_argument("--reweight_groups", default=False, action='store_true')
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--scheduler", default=None, type=str)

# for hyper parameter searching 
parser.add_argument("--random_hparams", action='store_true', default=False, 
                    help='enable hyperparameter search')

# parameters about training twice
parser.add_argument("--save_pred", default=False, action='store_true')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--use_bert_params", default=1, type=int)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--warmup_steps", default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
print(args.seed)
torch.backends.cudnn.deterministic = True

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
directory_name = '../experiments/{}'.format(args.experiment)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# for testing
# runPath = '../experiments/camelyon_sam_1567879618/2023-11-15T12_21_14.355588d_c4j0f_/'
# runPath = ''

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

if args.algorithm in ['xdomain_mix', 'rsc', 'xdomain_mix_sam', 'clip_xdomain_mix']:
    if args.random_hparams:
        hparams = random_hparams(args.seed)
    else:
        hparams = default_hparams()
    print(hparams)
    with open('{}/args.json'.format(runPath), 'w') as fp:
        json.dump(hparams, fp)
    torch.save(hparams, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
if args.group_by_label: args.n_groups_per_batch = 2
if args.algorithm == 'lisa': args.batch_size //= 2
if args.dataset == 'camelyon': args.n_groups_per_batch = 3
train_loader, tv_loaders = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']

n_class = getattr(models, f"{args.dataset}_n_class")
n_domain = getattr(models, f"{args.dataset}_n_domain")
feature_dim = getattr(models, f"{args.dataset}_feature_dim")

# create model based on method
if args.algorithm not in  ['xdomain_mix', 'xdomain_mix_sam', 'clip_xdomain_mix']:
    model = modelC(args, weights=None, use_clip=True).to(device)
else:
    model = modelC(args, weights=None, mixup_module=DomainClassMixAugmentation(args.batch_size, n_class, n_domain, hparams), use_clip=True).to(device)
    

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)
if args.use_bert_params and args.dataset == 'civil':
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
                0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
                0.0,
        },
    ]
    optimiserC = opt(optimizer_grouped_parameters, **args.optimiser_args)
else:
    optimiserC = opt(model.classifier.parameters(), **args.optimiser_args)

# create optimizer for domain classifier
if args.algorithm in ['xdomain_mix', 'xdomain_mix_sam', 'clip_xdomain_mix']:
    domain_classifier = torch.nn.Linear(512, n_domain).to(device)
    optimiserD = opt(domain_classifier.parameters(), **args.optimiser_args)

predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)


def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts

def train_clip_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model.clip_forward(x)
        loss = criterion(y_hat, y)
        loss.backward()
        if args.use_bert_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
        optimiserC.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)

def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        if args.use_bert_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
        optimiserC.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)


def train_clip_xdomain_mix(train_loader, epoch, agg):
    running_loss = 0
    domain_loss = 0
    total_iters = len(train_loader)

    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        x, y, g = data[0].to(device), data[1].to(device), data[2].to(device)
        
        if (i + total_iters*epoch) < hparams['warmup_step']:
            optimiserC.zero_grad()
            y_hat = model.clip_forward(x)
            loss = criterion(y_hat, y)
            loss.backward()
            if args.use_bert_params:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)
            optimiserC.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
        else:
            if (i + total_iters*epoch) == hparams['warmup_step']:
                print('start mixup')
            
            class_gradient = None
            feature_c = model.clip_get_feature(x).detach()
            feature_c.requires_grad_(True)
            y_c = model.clip_get_result(feature_c)
            loss_c = sum([y_c[i][y[i].item()] for i in range(y.size(0))])
            optimiserC.zero_grad()
            loss_c.backward(retain_graph=True)
            class_gradient = feature_c.grad

            domain_gradient = None
            features = model.clip_get_feature(x).detach()
            features.requires_grad_(True)
            g_hat_result = domain_classifier(features)
            loss_d = sum([g_hat_result[i][g[i].item()] for i in range(g.size(0))])
            optimiserD.zero_grad()
            loss_d.backward(retain_graph=True)
            domain_gradient = features.grad

            y_hat = model.clip_forward(x)
            loss_1 = criterion(y_hat, y)
            loss_2 = criterion(model.clip_train_f(x, y, g, class_gradient, domain_gradient), y)
            loss = 0.5*loss_1 + 0.5*loss_2
            optimiserC.zero_grad()
            loss.backward()
            if args.use_bert_params:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)
            optimiserC.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()

        features = model.clip_get_feature(x).detach()
        g_hat = domain_classifier(features)
        loss_domain = F.cross_entropy(g_hat, g)
        optimiserD.zero_grad()
        loss_domain.backward()
        optimiserD.step()
        domain_loss += loss_domain.item()

        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['domain_loss'].append(domain_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f} domain loss: {:6.3f}'.format(i + 1, \
                    total_iters, running_loss / args.print_loss_iters, domain_loss / args.print_loss_iters))
            running_loss = 0.0
            domain_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (i + total_iters*epoch) >= hparams['warmup_step'] and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)
            save_best_model(domain_classifier, runPath, agg, name='DomainClassifier')


def save_pred(model, train_loader, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    yhats, ys, idxes = [], [], []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            model.eval()
            x, y, idx = data[0].to(device), data[1].to(device), data[-1].to(device)
            y_hat = model(x)
            ys.append(y.cpu())
            yhats.append(y_hat.cpu())
            idxes.append(idx.cpu())
            # if i > 10:
            #     break

        ypreds, ys, idxes = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(idxes)

        ypreds = ypreds[torch.argsort(idxes)]
        ys = ys[torch.argsort(idxes)]

        y = torch.cat([ys.reshape(-1, 1), ypreds.reshape(-1, 1)], dim=1)

        df = pd.DataFrame(y.cpu().numpy(), columns=['y_true', 'y_pred'])
        df.to_csv(os.path.join(save_dir, f"{args.dataset}_{args.algorithm}_{epoch}.csv"))

        # print accuracy
        wrong_labels = (df['y_true'].values == df['y_pred'].values).astype(int)
        from wilds.common.grouper import CombinatorialGrouper
        grouper = CombinatorialGrouper(train_loader.dataset.dataset.dataset, ['y', 'black'])
        group_array = grouper.metadata_to_group(train_loader.dataset.dataset.dataset.metadata_array).numpy()
        group_array = group_array[np.where(
            train_loader.dataset.dataset.dataset.split_array == train_loader.dataset.dataset.dataset.split_dict[
                'train'])]
        for i in np.unique(group_array):
            idxes = np.where(group_array == i)[0]
            print(f"domain = {i}, length = {len(idxes)}, acc = {np.sum(wrong_labels[idxes] / len(idxes))} ")

        def print_group_info(idxes):
            group_ids, group_counts = np.unique(group_array[idxes], return_counts=True)
            for idx, j in enumerate(group_ids):
                print(f"group[{j}]: {group_counts[idx]} ")

        correct_idxes = np.where(wrong_labels == 1)[0]
        print("correct points:")
        print_group_info(correct_idxes)
        wrong_idxes = np.where(wrong_labels == 0)[0]
        print("wrong points:")
        print_group_info(wrong_idxes)


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None, zero_shot=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            # y_hat = model(x)
            if zero_shot:
                y_hat = model.clip_zero_shot(x)
            else:
                y_hat = model.clip_forward(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            if args.dataset == 'poverty':
                save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                            f"{['A', 'B', 'C', 'D', 'E'][args.seed]}" \
                            f"_epoch:best_pred.csv"
            else:
                save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                            f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        if args.dataset == 'poverty':
            with open(f"{runPath}/{save_name}_ys", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ys.unsqueeze(1).cpu().tolist())
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")



if __name__ == '__main__':
    if args.scheduler == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimiserC,
            num_training_steps=len(train_loader) * args.epochs,
            **args.scheduler_kwargs)
        print("scheduler has been defined")
    elif args.scheduler == 'linear_scheduler':
        t_total = len(train_loader) * args.epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(optimiserC,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    else:
        scheduler = None

    print(
        "=" * 30 + f"Training: {args.algorithm}" + "=" * 30)
    
    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    if args.algorithm == 'clip_zero_shot':
        for split, loader in tv_loaders.items():
            test(loader, agg, loader_type=split, save_ypred=True, zero_shot=True)

    else:
        train = locals()[f'train_{args.algorithm}']
        for epoch in range(args.epochs):
            train(train_loader, epoch, agg)
            
            if args.algorithm == 'clip_xdomain_mix':
                if (len(train_loader)*(epoch+1)) >=  hparams['warmup_step']:
                    test(val_loader, agg, loader_type='val')
                    test(test_loader, agg, loader_type='test')
                    save_best_model(model, runPath, agg)
                    save_best_model(domain_classifier, runPath, agg, name='DomainClassifier')
                    if args.save_pred:
                        save_pred(model, train_loader, epoch, args.save_dir)
            else:
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                save_best_model(model, runPath, agg)
                if args.save_pred:
                    save_pred(model, train_loader, epoch, args.save_dir)

            
        model.load_state_dict(torch.load(runPath + '/model.rar'))
        print('Finished training! Loading best model...')

        for split, loader in tv_loaders.items():
            test(loader, agg, loader_type=split, save_ypred=True)
