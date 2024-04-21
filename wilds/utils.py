import pdb

import numpy as np
import os
import random
import shutil
import sys
import operator
from numbers import Number
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    filepath = filepath
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def unpack_data(data, device):
    return data[0].to(device), data[1].to(device)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        if hasattr(dataset, 'images'):
            self.images = dataset.images[indices]
            self.latents = dataset.latents[indices, :]
        else:
            self.targets = dataset.targets[indices]
            self.writers = dataset.domains[indices]
            self.data = [dataset.data[i] for i in indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def sample_domains(train_loader, N=1, stratified=True, power=0):
    """
    Sample N domains available in the train loader.
    """

    if N > train_loader.dataset.num_envs:
        return torch.arange(train_loader.dataset.num_envs)

    Ls = []
    probs = []
    assert len(train_loader.dataset.batches_left.values()) == len(train_loader.dataset.domain_counts)

    for i, tl in enumerate(train_loader.dataset.batches_left.values()):
        # stratified means no repetition in the selected domains
        Ls.append(max(tl, 0)) if stratified else Ls.append(min(tl, 1))
        # probs.append((train_loader.dataset.domain_counts[i]) ** power)
        Ls[-1] = Ls[-1] * train_loader.dataset.domain_counts[i] ** power

    positions = range(len(Ls))
    indices = []
    while True:
        needed = N - len(indices)
        if not needed:
            break

        for i in random.choices(positions, Ls, k=needed):
            if Ls[i]:
                Ls[i] = 0.0
                indices.append(i)
    return torch.tensor(indices)


def save_best_model(model, runPath, agg, name=None):
    if agg['val_stat'][-1] >= max(agg['val_stat'][:-1]):
        if name is None:
            save_model(model, f'{runPath}/model.rar')
            save_vars(agg, f'{runPath}/losses.rar')
        else:
            save_model(model, f'{runPath}/model_{name}.rar')
            save_vars(agg, f'{runPath}/losses_{name}.rar')


def single_class_predict_fn(yhat):
    _, predicted = torch.max(yhat.data, 1)

    # for debugging purposes
    # print("yhat.argmax(-1):", yhat.argmax(-1))
    # print("predicted:", predicted)
    # assert yhat.argmax(-1) == predicted

    return predicted


def return_predict_fn(dataset):
    return {
        'iwildcam': single_class_predict_fn,
        'camelyon': single_class_predict_fn,
        'rxrx': single_class_predict_fn,
    }[dataset]

def return_criterion(dataset):
    return {
        'iwildcam': nn.CrossEntropyLoss(),
        'camelyon': nn.CrossEntropyLoss(),
        'rxrx': nn.CrossEntropyLoss(),
    }[dataset]


class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights