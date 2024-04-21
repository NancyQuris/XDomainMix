import numpy as np
import torch
import torch.nn as nn
import random 
from contextlib import contextmanager


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def manifold_mix_process(x, y_onehot, args):
    mixed_x, mixed_y = mix_up(args, x, y_onehot)
    return mixed_x, mixed_y



def mix_up(args, x, y, x2=None, y2=None):

    # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]

    if x2 is None:
        idxes = torch.randperm(len(x))
        x1 = x
        x2 = x[idxes]
        y1 = y
        y2 = y[idxes]
    else:
        x1 = x
        y1 = y

    n_classes = y1.shape[1]
    bsz = len(x1)
    l = np.random.beta(args.mix_alpha, args.mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    l_y = np.tile(l, [1, n_classes])

    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

    return mixed_x, mixed_y



def bert_mix_forward(args, x, y_onehot, model):

    x = model.model[0](x)
    x, all_mix_y = manifold_mix_process(x, y_onehot, args)
    x = model.model[1](x)
    return x, all_mix_y



def manifold_mix_forward(model, x, y, y_onehot, args):

    '''
    x: list of x from 4 groups
    y: list of y from 4 groups
    y_onehot: list of y_onehot from 4 groups
    '''

    x = torch.cat(x).cuda()
    y = torch.cat(y)
    y_onehot = torch.cat(y_onehot)

    # lengths: length of x from 4 groups, example: [0, len_1, len_1+len_2, len_1+len_2_len_3, len_1+len_2+len_3+len_4]
    lengths = np.cumsum([0] + [len(x_i) for x_i in x])



    mix_layer = np.random.randint(0, 3)

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    if mix_layer == 0: x, all_mix_y, all_y, all_group = manifold_mix_process(x, y, y_onehot, args, lengths)
    x = model.layer1(x)
    if mix_layer == 1: x, all_mix_y, all_y, all_group = manifold_mix_process(x, y, y_onehot, args, lengths)
    x = model.layer2(x)
    if mix_layer == 2: x, all_mix_y, all_y, all_group = manifold_mix_process(x, y, y_onehot, args, lengths)
    x = model.layer3(x)
    if mix_layer == 3: x, all_mix_y, all_y, all_group = manifold_mix_process(x, y, y_onehot, args, lengths)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)

    return x, all_group, all_y, all_mix_y

def cut_mix_up(args, input, target):

    rand_index = torch.randperm(len(target))
    lam = np.random.beta(args.mix_alpha, args.mix_alpha)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, lam*target_a + (1-lam)*target_b

def mix_forward(args, x, y, model):

    if args.cut_mix:
        mixed_x, mixed_y = cut_mix_up(args, x, y)
    else:
        mixed_x, mixed_y = mix_up(args, x, y)

    outputs = model(mixed_x)

    return outputs, mixed_y


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

