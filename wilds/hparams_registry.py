# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hashlib
import numpy as np

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def _hparams(random_seed):
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('warmup_step', 4000, lambda r: 4000)
    _hparam('threshold', 0.9, lambda r: 0.9)
    _hparam('threshold_lower_bound', 0.5, lambda r: 0.5)
    _hparam('probability_to_discard', 0.2, lambda r: r.choice([0.2, 0.3, 0.4]))
    _hparam('value_to_change', 0.1, lambda r: 10**r.uniform(-1, -2))
    _hparam('step_to_change', 500, lambda r: int(50*r.uniform(1, 6)))
    _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
    _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    _hparam('rho', 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))
    _hparam('alpha', 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))
    return hparams

def default_hparams():
    return {a: b for a, (b, c) in _hparams(0).items()}


def random_hparams(seed):
    return {a: c for a, (b, c) in _hparams(seed).items()}
