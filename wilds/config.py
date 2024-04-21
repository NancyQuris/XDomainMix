
dataset_defaults = {
    'camelyon': {
        'epochs': 2,
        'batch_size': 32,
        'optimiser': 'SGD',
        'optimiser_args': {
            'momentum': 0.9,
            'lr': 1e-4,
            'weight_decay': 0,
        },
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'print_iters': 2000,
    },
    'fmow': {
        'epochs': 5,
        'batch_size': 32,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': 'acc_worst_region',
        'reload_inner_optim': True,
        'print_iters': 2000
    },
}

