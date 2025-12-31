from imagegym.config import cfg

import torch.optim as optim

from imagegym.contrib.optimizer import *
import imagegym.register as register


def create_optimizer(params, model=None):

    
    if cfg.model.type == 'deeptime_code':
        lambda_lr = 1
        group1 = []  # lambda
        group2 = []  # no decay
        group3 = []  # decay
        no_decay_list = ('bias', 'norm',)
        for param_name, param in model.named_parameters():
            if '_lambda' in param_name:
                group1.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2.append(param)
            else:
                group3.append(param)
        optimizer = optim.Adam([
            {'params': group1, 'weight_decay': 0, 'lr': lambda_lr, 'scheduler': 'cosine_annealing'},
            {'params': group2, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
            {'params': group3, 'scheduler': 'cosine_annealing_with_linear_warmup'}
        ], lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)

    else:
        params = filter(lambda p: p.requires_grad, params)
        if cfg.optim.optimizer == 'adamw':
            optimizer = optim.AdamW(
            [
                {"params": params, "lr": cfg.optim.base_lr, "weight_decay": cfg.optim.weight_decay},
            ],
            lr=cfg.optim.base_lr,
            weight_decay=0,
            )

        elif cfg.optim.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=cfg.optim.base_lr,
                                weight_decay=cfg.optim.weight_decay)
        elif cfg.optim.optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=cfg.optim.base_lr,
                                momentum=cfg.optim.momentum,
                                weight_decay=cfg.optim.weight_decay)
        else:
            raise ValueError('Optimizer {} not supported'.format(
                cfg.optim.optimizer))

    return optimizer

import math

def create_scheduler(optimizer):
    # Try to load customized scheduler
    if cfg.model.type == 'deeptime_code':
        eta_min = 0.0
        T_max = cfg.optim.max_epoch
        warmup_epochs = cfg.deeptime.warmup_epochs
        scheduler_fns = []
        for param_group in optimizer.param_groups:
            scheduler = param_group['scheduler']
            if scheduler == 'none':
                fn = lambda T_cur: 1
            elif scheduler == 'cosine_annealing':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                            1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
            elif scheduler == 'cosine_annealing_with_linear_warmup':
                lr = eta_max = param_group['lr']
                # https://blog.csdn.net/qq_36560894/article/details/114004799
                fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                            eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
            else:
                raise ValueError(f'No such scheduler, {scheduler}')
            scheduler_fns.append(fn)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fns)
    else:
        for func in register.scheduler_dict.values():
            scheduler = func(optimizer)
            if scheduler is not None:
                return scheduler
        if cfg.optim.scheduler == 'none':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.optim.max_epoch + 1)
        elif cfg.optim.scheduler == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=cfg.optim.steps,
                                                    gamma=cfg.optim.lr_decay)
        elif cfg.optim.scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=cfg.optim.max_epoch)
        elif cfg.optim.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=cfg.optim.gamma )
        elif cfg.optim.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            patience=cfg.optim.patience)
        else:
            raise ValueError('Scheduler {} not supported'.format(
                cfg.optim.scheduler))
    return scheduler
