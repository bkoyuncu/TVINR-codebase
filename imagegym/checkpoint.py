from imagegym.config import cfg
import wandb
import torch
import os
import logging
import numpy
import contextlib

def get_ckpt_dir(path_to_check=None):
    # print(cfg.out_dir if path_to_check is None else path_to_check)
    return '{}/ckpt/'.format(cfg.out_dir)


def get_all_epoch(path_to_check=None):
    d = get_ckpt_dir(path_to_check)
    names = os.listdir(d) if os.path.exists(d) else []
    if len(names) == 0:
        return [0]
    epochs = [int(name.split('.')[0]) for name in names]
    return epochs


def get_last_epoch():
    return max(get_all_epoch())


def load_ckpt(model, optimizer=None, scheduler=None, ckpt_path = None):
    # if cfg.train.epoch_resume < 0:
    #     epoch_resume = get_last_epoch()
    # else:
    #     epoch_resume = cfg.train.epoch_resume

    # if ckpt_number is not None:
    #     epoch_resume = ckpt_number
        
    # ckpt_name = '{}{}.ckpt'.format(get_ckpt_dir(), epoch_resume)
    # if not os.path.isfile(ckpt_name):
    #     return 0

    # print(torch.cuda.is_available(),torch.cuda.device_count())

    print("ckpt name:",ckpt_path)
    if cfg.device == "cpu":
        ckpt = torch.load(ckpt_path, map_location=torch.device(cfg.device))
    else:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state'],strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    return epoch + 1


def save_ckpt(model, optimizer, scheduler, epoch):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    os.makedirs(get_ckpt_dir(), exist_ok=True)
    ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
    torch.save(ckpt, ckpt_name)
    logging.info('Check point saved: {}'.format(ckpt_name))
    # wandb.save(ckpt_name)


def clean_ckpt():
    epochs = get_all_epoch()
    epoch_last = max(epochs)
    for epoch in epochs:
        if epoch != epoch_last:
            ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
            os.remove(ckpt_name)
            
def clean_ckpt_list(list_epochs):
    for epoch in list_epochs:
        ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
        with contextlib.suppress(FileNotFoundError):
            try:
                os.remove(ckpt_name)
            except:
                continue

def clean_ckpt_except_last():
    epochs = get_all_epoch()
    epoch_last = max(epochs)
    for epoch in epochs:
        if epoch != epoch_last:
            ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
            os.remove(ckpt_name)


def keep_best_ckpt(l_losts, l_epochs, n_keep=2):
    indices = numpy.argsort(l_losts)
    l_epochs = [l_epochs[i] for i in indices[n_keep:]]
    clean_ckpt_list(l_epochs)

    # epochs = get_all_epoch()
    # for epoch in epochs:
    #     if epoch not in l_epochs:
    #         ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
    #         os.remove(ckpt_name)

def load_inference_checkpoint(args, cfg, model, dump_cfg, load_latest=False):
    """
    Load the inference checkpoint.

    Parameters:
    args: Arguments containing inference and list_ckpts information.
    cfg: Configuration object with out_dir and inference details.
    model: Model to load the checkpoint into.
    get_all_epoch: Function to get all available epochs from the checkpoint directory.
    dump_cfg: Function to dump the updated configuration.
    """

    out_dir_parent = cfg.out_dir
    run_name = cfg.inference.wandb_run_id
    wandb_run_dir = os.path.dirname(cfg.inference.wandb_run_dir)
    ckpt_dir = os.path.join(out_dir_parent, "ckpt")

    a = get_all_epoch(ckpt_dir)
    a.sort()

    if load_latest:
        ckpt_name = a[-1]
    elif len(args.list_ckpts) > 0:
        ckpt_name = args.list_ckpts[0]
        if ckpt_name not in a:
            print("The checkpoint is not available, please select from the following list:")
            print(a)
            return
    elif len(a)>1:
        ckpt_name = a[-2]
    elif len(args.list_ckpts) == 0:
        ckpt_name = a[-1]

    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.ckpt')
    print("ckpt_path:", ckpt_path)
    cfg.inference['wandb_ckpt_name'] = ckpt_path

    dump_cfg(cfg)

    from imagegym.checkpoint import load_ckpt
    cur_epoch = load_ckpt(model, optimizer=None, scheduler=None, ckpt_path=ckpt_path)
    
    return cur_epoch