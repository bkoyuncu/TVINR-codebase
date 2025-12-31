import random
import numpy as np
import torch
import logging
from imagegym.cmd_args import parse_args
from imagegym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir)

from imagegym.logger import setup_printing, create_logger
from imagegym.optimizer import create_optimizer, create_scheduler
from imagegym.model_builder import create_model, create_scenerios, create_adjust_beta
from imagegym.train import train, test
from imagegym.utils.device import auto_select_device
from imagegym.contrib.train import *

from imagegym.loader import create_dataset, create_loader

import warnings
warnings.filterwarnings('ignore')
from pytorch_lightning.loggers import WandbLogger
import os
from imagegym.checkpoint import get_all_epoch, load_ckpt, load_inference_checkpoint
os.environ["WANDB__SERVICE_WAIT"] = "300"
import wandb
import os


if __name__ == '__main__':
    print("cuda is available:",torch.cuda.is_available())
    print("cuda device count :",torch.cuda.device_count())
    # Load cmd line args
    args = parse_args()
    # Repeat for different random seeds
    for i in range(args.repeat): 
        # Load config file
        # os.chdir('./run')
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_file(args.cfg_wandb)
        cfg.merge_from_list(args.opts)
        # assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        if args.inference ==1:
            cfg.train.mode = None
            cfg.inference.mode = 'standard'
        if cfg.train.resume_training:
            cfg.train.mode = 'standard'
        elif not cfg.train.resume_training:    
            if cfg.wandb.log_dir is not None:
                #join wandb log dir with out_dir
                out_dir_parent = os.path.join(out_dir_parent, cfg.wandb.log_dir)
                # create wandb .config directory to avoid errors caused by getpass.getuser() --> pwd.getpwuid(os.getuid())[0] inside a docker execution
            if cfg.wandb.create_cfgdir:
                wandb_config_dir = os.path.join(os.path.expanduser("~"), ".config", "wandb")
                if not os.path.exists(wandb_config_dir):
                    os.makedirs(wandb_config_dir)
                    print("WandB .config directory created.")
            if not os.path.exists('{}/logs/'.format(out_dir_parent)):
                os.makedirs('{}/logs/'.format(out_dir_parent))
        


        # ============= Activate CUDA ============= #
        arg_cuda = int(cfg.gpu>0) and torch.cuda.is_available()
        device = torch.device("cuda" if arg_cuda else "cpu")
        cfg.device =str(device)
        if str(device) == "cuda":
            print('cuda activated')
        else:
            print('cuda not activated')

        # ============= Set seeds ============= #
        cfg.seed = cfg.seed + i + args.seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        setup_printing()
        auto_select_device()
        print("Device:", cfg.device)
    


        # ============= Set Variables for WandB Logger ============= #
        # run_name is optional, if not given, wandb will generate a random name
        run_name = None
        if 'run_name'in cfg.wandb:
            run_name = cfg.wandb['run_name'] 

        # ckpt_name is optional, if not given, the latest epoch will be used
        ckpt_name = None

        
        # used to identify the project in wandb
        entity = None
        if 'entity' in cfg.wandb:
            entity = cfg.wandb.entity

        # version of the model in wandb, if not given, the latest version will be used
        version = 'latest'

        if 'mode' in cfg.wandb:
            mode = cfg.wandb['mode']
        else:
            mode = 'offline'

        project = cfg.wandb['project_name']

        if cfg.model.type == 'timeflow':
            project = 'timeflow'
    
        if cfg.train['mode'] == 'standard':
            resume_training = cfg.train['resume_training']
            id = None
            dir = None

            if resume_training:
                id = cfg.inference['wandb_run_id']
                dir = cfg.inference['wandb_run_dir']
            
            wandb_logger = WandbLogger(
                id=id,
                dir=dir,
                name=run_name, 
                project=project, 
                entity=entity, 
                job_type='train_test' if cfg.inference['mode'] == 'standard' else 'train',
                save_dir='{}/logs/'.format(out_dir_parent),
                log_model=False, # log models (from checkpoints) to wandb
                checkpoint_name=ckpt_name,
                mode = mode
            )
            print(cfg)
            wandb_logger.log_hyperparams(cfg)
            cfg.inference.wandb_run_id = wandb.run.id
            cfg.inference.wandb_run_dir = wandb.run.dir
            ckpt_path = None
            resume_training = cfg.train['resume_training']
            if not resume_training:
                update_out_dir(out_dir_parent, args, wandb_path=wandb.run.dir) #add wandb name to local name

        
        elif cfg.inference['mode'] == 'standard':
            
            if cfg.train.batch_size != 1:
                id_value = cfg.inference['wandb_run_id']
                dir_value = cfg.inference['wandb_run_dir']
                resume_value= 'must'
            else:
                id_value = None
                dir_value = None
                resume_value = 'allow'

            # cfg.train.batch_size = 1

            wandb_logger = WandbLogger(
                id = id_value,
                dir = dir_value,
                name=run_name, 
                project=project, 
                entity=entity, 
                job_type='test',
                save_dir='{}/logs/'.format(out_dir_parent),
                log_model=False, # log models (from checkpoints) to wandb
                checkpoint_name = ckpt_name,
                mode = mode,
                resume=resume_value
            )
            wandb_logger.log_hyperparams(cfg)

        if str(device) =="cuda":
            accelerator="gpu"
        else:
            accelerator="cpu"
        

        # Set learning environment

        datasets = create_dataset()
        loaders = create_loader(datasets)
        model = create_model(datasets)
        meters = create_logger(datasets)
        scenerios = create_scenerios()
        adjust_beta_dict = create_adjust_beta()
        if model.name in ["timeflow", "saits_wrapper", "deeptime_wrapper"]:
            optimizer = create_optimizer(model.parameters(), model)
        elif model.name == 'tv_inr':
            optimizer = create_optimizer(list(model.parameters()) + list(model.fparams_shared.values()) if hasattr(model, 'fparams_shared') else model.parameters())
        elif model.name == 'deeptime_wrapper':
            optimizer = create_optimizer(model.parameters())
        if model.name in ["timeflow", 'tv_inr',"saits_wrapper", 'deeptime_wrapper']:
            scheduler = create_scheduler(optimizer)

        dict_model_tuners = {
            "scenerios": scenerios,
        }

        dict_model_tuners.update(adjust_beta_dict)

        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = model.print_params_count(logging)
        logging.info('Num parameters: {}'.format(cfg.params))
        dump_cfg(cfg)
        
        if cfg.train.mode == 'standard':
            if cfg.train.resume_training:
                cur_epoch = load_inference_checkpoint(args, cfg, model, dump_cfg)
            else:
                cur_epoch = 0
            train(meters, loaders, model, optimizer, scheduler, dict_model_tuners, wandb_logger, cur_epoch)
            print("Finished training.")



        if cfg.inference.mode == 'standard':
            if cfg.model.type == 'timeflow':
                load_latest = True
            else:
                load_latest = False

            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            cur_epoch = load_inference_checkpoint(args, cfg, model, dump_cfg, load_latest=load_latest)
            test(meters, loaders, model, optimizer, scheduler, dict_model_tuners, wandb_logger, cur_epoch)
            print("Finished testing.")
        
        # wandb.finish()
        print("Finished run.")
        