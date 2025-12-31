import logging
import time
import torch
from imagegym.checkpoint import load_ckpt, save_ckpt, clean_ckpt, clean_ckpt_list, keep_best_ckpt
from imagegym.config import cfg
from imagegym.utils.epoch import is_eval_epoch
from imagegym.utils.scenerios import * 
import matplotlib.pyplot as plt
import wandb 
from imagegym.eval_imputation import eval_epoch as eval_epoch_imputation
from imagegym.eval_forecast import eval_epoch as eval_epoch_forecast
from imagegym.utils.scaler import z_normalize_out, z_denormalize_out, z_normalize_other
from imagegym.utils.check_dropout import check_dropout_layers
from imagegym.train_fwd import get_fwd, add_custom_stats_from_loss_dict

# Get the color map by name:
cm = plt.get_cmap('jet')

def detach_tensors_in_dict(data_dict):
    detached_dict = {}
    
    for key, value in data_dict.items():
        try:
            # Check if the value is a PyTorch tensor
            if isinstance(value, torch.Tensor):
                # Detach the tensor and store it in the new dictionary
                detached_dict[key] = value.detach()
            else:
                # For non-tensor values, keep them unchanged
                detached_dict[key] = value
        except Exception as e:
            # Handle any unexpected errors
            print(f"Failed to detach value for key '{key}': {e}")
    
    return detached_dict

def train_epoch_acc(logger, loader, model, optimizer, scheduler, epoch, dict_model_tuners, accumulation_steps=None):
    missing_perc = None
    if model.model_type == 'tv_inr':
        _ = dict_model_tuners["scenerios"].check_scenerio(epoch, model)
        if dict_model_tuners["adjust_missing_perc"].check_update(epoch)!=None:
            model.missing_perc = dict_model_tuners["adjust_missing_perc"].check_update(epoch)
            # if dict_model_tuners["adjust_beta_z"].check_update(epoch)!=None:
            # model.beta_z = dict_model_tuners["adjust_beta_z"].check_update(epoch)
        
    
    model.train()
    num_batches = len(loader)

    num_batches_max = cfg.dataset.use_number_batch if cfg.dataset.use_number_batch > 0 else num_batches
    # num_batches_max = num_batches_max * accumulation_steps if accumulation_steps is not None else num_batches_max
    print(f"Using {num_batches_max}/{num_batches} batches of the dataset")    
    
    loader.dataset.full_length = False
    split = loader.dataset.split
    for i, batch in enumerate(loader, 1):
            
        batch = [b.to(torch.device(cfg.device)) for b in batch]
        
        time_start_gpu = time.time()

        x_h, x_f, t_h, t_f, z, perm_h, perm_f, c, tm, target = (item for item in batch)
        bs = x_h.shape[0]
        nan_mask = torch.isnan(x_h)
        # x_h[nan_mask] = 0 #DEBUG
        x_h_norm, x_mean, x_std = z_normalize_out(x_h)
        x_f_norm = None

        num_zeros = (x_std == 0).sum().item()
        # print(f"The tensor has {num_zeros} zero elements.")


        if model.task == 'forecasting':
            x_f_norm = z_normalize_other(x_f, x_mean, x_std)

        missing_perc_eff = cfg.dataset.missing_perc if model.missing_perc is None else model.missing_perc
        
        _ = check_dropout_layers(model)
        loss_dict = get_fwd(model, x_h_norm, x_f_norm, t_h, t_f, z, perm_h, perm_f, c, tm, missingness=missing_perc_eff, split=split)
        time_end_fwd = time.time()
        loss = loss_dict['loss']/accumulation_steps if accumulation_steps is not None else loss_dict['loss']
        # logging.info(f"loss: {loss} at step {i}")
        loss_dict['missing_perc'] = missing_perc_eff

    
        loss.backward(retain_graph=False)        

        # Logging
        with torch.no_grad():
            loss_dict = detach_tensors_in_dict(loss_dict)
            custom_stats = add_custom_stats_from_loss_dict(loss_dict)
            metrics_dict_recons = {'mse_mean': loss.mean().cpu().item()} if model.name == 'timeflow' else {}
            custom_stats_metric = add_custom_stats_from_loss_dict(metrics_dict_recons)
            custom_stats = {**custom_stats, **custom_stats_metric}
            
            logger.update_stats(
                batch_size=bs,
                loss=loss.item(),
                lr=scheduler.get_last_lr()[0] if cfg.optim.scheduler != 'plateau' else optimizer.param_groups[0]['lr'],
                time_used=time.time() - time_start_gpu,
                time_used_for_dl=time_end_fwd - time_start_gpu,
                params=cfg.params,
                **custom_stats
            )

    
        if accumulation_steps is None or i % accumulation_steps == 0 or i == (num_batches_max):
            # for name, param in model.named_parameters():
            #     logging.info(f"{name}: {param.grad}")

            if cfg.train.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
            if model.name == 'timeflow':
                torch.nn.utils.clip_grad_value_(model.inr.parameters(), clip_value=1.0)
            optimizer.step()
            #TODO here update weights of the INR shared.
            if model.name == "tv_inr":
                model.update_shared_weights()
            optimizer.zero_grad(set_to_none=True)

            # Scheduler step
            if cfg.optim.use_scheduler and model.name in ['timeflow', 'deeptime_code']:
                scheduler.step(logger._loss/logger._size_current)
            elif cfg.optim.use_scheduler:
                if cfg.optim.scheduler != 'plateau':
                    scheduler.step()
            
            # checker_list = [True if torch.isnan(param).any() or torch.isinf(param).any() else False for name, param in model.named_parameters()]
            # if any(checker_list):
                # print(f"NaN or Inf detected in the model parameters at step {i}")
                
            # # Compare the model weights to see if they are the same
            # print("\nAre the model parameters the same after both approaches?")
            # for param__model in  (model.parameters()):
            #     logging.info(param__model)



        if i == (num_batches_max): 
            print(f"training exited at batch number:{i}/{num_batches}")
            break


def train(loggers, loaders, model, optimizer, scheduler, scenerios,wandb_logger=None, cur_epoch=0):
    start_epoch = cur_epoch
    logging.info('Start from epoch {}'.format(start_epoch))

    split_names = cfg.dataset.split_names
    accumulation_steps = cfg.train.accumulation_steps
    num_splits = len(loggers)
    val_loss_list = []
    train_loss_list = []
    val_loss_list_epochs = []
    train_loss_list_epochs = []
    i = 0
    #initialize with a high value
    max_val_loss = np.inf
    max_train_loss = np.inf
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch_acc(loggers[0], loaders[0], model, optimizer, scheduler, cur_epoch, scenerios, accumulation_steps=accumulation_steps)
        logger_return_dict =loggers[0].write_epoch(cur_epoch)
        logger_return_dict_wandb = {f"train/{key}": value for key, value in logger_return_dict.items() if key != 'epoch'}
        logger_return_dict_wandb['epoch'] = logger_return_dict.get('epoch', None)
        
        if model.name == 'timeflow':
            train_loss_list.append(logger_return_dict_wandb[f'train/loss'])
            train_loss_list_epochs.append(cur_epoch)
            if train_loss_list[-1] < max_train_loss:
                max_train_loss = train_loss_list[-1]
                keep_best_ckpt(train_loss_list, train_loss_list_epochs, n_keep=1)
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                best_epoch = cur_epoch
                print(f"best epoch: {best_epoch}")

            
        wandb.log(logger_return_dict_wandb) #add train/{} except 'epoch'
        # if True:
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits): 
                if split_names[i] == "val": #we do not put test 
                    logger = loggers[i]
                    _split_name = split_names[i] + "_eval"
                    if cfg.dataset.task == 'imputation':
                        eval_epoch_imputation(logger, loaders[i], model,
                            cur_epoch=cur_epoch, split=_split_name,wandb_logger=wandb_logger, accumulation_steps=accumulation_steps)
                    elif cfg.dataset.task == 'forecasting':
                        eval_epoch_forecast(logger, loaders[i], model,
                            cur_epoch=cur_epoch, split=_split_name,wandb_logger=wandb_logger)
                    logger_return_dict = logger.write_epoch(cur_epoch)
                    logger_return_dict_wandb = {f"{_split_name}/{key}": value for key, value in logger_return_dict.items() if key != 'epoch'}
                    logger_return_dict_wandb['epoch'] = logger_return_dict.get('epoch', None)
                    val_metric = cfg.optim.metric
                    wandb.log(logger_return_dict_wandb)
                    if _split_name == "val_eval":
                        val_loss_list.append(logger_return_dict_wandb[f'{_split_name}/{val_metric}'])
                        val_loss_list_epochs.append(cur_epoch)

            if model.name != 'timeflow':      
                if cfg.optim.use_scheduler:
                    if cfg.optim.scheduler == 'plateau':
                        scheduler.step(val_loss_list[-1])
                        print(optimizer.param_groups[0]['lr'])

                if val_loss_list[-1] < max_val_loss:
                    max_val_loss = val_loss_list[-1]
                    keep_best_ckpt(val_loss_list, val_loss_list_epochs, n_keep=1)
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                    best_epoch = cur_epoch
                    patience = 0
                else:
                    patience += 1
                    if patience >= cfg.optim.early_stop:
                        logging.info(f"Early stopping at epoch {cur_epoch}")
                        keep_best_ckpt(val_loss_list, val_loss_list_epochs, n_keep=1)
                        #end the loop
                        break
    
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    
    # keep_best_ckpt(val_loss_list, val_loss_list_epochs, n_keep=3)
    # save_ckpt(model, optimizer, scheduler, cur_epoch)
    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


def test(loggers, loaders, model, optimizer, scheduler, scenerios, wandb_logger=None, cur_epoch=0):

    split_names = cfg.dataset.split_names
        
    for i in range (0, len(loaders)):
        _split_name_base = split_names[i] + "_testing"
        test_loader = loaders[i]
        test_logger = loggers[i]
        versions = loaders[i].dataset.other_versions
        if cfg.dataset.name in ['HAR']: #split_names[i] in ['train','val', 'test']:
            versions  = [[-1]]#[np.append([-1],versions[0])]
        elif cfg.dataset.name in ['P12','P12_new'] and split_names[i] in ['test']:
            selected_users = [i for i in range(0,50)]
            versions  = [selected_users]
        print(f"all versions")
        _split_name =  _split_name_base
        for version_list in versions:
            for v in version_list:
                if cfg.dataset.name not in ['HAR', 'P12', 'P12_new']:
                    if cfg.dataset.task == 'imputation':
                        test_loader.dataset._select_version(select_split = v)
                    if cfg.dataset.task == 'forecasting':
                        if "train" in _split_name_base or "val" in _split_name_base:
                            test_loader.dataset._select_version(select_user = -1)
                            test_loader.dataset.version = v
                        else:
                            test_loader.dataset._select_version(select_split = v)
                else:
                    test_loader.dataset._select_version(select_user = v)
                
                if cfg.dataset.task == 'imputation':
                        eval_epoch_imputation(test_logger, test_loader, model,
                            cur_epoch=cur_epoch, split=_split_name,wandb_logger=wandb_logger, 
                            test_mode=True if v == -1 else False, save_results = True if v == -1 else False)
                elif cfg.dataset.task == 'forecasting':
                        eval_epoch_forecast(test_logger, test_loader, model,
                            cur_epoch=cur_epoch, split=_split_name,wandb_logger=wandb_logger, test_mode=True if v == -1 else False)

                logger_return_dict = test_logger.write_epoch(cur_epoch)
                
                logger_return_dict_wandb = {
                                           f"{_split_name}/{key}_{v}": value
                                            for key, value in logger_return_dict.items()
                                            if not any(entry in key for entry in ['epoch', 'eta'])
}
                logger_return_dict_wandb['epoch'] = logger_return_dict.get('epoch', None)
                wandb.log(logger_return_dict_wandb)
                test_logger.reset()

    logging.info('Testing done, results saved in {}'.format(cfg.out_dir))

