import time
import torch
from imagegym.config import cfg
from imagegym.utils.scenerios import * 
import matplotlib.pyplot as plt
from imagegym.utils.mask import compute_occlusion_mask, apply_occlusion_mask, create_fixed_mask_missingness
from imagegym.models.tv_inr import prediction_metric_cross_corr_temporal, prediction_metrics_temporal
from imagegym.utils.scaler import z_normalize_out, z_denormalize_out, z_normalize_other
import wandb 
from imagegym.train_fwd import get_fwd, add_custom_stats_from_loss_dict
from imagegym.callbacks import plotPredictions_mean_var, plotPredictions_mean_var_plotly_dims


# Get the color map by name:
cm = plt.get_cmap('jet')

def add_custom_stats_from_loss_dict(loss_dict):
    custom_stats = {}
    for key, value in loss_dict.items():
        if key not in ["loss", "z"]:
            if isinstance(value, torch.Tensor):
                continue
            try:
                custom_stats[key] = value.item()
            except:
                custom_stats[key] = value
    return custom_stats

def get_output(model, x_h, x_f, t_h, t_f, z, perm_h, perm_p, number_of_samples, _mask = None):    
    if model.name == 'tv_inr' or model.name == 'tv_inr_c':
        
        info_x, info_z = model.reconstruct2(
            x_h=x_h, x_f=x_f, t_h=t_h, t_f=t_f, mask=_mask, n_sample= number_of_samples
                )
    
    
    elif model.name == 'timeflow':
        t_h = t_h[:,[0]]
        t_f = t_f[:,[0]]
        perm_h = perm_h.unsqueeze(2)
        perm_p = perm_p.unsqueeze(2)
        _mask = torch.zeros_like(x_h, dtype=torch.bool)
        _mask = _mask.scatter_(3, perm_h, True)
        t_h_masked = t_h[_mask].reshape(t_h.shape[0], t_h.shape[1], t_h.shape[2], -1)
        x_h_masked = x_h[_mask].reshape(x_h.shape[0], x_h.shape[1], x_h.shape[2], -1)
        output = model.reconstruct_t(batch=(x_h_masked, t_f, t_h_masked, t_f, z, perm_h, perm_p), t_h_full = t_h, t_p_full = t_f)
        output = output.unsqueeze(1)
        info_x = {"mean": output, "sample": output[..., None]}

    elif model.name == "deeptime_wrapper":
        x_h = x_h.squeeze(1).permute(0, 2, 1)
        x_f = x_f.squeeze(1).permute(0, 2, 1)
        x_h_time = torch.empty(x_h.shape[0], x_h.shape[1], 0).to(x_h.device)
        x_f_time = torch.empty(x_f.shape[0], x_f.shape[1], 0).to(x_f.device)
        if model.task == 'forecasting':
            loss_dict = {}
            preds = model(x = x_h, x_time = x_h_time, y_time = x_f_time, reconstruct = True)
            preds = preds.unsqueeze(1).permute(0, 1, 3, 2)
            # loss_dict['loss'] = model.compute_loss(preds, x_f)
            info_x = {"mean": preds, "sample": preds[..., None]}
    return info_x

def eval_epoch(logger, loader, model, cur_epoch, split="none", wandb_logger=None, test_mode=False):
    use_grad = model.name == 'timeflow'
    context_manager = torch.enable_grad if use_grad else torch.no_grad
    
    with context_manager():
        model.eval()
        num_batches = len(loader)
        num_batches_max = cfg.dataset.use_number_batch if cfg.dataset.use_number_batch > 0 else num_batches
        print(f"Using {num_batches_max}/{num_batches} batches of the dataset")

        loader.dataset.full_length = True
        for i, batch in enumerate(loader, 1):
            time_start_dl = time.time()
            batch = [b.to(torch.device(cfg.device)) for b in batch]
            time_passed_dl = time.time() - time_start_dl
            time_start_gpu = time.time()

            x_h, x_p, t_h, t_p, z, perm_h, perm_p, c ,tm, target = (item for item in batch)
            bs = x_h.shape[0]
            x_h_norm, x_mean, x_std = z_normalize_out(x_h)
            x_p_norm = z_normalize_other(x_p, x_mean, x_std)
            loss_dict = get_fwd(model, x_h_norm.clone(), x_p_norm.clone(), t_h, t_p, z, perm_h, perm_p, c, tm, missingness=0.0, is_train=False)
            time_start_gpu_end = time.time()
            number_of_samples = batch[0].shape[0]
            history_size = x_h.shape[2:]
            future_size = x_p.shape[2:]
            total_size = [history_size[0], history_size[1] + future_size[1]]
            resolution_0 = total_size
            occlusion_size = (0, total_size[-1])
            nan_present = False
            if torch.any(torch.isnan(x_h)) or torch.any(torch.isnan(x_p)):
                original_size = torch.cat([x_h[(~torch.isnan(x_h))], x_p[(~torch.isnan(x_h))]], dim=-1).shape 
                resolution_0 = tuple(original_size)
                nan_present = True

            task_type = "forecasting"
            sparsity = cfg.dataset.sparsity
            with torch.no_grad():
                if test_mode:
                    custom_stats = {}
                else:
                    #Only log following keys in test mode
                    # loss_dict = {key: value for key, value in loss_dict.items() if key in ["loss","log_prob_x"]}
                    custom_stats = add_custom_stats_from_loss_dict(loss_dict)

            
            all_custom_stats_metric = {}

            for tau in [1]:
                input_size = total_size
                occlusion_size=(0,history_size[-1],history_size[-1]*tau)

                perm_h_obs = perm_h[:,:,:int(occlusion_size[-1])]
                _mask = compute_occlusion_mask(input_size=input_size, task_type=task_type, occlusion_type = sparsity, occlusion_size=occlusion_size) #size [#points] 
                info_x = get_output(model, x_h_norm, x_p_norm, t_h, t_p, z, perm_h_obs, perm_p, number_of_samples, _mask)

                L_list=[96,192,336,720] if cfg.model.type == 'tv_inr' else [future_size[1]]
                for L in L_list:
                    if future_size[-1]>=L:
                        cut_to = history_size[-1]+L
                        with torch.no_grad():
                            if i == 1 and model.task == 'forecasting' and ((cur_epoch + 1) % cfg.plotting.res_epoch == 0 or cur_epoch == 0 or test_mode):
                                number_of_samples = 4 if bs>4 else bs
                                _x_h_norm, _x_p_norm, _t_h, _t_p, _z, _perm_h, _perm_p = (item[:number_of_samples] for item in [x_h_norm, x_p_norm, t_h, t_p, z, perm_h, perm_p])
                                _mask_plots = _mask.repeat(number_of_samples, _x_h_norm.shape[1], 1)
                                plotPredictions_mean_var().compute(x_hat_mu_z=info_x['mean'][:number_of_samples,...,:cut_to], 
                                                            x_hat_L=info_x['sample'][:number_of_samples,...,:cut_to,:], 
                                                            x=torch.concatenate([_x_h_norm, _x_p_norm], dim=-1)[:number_of_samples,...,:cut_to],
                                                            mode="reconstruction", 
                                                            epoch=cur_epoch, T0=None, T=None, wandb_logger=wandb_logger, observed_mask_all=_mask_plots[...,:cut_to], tau=tau, split=split, window_len=L)
                            metrics_dict_recons = prediction_metrics_temporal(theta_x= info_x['mean'][:,...,:cut_to],
                                                                               x = torch.concatenate([x_h_norm, x_p_norm], dim=-1)[:,...,:cut_to], 
                                                                            mask = _mask[...,:cut_to], 
                                                                            temporal=False, 
                                                                            history_size = history_size[-1], window_len=L, experiment_tau=1.0)
                            metrics_dict_recons = {f"{key}_mean": value.mean().cpu().item() for key, value in metrics_dict_recons.items()}
                            custom_stats_metric = add_custom_stats_from_loss_dict(metrics_dict_recons)
                            all_custom_stats_metric = {**all_custom_stats_metric, **custom_stats_metric}

            custom_stats = {**custom_stats, **all_custom_stats_metric}
            with torch.no_grad():
                logger.update_stats(batch_size=batch[0].shape[0],
                                    loss=loss_dict['loss'].item(),
                                    lr=0.0,
                                    time_used=time_start_gpu_end- time_start_gpu,
                                    time_used_for_dl = time_passed_dl,
                                    params=cfg.params,
                                    **custom_stats)

            if i == (num_batches_max): 
                print(f"eval exited at batch number:{i}/{num_batches}")
                break
