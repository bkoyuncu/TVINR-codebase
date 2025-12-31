import time
import torch
from imagegym.config import cfg
from imagegym.utils.scenerios import * 
import matplotlib.pyplot as plt
from imagegym.utils.mask import compute_occlusion_mask, apply_occlusion_mask
from imagegym.models.tv_inr import prediction_metrics_temporal
from imagegym.utils.scaler import z_normalize_out, z_denormalize_out
from imagegym.utils.save_output import save_output
from imagegym.utils.check_dropout import check_dropout_layers
from imagegym.callbacks import plotPredictions_mean_var, plotPredictions_mean_var_plotly_dims
from imagegym.train_fwd import get_fwd, add_custom_stats_from_loss_dict,wrapper_masking_saits, wrapper_masking_ours
# Get the color map by name:
cm = plt.get_cmap('jet')

def add_custom_stats_from_loss_dict(loss_dict):
    custom_stats = {}

    for key, value in loss_dict.items():
        if key not in ["loss","z"]:
            try:
                custom_stats[key] = value.item()
            except:
                custom_stats[key] = value
    return custom_stats

def get_output(model, x, t, modulations, perm, res, _mask, c, get_full=False):
    if model.name == 'tv_inr' or model.name == 'tv_inr_c': 

        info_x, info_z = model.reconstruct2(
            x_h=x, x_f=None, t_h=t, t_f=None, mask=_mask, n_sample= 5, label=c, get_full=get_full
                )


        # #Preprocess Data
        # bs = x.shape[0]
        # n_sample = 5
        # if torch.any(torch.isnan(x)):
        #     nan_mask = torch.isnan(x).reshape(x.shape)
        #     nan_mask = nan_mask.reshape(bs,1,-1).permute((0,2,1))
        # else:
        #     nan_mask = torch.zeros_like(x,dtype=torch.bool).reshape(bs,1,-1).permute((0,2,1))

        # coors_point_all_o, x_norm_point_all_o, x_mu_std = model._to_coordinates_and_features(x,t) #o stands for original
        # x_norm_point_all = x_norm_point_all_o[~nan_mask].reshape(bs,-1,1)
        # coors_point_all = coors_point_all_o[np.repeat(~nan_mask.cpu(), 2,axis=2)].reshape(bs,-1,2)
        # out_coordinates_norm = coors_point_all #model.create_outcoordinates(resolution=res, number_of_samples=number_of_samples, coors=coors_point_all)
        # coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)
        # _mask = _mask.repeat(x.shape[0], 1).numpy()
        # #mask to tensor
        # observed_mask = torch.tensor(_mask, dtype=torch.bool)
        # observed_mask = observed_mask.unsqueeze(-1).repeat(1, 1, x.shape[-2])

        # # if len(_mask.shape) == 2:
        # #     _mask = _mask.
        # #coors_point_masked, x_norm_point_masked shaped [bs, #points, 2], [bs, #points, 1]
        # #out_coordinates_norm shaped [bs, #points, 2]
        # #_mask shaped [bs, #points]
        # #x_norm_point_all shaped [bs, #points, 1]
        # info_x_, info_z_= model.reconstruct(coordinate_grid = coors_point_masked, input_x = x_norm_point_masked,
        #                                 resolution=res, 
        #                                 out_coordinates=out_coordinates_norm,
        #                                 mask=_mask,
        #                                 n_sample=5,
        #                                 x_mu_std=x_mu_std,
        #                                 input_original=x_norm_point_all,
        #                                 c = c,
        #                                 x_h=x,
        #                                 t_h=t,
        #                                 nan_mask=nan_mask,
        #                                 observed_mask=observed_mask)
        # info_x = {}
        # #put everything back to original shape
        # # Fill the tensor with NaN values
        # nan_tensor = torch.zeros_like(x_norm_point_all_o).fill_(float('nan'))
        # nan_tensor_shape = nan_tensor.shape
        # nan_tensor[~nan_mask] = info_x_['mean'].flatten()
        # nan_tensor = nan_tensor.reshape(nan_tensor_shape)
        # info_x['mean'] = nan_tensor.permute(0,2,1).reshape(x.shape)
        
        # nan_tensor = torch.zeros((n_sample,) + tuple(x_norm_point_all_o.shape)).fill_(float('nan')).cpu()
        # nan_tensor_shape = nan_tensor.shape
        # nan_tensor[:,~nan_mask] = info_x_['sample'].flatten().reshape(n_sample, -1).cpu()
        # nan_tensor = nan_tensor.reshape(nan_tensor_shape)
        # info_x['sample'] = nan_tensor.permute(0,1,3,2).reshape((n_sample,) + tuple(x.shape[:])).permute(1,2,3,4,0)
        # print("compare")
        
    elif model.name == 'timeflow':
        modulations = modulations
        t = t[:,[0]] #bs ch h T
        perm = perm.unsqueeze(2)
        _mask = torch.zeros_like(x, dtype=torch.bool)
        _mask = _mask.scatter_(3, perm, True) #bs ch h T
        t_masked = t[_mask].reshape(t.shape[0], t.shape[1], t.shape[2], -1) #bs ch h T
        x_masked = x[_mask].reshape(x.shape[0], x.shape[1], t.shape[2], -1) #bs ch h T
        output = model.reconstruct(coordinate_grid=t_masked, input_x = x_masked, modulations=modulations, out_coordinate_grid=t)
        output = output.unsqueeze(1)
        info_x = {"mean": output, "sample": output[...,None]}

    elif model.name =="saits_wrapper":
        stage = 'val'
        bs = x.shape[0]
        x = x.squeeze(1).permute(0, 2, 1)
        _mask = _mask.squeeze(1).permute(0, 2, 1)
        _mask_to_nan = ~(_mask.clone())
        #also reshape this _mask
        x_hat = x.clone()
        x_hat[_mask_to_nan] = torch.tensor(float('nan'))
        missing_mask = (~torch.isnan(x_hat)).float()
        indicating_mask = ((~torch.isnan(x_hat)) ^ (~torch.isnan(x))).float()
        fill_back_mask = (_mask & ~(torch.isnan(x))).bool()

        data_nan_mask = torch.isnan(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=None, neginf=None, out=None)
        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=None, neginf=None, out=None)
        
        input_dict = {
        "X_holdout": x,
        "X": x_hat,
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask}

        if model.task == 'imputation':
            loss_dict = model(inputs = input_dict, stage = stage)
            #change total loss to loss
            loss_dict['loss'] = torch.tensor(0.0,device=x.device)
            loss_dict['reconstruction_loss'] = loss_dict['reconstruction_loss'] * model.reconstruction_loss_weight
            loss_dict['imputation_loss'] = loss_dict['imputation_loss'] * model.imputation_loss_weight
            if model.MIT:
                loss_dict['loss'] += loss_dict['reconstruction_loss']
            if model.ORT:
                loss_dict['loss'] += loss_dict['imputation_loss']
        
            x_imputed = loss_dict['imputed_data_recons']
            x_imputed_obs = x_imputed.clone()
            # x_imputed_obs[data_nan_mask] = torch.tensor(float('nan'))

            #reshape them, add 1 axis 1 and permute last two
            x_imputed = x_imputed.unsqueeze(1).permute(0,1,3,2)
            x_imputed_obs = x_imputed_obs.unsqueeze(1).permute(0,1,3,2)

            info_x = {"mean": x_imputed_obs, "sample": x_imputed_obs[...,None],
                    "mean_full": x_imputed, "sample_full": x_imputed[...,None]}
        
        elif model.task == 'forecasting':
            raise NotImplementedError

    return info_x


# @torch.no_grad()
def eval_epoch(logger, loader, model, cur_epoch, split="none", wandb_logger=None, test_mode=False, accumulation_steps=1, save_results=False):
    use_grad = model.name == 'timeflow'
    context_manager = torch.enable_grad if use_grad else torch.no_grad
    
    with context_manager():
        model.eval()
        num_batches = len(loader)
        num_batches_max = cfg.dataset.use_number_batch if cfg.dataset.use_number_batch > 0 else num_batches
        # num_batches_max = cfg.dataset.use_number_batch if cfg.dataset.use_number_batch > 0 else num_batches
        # num_batches_max = num_batches_max * accumulation_steps if accumulation_steps is not None else num_batches_max
        print(f"Using {num_batches_max}/{num_batches} batches of the dataset")

        results_dict_save = {}
        loader.dataset.full_length = True
        for i, batch in enumerate(loader, 1):
            time_start_dl = time.time()
            batch = [b.to(torch.device(cfg.device)) for b in batch]
            time_passed_dl = time.time() - time_start_dl
            time_start_gpu = time.time()

            x_h, _, t_h, _, z, perm_h, _, c, tm, target= (item for item in batch)
            bs = x_h.shape[0]
            nan_mask = torch.isnan(x_h)
            # x_h[nan_mask] = 0 #DEBUG
            x_h_norm, x_mean, x_std = z_normalize_out(x_h)
            # a = z_denormalize_out(x_h_norm, x_mean, x_std)
            _ = check_dropout_layers(model)

            loss_dict = get_fwd(model, x_h_norm.clone(), _, t_h, _, z, perm_h, _, c, tm, missingness=0.0, is_train=False, split=split)
            time_start_gpu_end = time.time()
            number_of_samples = batch[0].shape[0]
            observed_size = t_h.shape[-1]
            resolution_0 = tuple(batch[0].shape[2:])
            original_size = batch[0].shape[2:]
            nan_present = False
            if torch.any(torch.isnan(x_h)):
                original_size = x_h[(~torch.isnan(x_h))].shape
                resolution_0 = tuple(original_size)
                nan_present = True

            task_type = "imputation"
            sparsity = cfg.dataset.sparsity
            with torch.no_grad():
                if test_mode:
                    custom_stats = {}
                else:
                    #Only log following keys in test mode
                    # loss_dict = {key: value for key, value in loss_dict.items() if key in ["loss","log_prob_x"]}
                    custom_stats = add_custom_stats_from_loss_dict(loss_dict)

            get_full = cfg.dataset.name in ['P12','P12_new']
            
            all_custom_stats_metric = {}
            if sparsity == 'DT':
                tau_levels = [1.0, 0.95, 0.5, 0.3, 0.05] if cfg.dataset.name not in ['P12','P12_new'] else [1.0]
            elif sparsity == 'T':
                tau_levels = [1.0, 0.95, 0.5, 0.3, 0.05]
            elif sparsity == 'D':
                tau_levels = [1.0, 0.68, 0.34] if cfg.dataset.name not in ['P12','P12_new'] else [1.0, 0.5, 0.3, 0.1, -1, -2, -3]
            
            #input size is ch, T
            for tau in tau_levels:
                if sparsity == 'DT':
                    input_size = [np.prod(original_size)]
                    occlusion_size=(0,np.prod(original_size),np.prod(original_size)*tau)
                elif sparsity=="T":
                    input_size = cfg.dataset.dims[1:]
                    occlusion_size=(0,cfg.dataset.dims[-1],cfg.dataset.dims[-1]*tau)
                elif sparsity=="D":
                    input_size = cfg.dataset.dims[1:]
                    occlusion_size=(0,cfg.dataset.dims[-2],cfg.dataset.dims[-2]*tau)
                else:
                    raise NotImplementedError
                    
                perm_obs = perm_h[:,:,:int(occlusion_size[-1])]
            
                # self.sensor_names = ['Urine', 'SysABP', 'DiasABP', 'MAP', 'HR', 'NISysABP', 'NIDiasABP', 'NIMAP'] # Weight left out due to strange values, use as covariate for now
                #SYSABP -> NI...
                #DiasABP -> NI...
                #sys, dias -> MAP / and same for NI


                if tau == -1:
                    _mask = torch.ones_like(x_h, dtype=torch.bool)
                    _mask[:,:,[1,5]] = False #sys, no diastolic -> map #learning the inverse?
                elif tau == -2:
                    _mask = torch.ones_like(x_h, dtype=torch.bool)
                    _mask[:,:,[3,7]] = False #MAP and NIMAP out #learning the formula?
                elif tau == -3:
                    _mask = torch.ones_like(x_h, dtype=torch.bool)
                    _mask[:,:,[1,2,3]] = False #invasives map out #put invasives back
                else:
                    if cfg.dataset.name in ['P12','P12_new']:
                        _mask_dict = wrapper_masking_ours(x_h, 1-tau)
                        _mask = _mask_dict['observed_mask']
                    else:
                        _mask = compute_occlusion_mask(input_size=input_size, task_type=task_type, occlusion_type = sparsity, occlusion_size=occlusion_size) #size [#points] 
                        _mask = _mask.repeat(bs,1,1).reshape(*x_h.shape)
                    #_mask is what we observe
                try:
                    info_x = get_output(model, x_h_norm, t_h, z, perm_obs, resolution_0, _mask, c, get_full)
                    pred_denorm = z_denormalize_out(info_x['mean'].cpu().numpy(), x_mean.cpu().numpy(), x_std.cpu().numpy()) #PUT THIS TO DEVICE
                    if get_full:
                        pred_denorm_full = z_denormalize_out(info_x['mean_full'].cpu().numpy(), x_mean.cpu().numpy(), x_std.cpu().numpy()) #PUT THIS TO DEVICE                                        

                    with torch.no_grad():
                        if i == 1 and model.task == 'imputation' and ((cur_epoch + 1) % cfg.plotting.res_epoch == 0 or cur_epoch == 0 or test_mode):
                            number_of_samples = 4 if bs>4 else bs
                            _x_h_norm, _t_h, _z, _perm_h = (item[:number_of_samples] for item in [x_h_norm, t_h, z, perm_h])    
                            if _mask.shape != x_h_norm.shape:
                                _mask_plots = _mask.repeat(number_of_samples, _x_h_norm.shape[1], 1)
                            else:
                                _mask_plots = _mask[:number_of_samples].reshape(number_of_samples, _x_h_norm.shape[1], -1)
                                
                            plotPredictions_mean_var().compute(x_hat_mu_z=info_x['mean'][:number_of_samples], 
                                                        x_hat_L=info_x['sample'][:number_of_samples],
                                                        x=x_h_norm[:number_of_samples].clone(), 
                                                        mode="reconstruction", 
                                                        epoch=cur_epoch, T0=None, T=None, wandb_logger=wandb_logger, observed_mask_all=_mask_plots, tau=tau, split=split, sparsity=sparsity, window_len=observed_size)
                            
                            if get_full:
                                plotPredictions_mean_var().compute(x_hat_mu_z=info_x['mean_full'][:number_of_samples], 
                                                        x_hat_L=info_x['sample_full'][:number_of_samples],
                                                        x=x_h_norm[:number_of_samples].clone(), 
                                                        mode="reconstruction", 
                                                        epoch=cur_epoch, T0=None, T=None, wandb_logger=wandb_logger, observed_mask_all=_mask_plots, tau=tau, split=split, sparsity=sparsity+"_full", window_len=observed_size)
                        
                        metrics_dict_recons= prediction_metrics_temporal(theta_x= info_x['mean_full'], x = x_h_norm, mask = _mask, temporal =False, experiment_tau = tau)
                        metrics_dict_recons = {f"{key}_mean": value.mean().cpu().item() for key, value in metrics_dict_recons.items()}
                        custom_stats_metric = add_custom_stats_from_loss_dict(metrics_dict_recons)
                        all_custom_stats_metric = {**all_custom_stats_metric, **custom_stats_metric}
                    
                    #save to current wandb path from wandb_logger
                    if save_results:
                        results_output_save = save_output(
                        batch_original = x_h.cpu().numpy(),
                        batch_imputed = pred_denorm,
                        batch_full = None if not get_full else  pred_denorm_full,
                        batch_times = t_h,
                        mask = _mask,
                        targets= target,
                        tau = tau)
                        results_dict_save[f'output_{split}_{cur_epoch}_{i}_{sparsity}_{tau}'] = results_output_save

                except Exception as e:
                    print(f"Error in batch {i} with tau {tau}")
                    print(e)
                    if save_results:
                        results_output_save = None
                        results_dict_save[f'output_{split}_{cur_epoch}_{i}_{sparsity}_{tau}'] = results_output_save
                    continue

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
        
        np.save(f"{cfg.out_dir}/output_{split}_{cur_epoch}.npy", results_dict_save, allow_pickle=True)

