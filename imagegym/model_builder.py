import torch
from imagegym.config import cfg
from imagegym.models.tv_inr import tv_inr
from imagegym.models.time_flow import timeflow
from imagegym.utils.scenerios import Scenerios, AdjustBeta
from imagegym.contrib.network import *
import imagegym.register as register
import copy
from imagegym.models.saits_code import SAITSWrapper
from imagegym.models.deeptime_code import DeepTimeWrapper


network_dict = {
    'tv_inr': tv_inr,
    'timeflow': timeflow,
    'saits_code': SAITSWrapper,
    'deeptime_code': DeepTimeWrapper
}
network_dict = {**register.network_dict, **network_dict}

def create_model(datasets=None,
                 to_device=True,
                 dim_in=None,
                 dim_out=None):

    model_kwargs = create_model_kwargs()
    model = network_dict[cfg.model.type](**model_kwargs)

    if to_device:
        model.to(torch.device(cfg.device))

    try:
        model.set_input_scaler(datasets['train'])
        model.set_input_scaler_temporal(datasets['train'])
    except Exception:
        pass

    if cfg.model.type == 'tv_inr':
        model.create_meta_model()
        print(f"fparams is cuda {list(model.fparams.values())[0].is_cuda}")

    return model

def create_scenerios():

    scenerios = Scenerios(cfg.model.scenerio,cfg.model.scenerio_start,cfg.model.scenerio_end)

    return scenerios

def create_adjust_beta():
    adjust_missing_perc = AdjustBeta([cfg.model.beta_missing_perc_scheduler, cfg.model.start_scheduler, cfg.model.end_scheduler, cfg.dataset.missing_perc_min, cfg.dataset.missing_perc, cfg.model.direction_scheduler])
    # adjust_beta_c = adjust_beta([cfg.model.beta_c_scheduler, cfg.model.start_scheduler, cfg.model.end_scheduler, 1, cfg.model.beta_c])
    # adjust_beta_z = adjust_beta([cfg.model.beta_z_scheduler, cfg.model.start_scheduler, cfg.model.end_scheduler, 1, cfg.model.beta_z])
    return {'adjust_missing_perc': adjust_missing_perc}
    

def create_model_kwargs():
    kwargs ={}
    if cfg.model.type in ['tv_inr', 'tv_inr']:
        #device 
        kwargs['device'] = cfg.device
        #task
        kwargs['task'] = cfg.dataset.task
        #data
        kwargs['dims_x'] = cfg.dataset.dims #[ch, h, w]
        kwargs['coordinate_dim'] = cfg.dataset.coordinate_dim
        kwargs['feature_dim'] = cfg.dataset.dims[0]
        kwargs['dims_c'] = cfg.dataset.dims_c if hasattr(cfg.dataset, 'dims_c') else None
        kwargs['dims_target'] = cfg.dataset.dims_target if hasattr(cfg.dataset, 'dims_target') else None
        kwargs['label_dim'] = cfg.dataset.label_dim if hasattr(cfg.dataset, 'label_dim') else None
        kwargs['cond_type'] = cfg.dataset.cond_type
        #model
        kwargs['distr_x'] = cfg.model.distr_x
        kwargs['name_encoding'] = cfg.model.name_encoding
        kwargs['params_encoding'] = cfg.params_encoding
        kwargs['params_hyper'] = cfg.params_hyper
        kwargs['params_fnrep'] = cfg.params_fnrep
        kwargs['params_pointconvnet'] = cfg.params_pointconvnet
        kwargs['params_transformer'] = cfg.params_transformer
        kwargs['dim_z'] = cfg.model.dim_z
        kwargs['distr_z'] = cfg.model.distr_z
        kwargs['drop_input'] = cfg.model.drop_input
        kwargs['encoder_type'] = cfg.model.encoder_type
        kwargs['params_convnet'] = None
        kwargs['params_cat_prior'] = cfg.params_cat_prior
        kwargs['params_cat_post'] = cfg.params_cat_post
        kwargs['params_cat_x'] = cfg.params_cat_x
        kwargs['params_cat_cond'] = cfg.params_cat_cond
        kwargs['K'] = cfg.params_k_mixture.K
        kwargs['temporal_grid_norm'] = cfg.model.temporal_grid_norm

        #loss function
        kwargs['beta_z'] = cfg.model.beta_z
        kwargs['beta_c'] = cfg.model.beta_c
        kwargs['beta_c_scheduler'] = cfg.model.beta_c_scheduler

        #optimizer
        kwargs['loss_fun'] = cfg.model.loss_fun

        #training
        kwargs['two_step_training'] = cfg.model.two_step_training 
        kwargs['first_step_ratio'] = cfg.model.first_step_ratio
        kwargs['learn_residual_posterior'] =cfg.model.learn_residual_posterior
        kwargs['post_cat_has_z'] = cfg.model.post_cat_has_z
        kwargs['fix_categorical_prior'] = cfg.model.fix_categorical_prior
        kwargs['learn_residual_posterior'] = cfg.model.learn_residual_posterior
        kwargs['distr_x_logscales'] = cfg.model.distr_x_logscales
        kwargs['model_type'] = cfg.model.type
        kwargs['conditional'] = cfg.model.conditional
        kwargs['use_same_label'] = cfg.model.use_same_label

    elif cfg.model.type == 'timeflow':
        #device 
        kwargs['device'] = cfg.device
        #task
        kwargs['task'] = cfg.dataset.task
        #data
        kwargs['dims_x'] = cfg.dataset.dims #[ch, h, w]
        kwargs['coordinate_dim'] = 1 #cfg.dataset.coordinate_dim
        kwargs['feature_dim'] = cfg.dataset.dims[0]

        #model
        kwargs['model_type'] = cfg.model.type
        kwargs['latent_dim'] = cfg.inr.latent_dim
        kwargs['depth'] = cfg.inr.depth
        kwargs['hidden_dim'] = cfg.inr.hidden_dim
        kwargs['num_frequencies'] = cfg.inr.num_frequencies
        kwargs['modulate_scale'] = cfg.inr.modulate_scale
        kwargs['modulate_shift'] = cfg.inr.modulate_shift
        kwargs['frequency_embedding'] = cfg.inr.frequency_embedding
        kwargs['max_frequencies'] = cfg.inr.max_frequencies
        kwargs['min_frequencies'] = cfg.inr.min_frequencies
        kwargs['base_frequency'] = cfg.inr.base_frequency
        kwargs['include_input'] = cfg.inr.include_input
        kwargs['scale'] = cfg.inr.scale
        kwargs['w_passed'] = cfg.inr.w_passed
        kwargs['w_futur'] = cfg.inr.w_futur
        kwargs['passed_ratio'] = cfg.inr.passed_ratio
        kwargs['horizon_ratio'] = cfg.inr.horizon_ratio
        kwargs['log_sampling'] = cfg.inr.log_sampling
        kwargs['lr_code'] = cfg.inr.lr_code
        kwargs['meta_lr_code']  = cfg.inr.meta_lr_code
        kwargs['weight_decay_code']  = cfg.inr.weight_decay_code
        kwargs['log_sampling'] = cfg.inr.log_sampling
    elif cfg.model.type == 'saits':
        # device
        #device 
        kwargs['device'] = cfg.device
        # #task
        # kwargs['task'] = cfg.dataset.task
        # #data
        # kwargs['dims_x'] = cfg.dataset.dims #[ch, h, w]
        # kwargs['coordinate_dim'] = 1 #cfg.dataset.coordinate_dim
        # kwargs['feature_dim'] = cfg.dataset.dims[0]
        kwargs['task'] = cfg.dataset.task
        # kwargs['model_type'] = 'saits'
        kwargs['n_steps'] =  cfg.dataset.dims[-1]
        kwargs['n_features'] =  cfg.dataset.dims[-2]
        kwargs['n_layers'] = 2
        kwargs['d_model'] = 256
        kwargs['n_heads'] = 4
        kwargs['d_k'] = 64
        kwargs['d_v'] = 64
        kwargs['d_ffn'] = 128
        kwargs['dropout'] = 0.1
        kwargs['epochs'] = cfg.optim.max_epoch

    elif cfg.model.type == 'saits_code':
        kwargs['task'] = cfg.dataset.task
        kwargs['missing_perc'] = cfg.dataset.missing_perc
        kwargs['device'] = cfg.device

        kwargs['n_steps'] = cfg.dataset.dims[-1]
        kwargs['n_features'] = cfg.dataset.dims[-2]

        kwargs['input_with_mask'] = cfg.saits.input_with_mask
        kwargs['model_type'] = 'saits_wrapper'
        kwargs['n_groups'] = cfg.saits.n_groups
        kwargs['n_group_inner_layers'] = cfg.saits.n_group_inner_layers
        kwargs['param_sharing_strategy'] = cfg.saits.param_sharing_strategy
        kwargs['d_model'] = cfg.saits.d_model
        kwargs['d_inner'] = cfg.saits.d_inner
        kwargs['n_head'] = cfg.saits.n_head
        kwargs['d_k'] = cfg.saits.d_k
        kwargs['d_v'] = cfg.saits.d_v
        kwargs['dropout'] = cfg.saits.dropout
        kwargs['diagonal_attention_mask'] = cfg.saits.diagonal_attention_mask
        kwargs['epochs'] =cfg.optim.max_epoch
        kwargs['MIT'] = cfg.saits.MIT
        kwargs['ORT'] = cfg.saits.ORT
        kwargs['d_time'] = cfg.dataset.dims[-1]
        kwargs['d_feature'] = cfg.dataset.dims[-2]
        kwargs['reconstruction_loss_weight'] = cfg.saits.reconstruction_loss_weight
        kwargs['imputation_loss_weight'] = cfg.saits.imputation_loss_weight

    elif cfg.model.type == 'deeptime_code':
        kwargs['task'] = cfg.dataset.task
        kwargs['missing_perc'] = cfg.dataset.missing_perc
        kwargs['device'] = cfg.device

        kwargs['n_steps'] = cfg.dataset.dims[-1]
        kwargs['n_features'] = cfg.dataset.dims[-2]

        kwargs['target'] = 'OT'
        kwargs['scale'] = True
        kwargs['features'] = 'M'

        kwargs['dim_size'] = cfg.dataset.dims[-2]
        kwargs['datetime_feats'] = 0
        # Model architecture
        kwargs['model_type'] = cfg.deeptime.model_type
        kwargs['layer_size'] = cfg.deeptime.layer_size
        kwargs['inr_layers'] = cfg.deeptime.inr_layers
        kwargs['n_fourier_feats'] = cfg.deeptime.n_fourier_feats
        kwargs['scales'] = cfg.deeptime.scales
        kwargs['d_model'] = cfg.deeptime.d_model
        kwargs['d_inner'] = cfg.deeptime.d_inner

        # kwargs['epochs'] = 50
        # kwargs['lr'] = 1e-3
        # kwargs['weight_decay'] = 0.0
        # kwargs['warmup_epochs'] = 5
        # kwargs['batch_size'] = 256
        kwargs['loss_name'] = 'mse'
        kwargs['clip'] = 10.0
        kwargs['patience'] = cfg.optim.patience

    
    return copy.deepcopy(kwargs)
