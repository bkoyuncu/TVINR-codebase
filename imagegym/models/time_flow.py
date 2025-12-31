#some snippets used from https://github.com/EmilienDupont/neural-function-distributions
import torch
import torch.nn as nn
import math
from imagegym.config import cfg
from imagegym.loss import ELBO_mixture_observed
from functorch import vmap

import torch
import torch.nn as nn
from functorch import vmap
from .tv_inr_base import tv_inr_base

from imagegym.utils.mask import *
from imagegym.utils.priors import *
from TimeFlow.src.network import ModulatedFourierFeatures
from TimeFlow.src.metalearning.metalearning_imputation import outer_step as outer_step_impute
from TimeFlow.src.metalearning.metalearning_forecasting import outer_step as outer_step_forecast
from imagegym.utils.scaler import z_normalize, z_denormalize_out, z_normalize_out, maxmin_normalize_out, maxmin_denormalize_out



class timeflow(nn.Module):
    def __init__(self, 
                 feature_dim,
                 coordinate_dim,
                 device,
                 weight_decay_code=0,
                 lr_code=0.01,
                 inner_steps=3,
                 name="timeflow",
                 model_type="fourier_features",
                 latent_dim=128,
                 depth=5,
                 hidden_dim=256,
                 num_frequencies=64,
                 modulate_scale=False,
                 modulate_shift=True,
                 frequency_embedding="nerf",
                 max_frequencies=10,
                 min_frequencies=0.0,
                 base_frequency=2,
                 include_input=True,
                 scale=5,
                 w_passed=0.5,
                 w_futur=0.5,
                 passed_ratio=0.3,
                 horizon_ratio=0.3,
                 log_sampling=True,
                 task = None,
                 **kwargs) -> None:
        super(timeflow, self).__init__()
        self.input_dim = 1
        self.output_dim = feature_dim
        self.inner_steps = inner_steps

        self.name = 'timeflow'
        self.task = task #'forecast' #impute or forecast
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.frequency_embedding = frequency_embedding
        self.max_frequencies = max_frequencies
        self.min_frequencies = min_frequencies
        self.base_frequency = base_frequency
        self.include_input = include_input
        self.scale = scale
        self.w_passed = w_passed
        self.w_futur = w_futur
        self.passed_ratio = passed_ratio
        self.horizon_ratio = horizon_ratio
        self.log_sampling = log_sampling
        self.lr_code = lr_code
        self.weight_decay_code = weight_decay_code
        self.missing_perc = None #not used only for compatibility.

        
        self.inr = ModulatedFourierFeatures(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_frequencies=self.num_frequencies,
            latent_dim=self.latent_dim,
            width=self.hidden_dim,
            depth=self.depth,
            modulate_scale=self.modulate_scale,
            modulate_shift=self.modulate_shift,
            frequency_embedding=self.frequency_embedding,
            include_input=self.include_input,
            scale=self.scale,
            max_frequencies=self.max_frequencies,
            base_frequency=self.base_frequency,
            log_sampling=self.log_sampling
        )
        #for us this is in INR parameters to be learned but in the baseline this is defined outside so they do not care if they have gradients to this or not
        self.alpha = nn.Parameter(torch.Tensor([self.lr_code]).to(device), requires_grad= False) 
        weight_decay_lr_code = weight_decay_code
    
    
    @staticmethod
    def kwargs(cfg, preparator):
        model = cfg.model
        layer = cfg.layer

        return {
            'dim_in': preparator.dim_coordinates(),
            'dim_out': preparator.dim_features(),
            'data_converter': preparator.data_converter(),
            'dim_latent': model.dim_latent,
            'num_layers': model.num_layers,
            'dim_hidden': model.dim_inner,
            'w0_initial': layer.w0_initial
        }
    
    def set_input_scaler(self, dataset):
        '''
        dataset: dataset object preferably train
        mode: 'none' or 'global' or 'channel' 
        '''
        from imagegym.utils.scaler import StandardScaler
        self.input_scaler = StandardScaler(mode=cfg.dataset.spatial_norm)
        return

    def set_input_scaler_temporal(self, dataset):
        from imagegym.utils.scaler import Temporal_Scaler
        self.input_scaler_temporal = Temporal_Scaler().get_scaler(dataset)[1] #only uses x axis (linear time)
        return
    
    def preprocess_batch(self,x, **kwargs):
        """
        :param batch: a dict of tensors
        :param missingness: give the missingness of the batch (somehow)
        :return: elbo
        """
        assert self.input_scaler is not None
        x_mean, x_std = None, None

        if cfg.dataset.spatial_norm == "none_z":
            x, x_mean, x_std = z_normalize_out(x)
            x_norm = self.input_scaler.transform(x)
            h=x_norm
        elif cfg.dataset.spatial_norm == "none_01":
            x, x_mean, x_std = maxmin_normalize_out(x)
            x_norm = self.input_scaler.transform(x)
            h=x_norm

        else:
            x_norm = x
            if cfg.dataset.use_bn_initial:
                h = self.bn_initial(x_norm)
            else:
                h = x_norm
        return x_norm, h, [x_mean, x_std]
    
    def postprocess_batch(self, x, x_true = None):
        if cfg.dataset.spatial_norm == "none_z":
            if x_true is not None:
                x_mean, x_std = x_true[0], x_true[1]
            else:
                raise ValueError("x_true should be given for denormalization")
            x = z_denormalize_out(x, x_mean, x_std)
            return x
        elif cfg.dataset.spatial_norm == "none_01":
            if x_true is not None:
                x_min, x_max = x_true[0], x_true[1]
            else:
                raise ValueError("x_true should be given for denormalization")
            x = maxmin_denormalize_out(x, x_min, x_max)
            return x
        else:
            return self.input_scaler.inverse_transform(x)

    def preprocess_batch_temporal(self, t):
        """
        :param batch: a dict of tensors
        :param missingness: give the missingness of the batch (somehow)
        :return: elbo
        """
        assert self.input_scaler_temporal is not None
        t_norm = self.input_scaler_temporal.transform(t)
        return t_norm
    
    def postprocess_batch_temporal(self, t):
        return self.input_scaler_temporal.inverse_transform(t)
    
    def forward(self, batch, missingness=0.0, is_train=True, gradient_checkpointing=False, **kwargs):

        if self.task == 'imputation':
            series, coords, modulations, idx = batch
            #use missingness to slice the series on time 
            series = series[:,:,: int(series.shape[2] * (1 - missingness))]
            coords = coords[:,:,: int(coords.shape[2] * (1 - missingness))]

            series, coords = series.transpose(1,2), coords.transpose(1,2)
            coords = self.preprocess_batch_temporal(coords)[:,0,0].unsqueeze(-1)
            series  = series[:,0,0].unsqueeze(-1)
            loss_dict = outer_step_impute(
                self.inr,
                coords,
                series,
                self.inner_steps,
                self.alpha,
                is_train=is_train,
                gradient_checkpointing=gradient_checkpointing,
                loss_type="mse",
                modulations=torch.zeros_like(modulations),
            )
        elif self.task == 'forecasting':
            series_h, series_f, coords_h, coords_f, modulations, idx_h, idx_f = batch

            series_h = series_h[:,:,: int(series_h.shape[2] * (1 - missingness))]
            series_f = series_f[:,:,: int(series_f.shape[2] * (1 - missingness))]
            coords_h = coords_h[:,:,: int(coords_h.shape[2] * (1 - missingness))]
            coords_f = coords_f[:,:,: int(coords_f.shape[2] * (1 - missingness))]

            series_h, coords_h, series_f, coords_f = series_h.transpose(1,2), coords_h.transpose(1,2), series_f.transpose(1,2), coords_f.transpose(1,2)
            coords_h = self.preprocess_batch_temporal(coords_h)[:,0,0].unsqueeze(-1)
            coords_f = self.preprocess_batch_temporal(coords_f)[:,0,0].unsqueeze(-1)
            series_h = series_h[:,0,0].unsqueeze(-1)
            series_f = series_f[:,0,0].unsqueeze(-1)

            loss_dict = outer_step_forecast(
                self.inr,
                coords_h,
                coords_f,
                series_h,
                series_f,
                self.inner_steps,
                self.alpha,
                cfg.model.tmh, #lookback
                cfg.model.tmp, #horizon,
                w_passed=0.5,
                w_futur=0.5,
                is_train=is_train,
                gradient_checkpointing=gradient_checkpointing,
                loss_type="mse",
                modulations=torch.zeros_like(modulations),
            )



        return loss_dict
    
    def reconstruct(self, coordinate_grid, input_x, modulations, out_coordinate_grid=None):

        if self.task == 'imputation':
            if out_coordinate_grid is None:
                #copy the tensor coordinate grid
                out_coordinate_grid = coordinate_grid.clone()
            output  = self.forward(batch=(input_x, coordinate_grid, modulations, None), is_train=False, gradient_checkpointing=False)
            modulations =  output['modulations'].detach()
            out_coordinate_grid = self.preprocess_batch_temporal(out_coordinate_grid)
            out_coordinate_grid = out_coordinate_grid.transpose(1,2)[:,0,0].unsqueeze(-1)
            output = self.inr.modulated_forward(out_coordinate_grid, modulations)
        
        return output.transpose(1,2).detach()
    
    def reconstruct_t(self, batch = None, t_h_full = None, t_p_full = None):
        assert self.task == 'forecasting'
        # x_h, x_p, t_h, t_p, z, perm_h, perm_p = (i for i in batch)
        output  = self.forward(batch=batch, is_train=False, gradient_checkpointing=False)
        modulations =  output['modulations'].detach()
        t_h_full = self.preprocess_batch_temporal(t_h_full)
        t_h_full = t_h_full.transpose(1,2)[:,0,0].unsqueeze(-1)
        fit = self.inr.modulated_forward(t_h_full, modulations)
        t_p_full = self.preprocess_batch_temporal(t_p_full)
        t_p_full = t_p_full.transpose(1,2)[:,0,0].unsqueeze(-1)
        forecast_train = self.inr.modulated_forward(t_p_full, modulations)
        output = torch.concatenate([fit.transpose(1,2).detach(), forecast_train.transpose(1,2).detach()], dim=-1)
        return output

    
    def print_params_count(self, logging):
        num_params = sum(p.numel() for p in self.parameters())
        logging.info('Num parameters: {}'.format(num_params))
        return num_params
