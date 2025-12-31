#some snippets used from https://github.com/EmilienDupont/neural-function-distributions, https://github.com/bkoyuncu/vamoh
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

class tv_inr(tv_inr_base):
    def __init__(self, **kwargs) -> None:
        super(tv_inr,self).__init__(**kwargs)

        self.prior_z = self.set_z_prior_distr(kwargs['distr_z'])
        self.name = kwargs['model_type']
        self.model_type = kwargs['model_type']
        assert cfg.model.direction_scheduler in ['min_to_max','max_to_min'], 'Direction scheduler should be min_to_max or max_to_min'
        self.missing_perc = cfg.dataset.missing_perc if cfg.model.direction_scheduler == 'max_to_min' else cfg.dataset.missing_perc_min

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

    
    def forward(self, batch, missingness: float = 0.0, reconstruct=False, **kwargs):
        '''
        batch = [x, t, c]
        Shape of x: [bs, 1, dim_x, T]
        Shape of t: [bs, 2, dim_x, T]
        Shape of c: [bs, 1]
        '''

        elbo_mode = cfg.model.elbo_mode #'observed' # or full
        
        if self.task == 'imputation':
            #Get Data
            x, ts, label, tm = batch #x, ts = batch[:2] #ts are coordinates
            bs = x.shape[0]

            #Mask for nan entries (missing from data itself)
            if torch.any(torch.isnan(x)):
                nan_mask = torch.isnan(x).reshape(x.shape)
                nan_mask = nan_mask.reshape(bs,1,-1).permute((0,2,1))
            else:
                nan_mask = torch.zeros_like(x,dtype=torch.bool).reshape(bs,1,-1).permute((0,2,1))

            coors_point_all, x_norm_point_all, _ = self._to_coordinates_and_features(x,ts)
            encoded_coors_point_all_transformer = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)

            #Mask Point Data
            # x_norm_point_all = x_norm_point_all[~nan_mask].reshape(bs,-1,1) #WE ARE NOT DROPPING NAN POINTS AS WE DO NOT USE PVAE
            coors_point_all = coors_point_all#[np.repeat(~nan_mask.cpu(), 2,axis=2)].reshape(bs,-1,2)
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)
            
            #Get scale coeff for total/non-nan pixels
            nonnan_scale = torch.sum(~nan_mask[0]).float() / nan_mask[0].flatten().shape[0]

            #Create missigness mask 
            _, observed_mask, observed_mask_point = create_mask_missingness(x_norm_point_all.permute(0,2,1),missingness=missingness) #TODO add missingness
            observed_mask_full = torch.zeros_like(x,dtype=torch.bool).reshape(bs,1,-1).permute((0,2,1)).to(x.device)
            observed_mask = observed_mask.permute((0,2,1)).to(x.device)
            observed_mask_full[nan_mask==False] = torch.tensor(observed_mask[nan_mask==False], dtype=torch.bool).flatten().to(x.device) #this is for the ELBO calculation
            observed_mask_full = observed_mask_full.reshape(bs,-1,1)
            
            #Mask Data
            # x_norm_point = self.mask_to_input(input=x_norm_point_all,mask=observed_mask_point) #[bs,#points,ch]
            # encoded_coors_point = self.mask_to_input(input=self.encoded_coors_point_all,mask=observed_mask_point) #[bs,#points,ch]
            # coors_point =  self.mask_to_input(input=coors_point_all,mask=observed_mask_point) #[bs,#points,ch]

            # if self.conditional:
            #     #Get Labels to the same shape as X and concat 
            #     label_x = label.repeat(1,x_norm_point.shape[1],1)
            #     x_norm_point = torch.cat([x_norm_point,label_x],dim=-1)

            #Encode to Z
            outputs_z_dict = self._encode_z(coordinate=self.encoded_coors_point_all, features=None, x_h=x, t_h=ts, observed_mask = observed_mask_full, nan_mask=nan_mask) #bs #points K
            qz_x = outputs_z_dict["qz_x"] 
            # z = self._sample_z(qz_x)
            z = outputs_z_dict["mean_z"]
            # PSEUDO C
            outputs_c_dict = self._encode_c(coordinate=self.encoded_coors_point_all, features=None, z=z) #bs #points K

            if self.use_same_label:
                label = torch.ones_like(label)
                if cfg.dataset.name in ['P12','P12_new']:
                    label = label * -1
            
            if self.conditional:
                #Get labels to the same shape as Z and concat
                # label_z = label/self.total_dim_cond
                #make it float 32
                # label_z = label_z.float()
                # z = torch.cat([z,label_z],dim=-1)
                #get 1 hot encoding
                #one hot encode the class using self.dims_c
                # label = torch.nn.functional.one_hot(label.long(), num_classes=self.dims_c).float() #torch.Size([8, 1, 30])
                if self.cond_one_hot:
                    hot_label = torch.nn.functional.one_hot(label.long()-1, num_classes=self.dims_target).float() #torch.Size([8, 1, 30])
                else:
                    hot_label = label.float()
                label_z = self._encode_cond(hot_label.squeeze(1))
                z = torch.cat([z,label_z],dim=-1)

            #Decode to X
            logits_x = self._decode(z,self.encoded_coors_point_all) #[bs*K, #points, 2*ch]
            # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0)) # [bs, #points, 2*ch, K]
            logits_x = logits_x.permute((1,2,3,0)) # [bs, #points, 2*ch, K]

            #Get Likelihood
            if "logistic" in self.distr_x:
                mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                        return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]
            else:
                mean_x, px_z = self.lik_x(logits=logits_x,
                                        return_mean=True, dim=2) #this has the whole [bs,pixels,ch,K]

            loss_dict = ELBO_mixture_observed(qz_x=qz_x,
                            pz=self.prior_z,
                            px_z=px_z,
                            x=x_norm_point_all, #[bs,full_points,ch]
                            beta=self.beta_z,
                            K=self.K,
                            mask = observed_mask_point if elbo_mode == 'observed' else (~nan_mask if nan_mask is not None else np.ones_like(observed_mask_point, dtype=bool)),
                            qc=outputs_c_dict["qc"],
                            pc=outputs_c_dict["pc"],
                            beta_c = self.beta_c,
                            non_nan_scaling=nonnan_scale)
            
            loss_dict['loss'] = - (loss_dict['elbo'])
            loss_dict['missing_perc_tau'] = (~observed_mask_point).sum() / len(observed_mask_point.flatten())
            loss_dict['missing_perc_nan'] = nonnan_scale
            loss_dict['missing_perc_total'] = loss_dict['missing_perc_nan']*loss_dict['missing_perc_tau']


        elif self.task == 'forecasting':
            x_h, x_f, t_h, t_f, mod, perm_h, perm_f, c = batch
            bs = x_h.shape[0]
            nan_mask_h = self.create_nan_mask(x_h)
            nan_mask_f = self.create_nan_mask(x_f)

            coors_point_all_h, x_norm_point_all_h, _ = self._to_coordinates_and_features(x_h,t_h)
            coors_point_all_f, x_norm_point_all_f, _ = self._to_coordinates_and_features(x_f,t_f)
            x_norm_point_all_h, coors_point_all_h, encoded_coors_point_all_h = self.process_coordinates(x_norm_point_all_h, coors_point_all_h, nan_mask_h)
            x_norm_point_all_f, coors_point_all_f, encoded_coors_point_all_f = self.process_coordinates(x_norm_point_all_f, coors_point_all_f, nan_mask_f)

            #Get scale coeff for total/non-nan pixels
            nonnan_scale_h = torch.sum(~nan_mask_h[0]).float() / nan_mask_h[0].flatten().shape[0]
            nonnan_scale_f = torch.sum(~nan_mask_f[0]).float() / nan_mask_f[0].flatten().shape[0]
            #get total nonnan_scale
            total_nonnnan_scale = ( torch.sum(~nan_mask_h[0]).float()+ torch.sum(~nan_mask_f[0]).float()) / (nan_mask_h[0].flatten().shape[0]+nan_mask_f[0].flatten().shape[0])

            series_h, observed_mask_h, observed_mask_point_h = create_mask_missingness(x_norm_point_all_h.permute(0,2,1),missingness=0)
            series_f, observed_mask_f, observed_mask_point_f = create_mask_missingness(x_norm_point_all_f.permute(0,2,1),missingness=missingness)
            observed_mask_h = torch.tensor(observed_mask_h).to(x_h.device)
            observed_mask_f = torch.tensor(observed_mask_f).to(x_h.device)

            observed_mask_full_h = self.create_observed_mask(x_h, nan_mask_h, observed_mask_h)
            observed_mask_full_f = self.create_observed_mask(x_f, nan_mask_f, observed_mask_f)

            x_norm_point_h = self.mask_to_input(input=x_norm_point_all_h,mask=observed_mask_point_h) #[bs,#points,ch]
            encoded_coors_point_h = self.mask_to_input(input=encoded_coors_point_all_h,mask=observed_mask_point_h) #[bs,#points,ch]
            coors_point_h =  self.mask_to_input(input=coors_point_all_h,mask=observed_mask_point_h) #[bs,#points,ch]

            t_h = coors_point_all_h.permute(0,2,1).unsqueeze(2)
            t_f = coors_point_all_f.permute(0,2,1).unsqueeze(2)
            
            #Encode to Z
            #concat x_h and x_f
            x_hf = torch.cat([x_h,x_f],dim=-1)
            t_hf = torch.cat([t_h,t_f],dim=-1)
            observed_mask_full_hf = torch.cat([observed_mask_full_h,observed_mask_full_f],dim=1)
            observed_mask_full_hf_prior = torch.cat([observed_mask_full_h,torch.zeros_like(observed_mask_full_f)],dim=1)
            nan_mask_hf = torch.cat([nan_mask_h,nan_mask_f],dim=1)
            outputs_z_dict = self._encode_z(coordinate=None, features=None, x_h=x_hf, t_h=t_hf, observed_mask = observed_mask_full_hf, nan_mask=nan_mask_hf) #bs #points K
            
            qz_x = outputs_z_dict["qz_x"]
            z = self._sample_z(qz_x)

            outputs_z_dict_prior = self._encode_z(coordinate=None, features=None,x_h=x_hf, t_h=t_hf, observed_mask = observed_mask_full_hf_prior, nan_mask=nan_mask_hf, mode="prior") #bs #points K
            pz_x = outputs_z_dict_prior["qz_x"]
            
            self.data_converter.set_coors_manual(t_hf) #this is for temporal data
            coors_point_all, x_norm_point_all = self.data_converter.batch_to_coordinates_and_features(data_batch=torch.concatenate([x_h, x_f],dim=-1)) #[bs, h*w, 2] #[bs,h*w,ch] (so this is also proper for shapenet + voxels)
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)

            #now we need to use for all grid
            logits_x = self._decode(z,self.encoded_coors_point_all)
            logits_x = logits_x.permute((1,2,3,0)) # [bs, #points, 2*ch, K]
            # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0))

            pi_x = nn.functional.softmax(logits_x,dim=-2) # [bs, pix_full, ch, K]
            pi_x = torch.clamp(pi_x, min=1e-6, max=None)

            if "logistic" in self.distr_x:
                mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                        return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]
            else:
                mean_x, px_z = self.lik_x(logits=logits_x,
                                        return_mean=True, dim=2) #this has the whole [bs,pixels,ch,K]

            if elbo_mode == "observed":
                mask_all_obs = torch.concatenate([observed_mask_point_h,observed_mask_point_f],axis=-1)
            
            elif elbo_mode == "predictive":
                mask_all_obs = torch.concatenate([torch.zeros_like(observed_mask_point_h),observed_mask_point_f],axis=-1)

            elif elbo_mode == "full":
                mask_all_obs = torch.concatenate([torch.ones_like(observed_mask_point_h),torch.ones_like(observed_mask_point_f)],axis=-1)

            else:
                raise ValueError("Unknown ELBO mode")
            
            loss_dict = ELBO_mixture_observed(qz_x=qz_x,
                            pz=pz_x,
                            px_z=px_z,
                            x=x_norm_point_all, #[bs,full_points,ch]
                            beta=self.beta_z,
                            K=self.K,
                            mask = mask_all_obs)
            
            loss_dict['loss'] = - (loss_dict['elbo'])
            loss_dict['missing_perc_tau'] = round((~observed_mask_point_h).sum().item() / len(observed_mask_point_h.flatten()),2)
            loss_dict['missing_perc_no_nan'] = round(nonnan_scale_h.item(), 2)
            loss_dict['missing_perc_no_nan_f'] = round(nonnan_scale_f.item(), 2)
            loss_dict['missing_perc_total'] = round(total_nonnnan_scale.item(),2)
            
        return loss_dict
    
    @staticmethod
    def create_nan_mask(x: torch.Tensor) -> torch.Tensor:
        """
        Creates a boolean mask for NaN values in the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to check for NaN values
            
        Returns:
            torch.Tensor: Boolean mask where True indicates NaN values, reshaped to (batch_size, sequence_length, 1)
        """
        bs = x.shape[0]  # Get batch size
        
        if torch.any(torch.isnan(x)):
            nan_mask = torch.isnan(x).reshape(x.shape)
            nan_mask = nan_mask.reshape(bs, 1, -1).permute((0, 2, 1))
        else:
            nan_mask = torch.zeros_like(x, dtype=torch.bool).reshape(bs, 1, -1).permute((0, 2, 1))
        
        return nan_mask
    
    def process_coordinates(self, 
                          x_norm_point_all: torch.Tensor,
                          coors_point_all: torch.Tensor,
                          nan_mask: torch.Tensor) -> tuple:
        """
        Process normalized points and coordinates by applying NaN masking and reshaping.
        
        Args:
            x_norm_point_all (torch.Tensor): Normalized points tensor
            coors_point_all (torch.Tensor): Coordinates tensor
            nan_mask (torch.Tensor): Boolean mask for NaN values
            
        Returns:
            tuple: (processed normalized points, processed coordinates, encoded coordinates)
        """
        bs = x_norm_point_all.shape[0]  # Get batch size
        
        # Process normalized points
        x_norm_processed = x_norm_point_all[~nan_mask].reshape(bs, -1, 1)
        
        # Process coordinates
        # Repeat mask for 2D coordinates
        expanded_mask = np.repeat(~nan_mask.cpu(), 2, axis=2)
        coors_processed = coors_point_all[expanded_mask].reshape(bs, -1, 2)
        
        # Create encoded coordinates
        encoded_coors = self.create_encoded_coordinates_fwd(coordinates=coors_processed)
        
        return x_norm_processed, coors_processed, encoded_coors
    def create_observed_mask(self, x: torch.Tensor, 
                        nan_mask: torch.Tensor, 
                        observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Creates an observed mask for ELBO calculation.
        
        Args:
            x (torch.Tensor): Input tensor to match shape [bs, 1, ch, len]
            nan_mask (torch.Tensor): Boolean mask indicating NaN values [bs, #point, 1]
            observed_mask (torch.Tensor): Original observed mask [bs, 1, #point]
            
        Returns:
            torch.Tensor: Full observed mask reshaped to (batch_size, sequence_length, 1) 
        """
        bs = x.shape[0]  # Get batch size
        
        # Create initial zero mask with same shape as input
        observed_mask_full = torch.zeros_like(x, dtype=torch.bool).reshape(bs, 1, -1).permute((0, 2, 1)).to(x.device) #[bs, #point, 1]
        
        # Apply observed mask to non-NaN positions
        observed_mask_full[nan_mask == False] = torch.tensor(observed_mask, dtype=torch.bool).flatten().to(x.device) #p12 seems observed mask has a different axis should be reordered (0,2,1)
        
        # Reshape to final dimensions
        observed_mask_full = observed_mask_full.reshape(bs, -1, 1)
        
        return observed_mask_full
    
    @torch.no_grad()
    def reconstruct2(self, mask=None, n_sample =1, x_h=None, x_f=None, t_h=None, t_f=None, label=None, get_full=False, **kwargs):

        """
        coordinate_grid: [bs, #point_obs, 2]
        input_x: [bs, #point_obs, ch]
        resolution: [#point]
        out_coordinates: [bs, #point_full, 2]
        mask: [bs, #point_full]
        n_sample: int
        x_mu_std: [bs, ch, 2]
        input_original: [bs, size*size, ch]
        label: [bs, cond_dim]
        """

        if self.task == 'imputation':
            bs = x_h.shape[0]
            Lh = x_h.shape[-1]
            #repeat mask for bs
            # mask = mask.repeat(bs,1)
            # mask = mask.reshape(bs,*x_h.shape[-2:])
            nan_mask_h = self.create_nan_mask(x_h)

            coors_point_all, x_norm_point_all_h, _ = self._to_coordinates_and_features(x_h,t_h)
            encoded_coors_point_all_h = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)
            # x_norm_point_all_h, coors_point_all_h, encoded_coors_point_all_h = self.process_coordinates(x_norm_point_all_h, coors_point_all_h, nan_mask_h)
            
            if mask.shape == x_h.shape:
                observed_mask = mask.reshape(bs,-1,1)
            else:
                observed_mask = mask.repeat(bs,1)[:,:,None].to(x_h.device)
            # observed_mask_h = observed_mask_h[nan_mask_h.cpu() == False]
            # observed_mask_full_h = self.create_observed_mask(x_h, nan_mask_h, observed_mask_h) #observed_mask_h shape (8,1,512)
            # observed_mask_h = observed_mask_h.to(x_h.device)

            observed_mask_full = torch.zeros_like(x_h,dtype=torch.bool).reshape(bs,1,-1).permute((0,2,1)).to(x_h.device)
            observed_mask_full[nan_mask_h==False] = torch.tensor(observed_mask[nan_mask_h==False], dtype=torch.bool).flatten().to(x_h.device) #this is for the ELBO calculation
            observed_mask_full = observed_mask_full.reshape(bs,-1,1)

            x_hf = x_h
            t_hf = t_h
            observed_mask_full_hf = observed_mask_full
            nan_mask_hf = nan_mask_h
            outputs_z_dict = self._encode_z(coordinate=None, features=None,x_h=x_hf, t_h=t_hf, observed_mask = observed_mask_full_hf, nan_mask=nan_mask_hf) #bs #points K
            qz_x = outputs_z_dict["qz_x"]
            z = outputs_z_dict["mean_z"]
            
            #concat condition
            #one hot encode the class using self.dims_c
            if self.use_same_label:
                label = torch.ones_like(label) 
                if cfg.dataset.name in ['P12','P12_new']:
                    label = label * -1

            if self.conditional:
                if self.cond_one_hot:
                    hot_label = torch.nn.functional.one_hot(label.long()-1, num_classes=self.dims_target).float() #torch.Size([8, 1, 30])
                else:
                    hot_label = label.float()
                label_z = self._encode_cond(hot_label.squeeze(1))
                z = torch.cat([z,label_z],dim=-1)
                #Get labels to the same shape as Z and concat
                # label_z = label/self.total_dim_cond
                #make it float 32
                # label_z = label_z.float()
                # z = torch.cat([z,label_z],dim=-1)
            
            #Decode to X
            encoded_out_coords_recons_all = encoded_coors_point_all_h
            logits_x = self._decode(z,
                                encoded_out_coords_recons_all) #[bs*K, #points, 2*ch]
            # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0)) # [bs, #points, 2*ch, K]
            logits_x = logits_x.permute((1,2,3,0)) # [bs, #points, 2*ch, K]

            #Get Likelihood
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=2) #this has the whole (bs, #points_all, ch, K)
            
            #Sample X
            x_rec = px_z.sample([n_sample]) ##[n_sample, bs, #points, ch, K]

            mean_x_out = (mean_x*1).sum(-1) #  (bs, #points_all, ch, K) * (bs, #points_all, 1, K) -> (bs, #points_all, ch)
            x_rec_out = (x_rec*1).sum(-1) # [n_sample, bs,ch,h,w,K] * [1,bs,#points_all, 1, K] -> [n_sample, bs, #points_all,ch]

            loss_dict = {}
            
            outputs_x_dict = {
                'mean_full': mean_x_out.permute(0,2,1).reshape(x_hf.shape), 
                'sample_full': x_rec_out.permute(0,1,3,2).reshape((n_sample,) + tuple(x_hf.shape[:])).permute(1,2,3,4,0)
            }

            nan_tensor = torch.zeros(tuple(observed_mask_full_hf.shape)).fill_(float('nan'))
            nan_tensor_shape = nan_tensor.shape
            nan_tensor[~nan_mask_hf] = mean_x_out[~nan_mask_hf].cpu()
            nan_tensor = nan_tensor.reshape(nan_tensor_shape)
            outputs_x_dict['mean'] = nan_tensor.permute(0,2,1).reshape(x_hf.shape)
            
            nan_tensor = torch.zeros((n_sample,) + tuple(observed_mask_full_hf.shape)).fill_(float('nan')).cpu()
            nan_tensor_shape = nan_tensor.shape
            nan_tensor[:,~nan_mask_hf] =x_rec_out[:,~nan_mask_hf].cpu()
            nan_tensor = nan_tensor.reshape(nan_tensor_shape)
            outputs_x_dict['sample'] = nan_tensor.permute(0,1,3,2).reshape((n_sample,) + tuple(x_hf.shape[:])).permute(1,2,3,4,0)

            outputs_x_dict = {**outputs_x_dict, **loss_dict}

            # if get_full:
            #     #get full tensor with False
            #     nan_mask_false = torch.zeros_like(nan_mask_h).bool() 
            #     coors_point_all_h, x_norm_point_all_h, _ = self._to_coordinates_and_features(x_h,t_h)
            #     _x, _c, encoded_coors_point_all_h = self.process_coordinates(x_norm_point_all_h, coors_point_all_h, nan_mask_false)

            #     logits_x = self._decode(z,
            #             encoded_coors_point_all_h) #[bs*K, #points, 2*ch]
            #     # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0)) # [bs, #points, 2*ch, K]
            #     logits_x = logits_x.permute((1,2,3,0)) # [bs, #points, 2*ch, K]

            #     #Get Likelihood
            #     mean_x, px_z = self.lik_x(logits=logits_x,
            #                             return_mean=True, dim=2) #this has the whole (bs, #points_all, ch, K)
                
            #     #Sample X
            #     x_rec = px_z.sample([n_sample]) ##[n_sample, bs, #points, ch, K]

            #     mean_x_out = (mean_x*1).sum(-1) #  (bs, #points_all, ch, K) * (bs, #points_all, 1, K) -> (bs, #points_all, ch)
            #     x_rec_out = (x_rec*1).sum(-1) # [n_sample, bs,ch,h,w,K] * [1,bs,#points_all, 1, K] -> [n_sample, bs, #points_all,ch]
            #     # return (x_hf,t_hf,observed_mask_full_hf,nan_mask_hf, z, encoded_out_coords_recons_all, logits_x, mean_x, mean_x_out)

            #     loss_dict = {}
            #     bs, _, ch, t = x_h.shape
            #     mean_x_out = mean_x_out.reshape(bs, 1, ch, t)
            #     x_rec_out = x_rec_out.reshape(n_sample, bs, 1, ch, t).permute(1,2,3,4,0)
                
            #     outputs_x_dict_full = {
            #         'mean_full': mean_x_out, 
            #         'sample_full': x_rec_out
            #     }

            #     outputs_x_dict = {**outputs_x_dict, **outputs_x_dict_full}




        
        elif self.task == 'forecasting':
            bs = x_h.shape[0]
            Lh = x_h.shape[-1]
            Lf = x_f.shape[-1]

            nan_mask_h = self.create_nan_mask(x_h)
            nan_mask_f = self.create_nan_mask(x_f)


            coors_point_all_h, x_norm_point_all_h, _ = self._to_coordinates_and_features(x_h,t_h)
            coors_point_all_f, x_norm_point_all_f, _ = self._to_coordinates_and_features(x_f,t_f)
            x_norm_point_all_h, coors_point_all_h, encoded_coors_point_all_h = self.process_coordinates(x_norm_point_all_h, coors_point_all_h, nan_mask_h)
            x_norm_point_all_f, coors_point_all_f, encoded_coors_point_all_f = self.process_coordinates(x_norm_point_all_f, coors_point_all_f, nan_mask_f)


            observed_mask_h = mask[:Lh][None, None, :].repeat(bs, 1, 1)
            observed_mask_f = mask[Lh:Lh+Lf][None, None, :].repeat(bs, 1, 1)
            observed_mask_h = observed_mask_h.to(x_h.device)
            observed_mask_f = observed_mask_f.to(x_h.device)
            observed_mask_full_h = self.create_observed_mask(x_h, nan_mask_h, observed_mask_h) #observed_mask_h shape (8,1,512)
            observed_mask_full_f = self.create_observed_mask(x_f, nan_mask_f, observed_mask_f) #observed_mask_f shape (8,1,720)

            t_h = coors_point_all_h.permute(0,2,1).unsqueeze(2)
            t_f = coors_point_all_f.permute(0,2,1).unsqueeze(2)
            
            #Encode to Z
            #concat x_h and x_f
            x_hf = torch.cat([x_h,x_f],dim=-1)
            t_hf = torch.cat([t_h,t_f],dim=-1)
            observed_mask_full_hf = torch.cat([observed_mask_full_h,observed_mask_full_f],dim=1)
            nan_mask_hf = torch.cat([nan_mask_h,nan_mask_f],dim=1)
            outputs_z_dict = self._encode_z(coordinate=None, features=None,x_h=x_hf, t_h=t_hf, observed_mask = observed_mask_full_hf, nan_mask=nan_mask_hf, mode="prior") #bs #points K
            qz_x = outputs_z_dict["qz_x"]
            z = outputs_z_dict["mean_z"]

            #Decode to X
            encoded_out_coords_recons_all = torch.cat([encoded_coors_point_all_h,encoded_coors_point_all_f],dim=1)
            logits_x = self._decode(z,
                                encoded_out_coords_recons_all) #[bs*K, #points, 2*ch]
            logits_x = logits_x.permute((1,2,3,0)) # [bs, #points, 2*ch, K]
            # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0)) # [bs, #points, 2*ch, K]

            #Get Likelihood
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=2) #this has the whole (bs, #points_all, ch, K)
            #Sample X
            x_rec = px_z.sample([n_sample]) ##[n_sample, bs, #points, ch, K]

            mean_x_out = (mean_x*1).sum(-1) #  (bs, #points_all, ch, K) * (bs, #points_all, 1, K) -> (bs, #points_all, ch)
            x_rec_out = (x_rec*1).sum(-1) # [n_sample, bs,ch,h,w,K] * [1,bs,#points_all, 1, K] -> [n_sample, bs, #points_all,ch]

            loss_dict = {}
            
            outputs_x_dict = {
                'mean': mean_x_out, 
                'sample': x_rec_out
            }


            nan_tensor = torch.zeros(tuple(observed_mask_full_hf.shape)).fill_(float('nan'))
            nan_tensor_shape = nan_tensor.shape
            nan_tensor[~nan_mask_hf] = outputs_x_dict['mean'].flatten().cpu()
            nan_tensor = nan_tensor.reshape(nan_tensor_shape)
            outputs_x_dict['mean'] = nan_tensor.permute(0,2,1).reshape(x_hf.shape)
            
            nan_tensor = torch.zeros((n_sample,) + tuple(observed_mask_full_hf.shape)).fill_(float('nan')).cpu()
            nan_tensor_shape = nan_tensor.shape
            nan_tensor[:,~nan_mask_hf] = outputs_x_dict['sample'].flatten().reshape(n_sample, -1).cpu()
            nan_tensor = nan_tensor.reshape(nan_tensor_shape)
            outputs_x_dict['sample'] = nan_tensor.permute(0,1,3,2).reshape((n_sample,) + tuple(x_hf.shape[:])).permute(1,2,3,4,0)


            outputs_x_dict = {**outputs_x_dict, **loss_dict}

        return outputs_x_dict, outputs_z_dict
    
    @torch.no_grad()
    def reconstruct(self, coordinate_grid, input_x, resolution=None, out_coordinates=None, mask=None, n_sample =1, x_mu_std=None, input_original=None, c=None, **kwargs):

        """
        coordinate_grid: [bs, #point_obs, 2]
        input_x: [bs, #point_obs, ch]
        resolution: [#point]
        out_coordinates: [bs, #point_full, 2]
        mask: [bs, #point_full]
        n_sample: int
        x_mu_std: [bs, ch, 2]
        input_original: [bs, size*size, ch]
        c
        """


        raise NotImplementedError("Reconstruct not implemented for tv_inr")
        # #Get Data
        # coors_point_all, x_norm_point_all = coordinate_grid, input_x
        # bs = coors_point_all.shape[0]
        # ch_count = x_norm_point_all.shape[-1]

        # #Preprocess Data
        # encoded_coords = [self.position_encoding(coord) for coord in coors_point_all] #this makes [bs,size*size,2] -> [bs,size*size,256]
        # encoded_coords_recons_all = torch.stack(encoded_coords, 0).expand(bs,-1,-1)        
        # encoded_out_coords = [self.position_encoding(coord) for coord in out_coordinates] #this makes [bs,size*size,2] -> [bs,size*size,256]
        # encoded_out_coords_recons_all = torch.stack(encoded_out_coords, 0).expand(bs,-1,-1)

        # #Encode to Z
        # if self.conditional:
        #     if self.use_same_label:
        #         c = torch.ones_like(c)
        #     label = torch.nn.functional.one_hot(c.long(), num_classes=self.total_dim_cond).float()
        #     label_x = label.repeat(1,x_norm_point_all.shape[1],1)
        #     x_norm_point_all = torch.cat([x_norm_point_all,label_x],dim=-1)

            
        # # outputs_z_dict = self._encode_z(coordinate=encoded_coords_recons_all, features=x_norm_point_all)
        # outputs_z_dict = self._encode_z(coordinate=encoded_coords_recons_all, features=x_norm_point_all, x_h=kwargs["x_h"], t_h=kwargs["t_h"], observed_mask = kwargs["observed_mask"], nan_mask=kwargs["nan_mask"]) #bs #points K
        # qz_x = outputs_z_dict["qz_x"]
        # z = outputs_z_dict["mean_z"]

        # # return (kwargs["x_h"],kwargs["t_h"],kwargs["observed_mask"],kwargs["nan_mask"],z, encoded_out_coords_recons_all, logits_x)            

        # logits_pi = self.prior_cat(coordinates=encoded_out_coords_recons_all,z=z)
        # # logits_post_final = torch.zeros_like(logits_pi)
        # logits_post = self.post_cat(coordinates=encoded_coords_recons_all,features=x_norm_point_all,z=z)
        # #mask to tensor
        # mask = torch.tensor(mask, dtype=torch.bool)
        # # We expand the mask to broadcast it across the third dimension (K)
        # mask_expanded = mask.unsqueeze(-1).expand_as(logits_pi)  # Shape: (bs, full_size, K)

        # pi = nn.functional.softmax(logits_pi,dim=-1) # [bs, #points, K]
        # post = nn.functional.softmax(logits_post,dim=-1) # [bs, #points, K]

        # #only on the different pixels otherwise means are same KL 0
        # pc = torch.distributions.categorical.Categorical(probs=pi[mask].reshape(*logits_post.shape))
        # qc = torch.distributions.categorical.Categorical(probs=post)
        # #use prior for the unobserved pixels
        # pi[mask] = post.reshape(-1,pi.shape[-1]).clone()
        # pi = pi.reshape(*logits_pi.shape)

        # if self.conditional:
        #     label_z = label[:,0]
        #     zc = torch.cat([z,label_z],dim=-1)
        #     z = zc

        # #Decode to X
        # logits_x = self._decode(z,
        #                         encoded_out_coords_recons_all) #[bs*K, #points, 2*ch]
        # logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0)) # [bs, #points, 2*ch, K]

        # #Get Likelihood
        # mean_x, px_z = self.lik_x(logits=logits_x,
        #                         return_mean=True, dim=2) #this has the whole (bs, #points_all, ch, K)
        # #Sample X
        # x_rec = px_z.sample([n_sample]) ##[n_sample, bs, #points, ch, K]
        # pi = pi.unsqueeze(-2) # [bs, #points, 1, K]

        # mean_x_out = (mean_x*pi).sum(-1) #  (bs, #points_all, ch, K) * (bs, #points_all, 1, K) -> (bs, #points_all, ch)
        # x_rec_out = (x_rec*pi.unsqueeze(0)).sum(-1) # [n_sample, bs,ch,h,w,K] * [1,bs,#points_all, 1, K] -> [n_sample, bs, #points_all,ch]
        # # return (kwargs["x_h"],kwargs["t_h"],kwargs["observed_mask"],kwargs["nan_mask"],z, encoded_out_coords_recons_all, logits_x, mean_x, mean_x_out)

        # if len(mask.shape)==1:
        #     #repeat mask for all samples
        #     mask = mask.unsqueeze(0).expand(bs, -1, -1)

        # loss_dict = {}
        
        # outputs_x_dict = {
        #     'mean': mean_x_out, 
        #     'sample': x_rec_out
        # }

        # outputs_x_dict = {**outputs_x_dict, **loss_dict}
        # return outputs_x_dict, outputs_z_dict
    
    def _encode_z(self, coordinate, features, x_h, t_h, observed_mask=None, nan_mask=None, mode="post"):
        # logits_z = self.encoder_z(coordinate, features)
        # print("CHECKING SHAPES in _encode_z")
        # print(coordinate.shape)
        # print(features.shape)
        if self.encoder_type == "pointconv":
            logits_z = self.encoder_z(coordinate, features)
        elif self.encoder_type == "transformer":
            if mode == "prior":
                logits_z = self.encoder_z_prior(x_h, t_h, observed_mask, nan_mask)
            else:
                logits_z = self.encoder_z(x_h, t_h, observed_mask, nan_mask)
        else:
            raise ValueError("Unknown encoder type")
        
        mean_z, qz_x = self.lik_z(logits=logits_z,
                                    return_mean=True)

        outputs_dict = {
            "mean_z": mean_z,
            "qz_x": qz_x,
        }

        return outputs_dict
    
    def _encode_cond(self, condition):
        '''
        condition: [bs, cond_dim] #one hot encoded 
        '''
        
        encoded_condition = self.cat_cond_encoder(condition)

        return encoded_condition
    
    def _encode_c(self, coordinate, features, z):

        if self.K == 1:
            outputs_dict = {
                'qc': None,
                'pc': None,
            }
            return outputs_dict
        
        #Categorical Logits
        logits_prior = self.prior_cat(coordinates=coordinate,z=z)
        logits_post = self.post_cat(coordinates=coordinate,features=features,z=z)
        if self.learn_residual_posterior==True:
            logits_post = logits_prior + logits_post
        
        #PIs (normalized)
        pi_prior = nn.functional.softmax(logits_prior,dim=-1)
        pi_post = nn.functional.softmax(logits_post,dim=-1) #bs,h*w,K
        
        #Categorical Distributions
        qc = torch.distributions.categorical.Categorical(probs=pi_post)
        pc = torch.distributions.categorical.Categorical(probs=pi_prior)

        outputs_dict = {
            'qc': qc,
            'pc': pc,
            'pi_prior': pi_prior,
            'pi_post': pi_post,
            'logits_prior': logits_prior,
            'logits_post': logits_post,
        }

        return outputs_dict

    def _return_image_format(self, logits, coordinates=None, resolution=None, bs=None, K=None, mask=None):
        if self.task == "image":
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,0))

        elif self.task == "temporal" or self.task == "forecasting" or self.task == "imputation":
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution) #(bs*K, 2*ch, #points)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,0)) #(bs*K, 2*ch, #points) -> (bs, #points, 2*ch, K)

        elif self.task == "era5_polar":
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,0))
            logits = logits[:,[2]] #even though we use polar for input it returns 2d due to dataconverter
        elif self.task == "voxels_chairs":
            if mask is not None:
                logits[:,~mask[0],:]=0
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,5,0))

        return logits


    def _get_layer_name_torch(self, param_type, indx):
        if param_type == 'w':
            return f"layers.{indx}.0.weight"
        elif param_type == 'b':
            return f"layers.{indx}.0.bias"
        else:
            raise ValueError("Unknown parameter type")
        
    def _get_layer_name_inr(self, layer_name):        
        # Split the string by '.' and get the relevant parts
        parts = layer_name.split('.')
        
        if len(parts) < 4 or parts[0] != 'layers':
            return None  # Return None if the format does not match expected structure
        
        # Extract layer number and parameter type
        layer_number = parts[1]   # Second part (layer number)
        param_type = parts[3]     # Fourth part (weight or bias)
        
        # Map "weight" to "w" and "bias" to "b"
        prefix = 'w' if param_type == 'weight' else 'b' if param_type == 'bias' else None
        
        # Return None if param_type is neither "weight" nor "bias"
        if prefix is None:
            return None
        
        # Return the mapped name with prefix and layer number
        return f'{prefix}{layer_number}'

    def _decode(self, z, coordinates_encoded=None):
        created_inr = self.hyper_list[0](latents=z)
        bs = z.shape[0]
                
        #these are weights and biases for all layers
        for param_name in self.hyper_list[0].hypernetwork.inr.non_shared_layer_names:
            param_type = param_name[0]
            indx = param_name[1]
            layer_name = self._get_layer_name_torch(param_type, indx)
            if param_type == 'w':
                self.fparams[layer_name] = created_inr[param_name].permute(0,2,1)
            elif param_type == 'b':
                self.fparams[layer_name] = created_inr[param_name].squeeze(1)

        for param_name in self.hyper_list[0].hypernetwork.inr.shared_layer_names:
            param_type = param_name[0]
            indx = param_name[1]
            layer_name = self._get_layer_name_torch(param_type, indx)
            if param_type == 'w':
                params_expand = self.fparams_shared[layer_name].expand(bs,-1,-1,-1).permute(1,0,2,3)
                self.fparams[layer_name] = params_expand.reshape(-1,self.fparams[layer_name].shape[-2],self.fparams[layer_name].shape[-1])
            elif param_type == "b":
                params_expand = self.fparams_shared[layer_name].expand(bs,-1,-1).permute(1,0,2)
                self.fparams[layer_name] = params_expand.reshape(-1,self.fparams[layer_name].shape[-1])


        cnew = coordinates_encoded.expand(self.K,-1,-1,-1)
        cnew = cnew.reshape(-1, cnew.shape[-2],cnew.shape[-1]) #this also has the same bs/bs/bs K times

        import copy
        base_model = copy.deepcopy(self.models_Kbs[0])
        base_model.to('meta')
    
        def call_single_model(params, buffers, data):
            return torch.func.functional_call(base_model, (params, buffers), (data,))

        logits_new = torch.vmap(call_single_model, randomness='different')(self.fparams, self.fbuffers, cnew) #K in the first dim
        logits_new = logits_new.unsqueeze(0)

        return logits_new

    def _sample_z(self, dist):
        #samples a batch of z
        z = dist.rsample()
        return z
    def _sample_c(self, dist):
        #samples a batch of c
        return NotImplementedError

    @torch.no_grad()
    def _predict(self, latent, use_super_resolution=False):
        pred_img_i = self.wrapper(latent=latent, use_super_resolution=use_super_resolution)  # (1, 3, 256, 256)
        return pred_img_i
    @torch.no_grad()
    def _sample(self, latent, use_super_resolution=False):
        pred_img_i = self.wrapper(latent=latent, use_super_resolution=use_super_resolution)  # (1, 3, 256, 256)
        return pred_img_i
    
    def set_z_prior_distr(self,dist_name):
        
        self.prior_distr_z = PriorDistribution(dist_name,self.dim_z,self.device)
        
        return self.prior_distr_z
    
        
    def _to_coordinates_and_features(self, batch:torch.Tensor, ts:torch.Tensor):
        '''
        This function takes the feature and coordinates and proprocess both; then uses normalized coordinates are set as the coordinates.
        '''
        x = batch
        device = x.device
        x_norm_im, x_norm_im_bn, x_mu_std= self.preprocess_batch(x) #all of them (including input) [bs, 1, 28, 28] #if scaler is minmax1 they are same
        ts_norm = self.preprocess_batch_temporal(ts.cpu()).to(device) #all of them (including input) [bs, 1, 28, 28] #if scaler is minmax1 they are same
        self.data_converter.set_coors_manual(ts_norm) #this is for temporal data
        coors_point_all, x_norm_point_all = self.data_converter.batch_to_coordinates_and_features(data_batch=x_norm_im) #[bs, h*w, 2] #[bs,h*w,ch]
        return coors_point_all, x_norm_point_all, x_mu_std
    # def _create_scenerios(self):

def prediction_metrics_temporal( theta_x: torch.Tensor, x: torch.Tensor, mask:torch.Tensor, window_len=5, stride=3, temporal=True, history_size = None, experiment_tau = None):
        """
        #TODO implement there also MAE
        Given theta_x (x_pred), x (x_gt) computes RMSE metric after masking them with temporal mask
        Args:
            x [bs, ch, dim_x, T_full]
            theta_x [bs, ch, dim_x, T_full]
            mask [T] : observed pixels/coors
            temporal : bool
        returns:
            if temporal metric returns a list of convolution results
            if not temporal  
        """

        bs = x.shape[0]
        x = x.cpu()
        theta_x = theta_x.cpu()
        mask = mask.cpu()
        if history_size is None:
            history_size = mask.shape[-1]
        
        theta_x_denorm = theta_x
        x_denorm = x

        nan_mask  = torch.isnan(x)
        non_nan_mask = ~torch.isnan(x) #bs ch dim_x T
        # non_nan_mask[~nan_mask] = mask
        # non_nan_mask = non_nan_mask.reshape(*x.shape)


        if len(x.shape) == 3:
            bs, dim_x, T = x.shape
        if len(x.shape) == 4:
            bs, ch, dim_x, T = x.shape #?

        if len(mask.shape) == 3:
            tau = mask[:,:,:history_size].sum().item()/(bs*dim_x*T) #number of observed pixels
        elif len(mask.shape) == 4: #should be L
            tau = mask[:,:,:,:history_size].sum().item()/(bs*dim_x*T)
        elif len(mask.shape) == 1: #should be T
            tau = mask[:history_size].sum().item()/ mask[:history_size].reshape(-1).shape[0] #number of observed pixels
        # tau = mask.sum() / non_nan_mask.sum()
        missing_pix = T * (1-tau)

        if mask.shape != x.shape:
            mask_obs = mask.unsqueeze(0).repeat(bs,1).reshape(x.shape)
        else:
            mask_obs = mask
        mask_missing = ~mask_obs #change it to missing ones
        mask_missing_nan = (mask_missing | nan_mask) #missing and nan together.
        mask_observed_nan = (mask_obs | nan_mask) #observed and nan together

        if len(x.shape) == 3 and temporal==True:
            print("x should have 4 dimensions for temporal case")
            raise NotImplementedError
        
        # if len(x.shape) == 4 and temporal==False:
        #     print("x should have 3 dimensions for non-temporal case")
        #     raise NotImplementedError

        mse_obs = calculate_mse(theta_x_denorm, x_denorm, mask_missing_nan, temporal)
        mae_obs = calculate_mae(theta_x_denorm, x_denorm, mask_missing_nan, temporal)

        
        if missing_pix == 0 and cfg.dataset.task == "imputation":
            return {
                f'mse_obs': mse_obs,
                f'mae_obs': mae_obs,
            }
        else:

            mse_missing = calculate_mse(theta_x_denorm, x_denorm, mask_observed_nan, temporal)
            mae_missing = calculate_mae(theta_x_denorm, x_denorm, mask_observed_nan, temporal)

            mse_full = calculate_mse(theta_x_denorm, x_denorm, nan_mask, temporal)
            mae_full = calculate_mae(theta_x_denorm, x_denorm, nan_mask, temporal)


            result_dict = {
                f'mse_missing_{experiment_tau:.2f}': mse_missing,
                f'mae_missing_{experiment_tau:.2f}': mae_missing,

                f'mse_obs_{experiment_tau:.2f}': mse_obs,
                f'mae_obs_{experiment_tau:.2f}': mae_obs,

                f'mse_full_{experiment_tau:.2f}': mse_full,
                f'mae_full_{experiment_tau:.2f}': mae_full
            }

            if cfg.dataset.task == "forecasting": #mask is missing ones, which are excluded from the loss #TODO we need to fix this
                mask_horizon = torch.zeros_like(~mask, dtype=torch.bool)
                mask_horizon[:history_size] = True
                
                mse_frc = calculate_mse(theta_x_denorm, x_denorm, mask_horizon, temporal)
                mae_frc = calculate_mae(theta_x_denorm, x_denorm, mask_horizon, temporal)

                #add them to the result_dict
                result_dict[f'mse_frc_{tau:.2f}_{window_len}'] = mse_frc
                result_dict[f'mae_frc_{tau:.2f}_{window_len}'] = mae_frc

                #mask imputation
                mask_imp = mask.clone()
                mask_imp = ~mask_imp
                mask_imp[history_size:] = True

                mse_imp = calculate_mse(theta_x_denorm, x_denorm, mask_imp, temporal)
                mae_imp = calculate_mae(theta_x_denorm, x_denorm, mask_imp, temporal)

                #add them to the result_dict
                result_dict[f'mse_frc_imp_{tau:.2f}_{window_len}'] = mse_imp
                result_dict[f'mae_frc_imp_{tau:.2f}_{window_len}'] = mae_imp

            return result_dict
        
    
def prediction_metric_cross_corr_temporal( theta_x: torch.Tensor, x: torch.Tensor, mask:torch.Tensor, window_len=5, stride=3, temporal=True)-> torch.Tensor:
        """
        Given theta_x (x_pred), x (x_gt) computes CROSS CORR metric after masking them with temporal mask
        Args:
            theta_x [bs, L, T, dim_x]
            x [bs, L, T, dim_x]
            mask [bs, L, T]
        returns:
            metric [bs, L, T, dim_x]
        """
        # TODO #: maybe upgrade the documentation a bit?
        # theta_x_denorm = self.denormalize_x(theta_x)
        # x_denorm = self.denormalize_x(x)
        # x is already denormalized in training/test step
        theta_x_denorm = theta_x
        x_denorm = x

        if len(x.shape) == 3 and temporal==True:
            print("x should have 4 dimensions for temporal case")
            raise NotImplementedError
        
        if len(x.shape) == 4 and temporal==False:
            print("x should have 3 dimensions for non-temporal case")
            raise NotImplementedError

        if len(x.shape) == 4 and temporal==True:
            mask = mask.unsqueeze(-1).repeat(1,x.shape[1],1,x.shape[-1]) #[bs,L,T,d]
            bs, L,  T, dim_x = x.shape
            list1 = list(theta_x_denorm.reshape(-1,T,dim_x)) # list of BS*L: each jas tensor [T,d]
            list2 = list(x_denorm.reshape(-1,T,dim_x)) # list of BS*L: each jas tensor [T,d]
            mask = mask.reshape(-1,T,dim_x) # BS*L,T,d

            list1_masked, list2_masked = cross_cor_temporal(list1,list2,mask,dim_x,window_len) #each has bs*L,

            cross_cor = []

            for i,j in zip(list1_masked,list2_masked): #each has list1(list2([window_len,dim_x])) list1 len bs*L list2 len #windows
                _per_batch =[]
                for a,b in zip(i,j):
                    metric_all = []
                    for dim_x_s in range(dim_x):
                        metric = np.correlate(a[:,dim_x_s].flatten().cpu().detach().numpy(),b[:,dim_x_s].flatten().cpu().detach().numpy(),mode="same").max()/a[:,dim_x_s].shape[0]
                        metric_all.append(metric)
                    _per_batch.append(np.stack(metric_all,axis=0))
                cross_cor.append(_per_batch)
            return (cross_cor)

        if len(x.shape) == 3 and temporal==False:
            mask = mask.squeeze(1) #looks [bs,T] -> [bs,T]
            mask = mask.unsqueeze(-1).repeat(1,1,x.shape[-1])
            bs, T, dim_x = x.shape

            theta_x_denorm = theta_x_denorm.squeeze(1)
            
            list1 = list(theta_x_denorm)
            list2 = list(x_denorm)

            list1_masked, list2_masked = cross_cor_temporal(list1,list2,mask,dim_x,None)

            cross_cor = []

            for i,j in zip(list1_masked,list2_masked):
                #takes the max value of the correlation between x, theta_x
                metric = np.correlate(i.flatten().cpu().detach().numpy(),j.flatten().cpu().detach().numpy(),mode="same").max()/i.shape[0]
                cross_cor.append(metric)

            cross_cor_array = np.array(cross_cor,  dtype=np.float32).reshape(bs) #[bs]

            return torch.Tensor(cross_cor_array)

def calculate_mse(theta_x_denorm, x_denorm, mask, temporal=False):
    '''
    theta_x_denorm: [bs, ch, dim_x, T]
    x_denorm: [bs, ch, dim_x, T]
    mask: [bs, ch, dim_x, T] mask is missing ones, which are excluded from the loss
    '''

    if len(x_denorm.shape) == 2 and temporal == False:
        bs, num_points = theta_x_denorm.shape

        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0).repeat(bs,1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
        
        # Squared error summed over T and dim_x
        se = ((theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0))**2).sum((1)) #[bs]
        
        # Gets mean over T and dim_x
        metric = se / (~mask).sum((1)).float() #[bs]


    
    elif len(x_denorm.shape) == 3 and temporal == False:
        bs, dim_x, _ = theta_x_denorm.shape

        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0).unsqueeze(1).repeat(bs, dim_x, 1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
        
        # Squared error summed over T and dim_x
        se = ((theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0))**2).sum((1,2)) #[bs]
        
        # Gets mean over T and dim_x
        metric = se / (~mask).sum((1,2)).float() #[bs]
    
    
    elif len(x_denorm.shape) == 4 and temporal == False:
        bs, ch, dim_x, T = x_denorm.shape
        theta_x_denorm = theta_x_denorm.reshape(bs, 1, dim_x, T)

        if len(mask.shape) == 1:
            mask = mask.reshape(dim_x, T).unsqueeze(0).unsqueeze(1).repeat(bs,1, 1, 1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
            # mask = mask.unsqueeze(0).unsqueeze(1).repeat(bs, dim_x, 1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
        
        # Squared error summed over T and dim_x
        se = ((theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0))**2).sum((1,2,3)) #[bs]

        # Gets mean over T and dim_x
        metric = se / (~mask).sum((1,2,3)).float() #[bs]


    return metric



def calculate_mae(theta_x_denorm, x_denorm, mask, temporal=False):


    if len(x_denorm.shape) == 2 and temporal == False:
        bs, num_points = theta_x_denorm.shape

        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0).repeat(bs,1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
        
        ae = torch.abs(theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0)).sum((1)) #[bs]
        
        # Gets mean over T and dim_x
        metric = ae / (~mask).sum((1)).float() #[bs]
    
    elif len(x_denorm.shape) == 3 and temporal == False:
        bs, dim_x, _ = theta_x_denorm.shape
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0).unsqueeze(1).repeat(bs, dim_x, 1).to(theta_x_denorm.device)   #[bs,dim_x]-> [bs,dim_x,T]
        
        # Absolute loss summed over T and dim_x
        ae = torch.abs(theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0)).sum((1,2)) #[bs]
        
        # Gets mean over T and dim_x
        metric = ae / (~mask).sum((1,2)).float() #[bs]
        
    
    elif len(x_denorm.shape) == 4 and temporal == False:
        bs, L, dim_x, T = x_denorm.shape
        theta_x_denorm = theta_x_denorm.reshape(bs, 1, dim_x, T)

        if len(mask.shape) == 1:
            mask = mask.reshape(dim_x, T).unsqueeze(0).unsqueeze(1).repeat(bs,1, 1, 1).to(theta_x_denorm.device)  #[T]-> [bs,dim_x,T] we assume it is for every data.
        
        ae = torch.abs(theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0)).sum((1,2,3)) #[bs]

        metric = ae / (~mask).sum((1,2,3)).float() #[bs]
        
    return metric