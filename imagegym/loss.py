import torch
import torch.nn as nn
import torch.nn.functional as F

from imagegym.contrib.loss import *
import imagegym.register as register
from imagegym.config import cfg

import numpy as np

def ELBO_mixture_observed(qz_x, pz, px_z, x, beta, K, mask, qc=None, pc=None, beta_c=1, non_nan_scaling = 1):
    '''
    ELBO for mixture of 
    qz_x: q(z|x) [bs,dim_z]
    pz: p(z) [dim_z]
    px_z: p(x|z) [bs,num_pix_full,channels,K]
    x: ground truth [bs,num_pix_full,channels]
    beta: beta for KL loss
    beta_c: beta for KL loss of categorical
    qc: q(c|x) [bs,num_pix,K]
    pc: p(c) [bs,num_pix,K]
    K: number of mixture components
    mask: [bs,num_pix_full] observed pixels
    '''

    # Assuming `mask` can be either a PyTorch tensor or a NumPy array
    if isinstance(mask, torch.Tensor):
        mask_sum = torch.sum(mask).item()  # Convert the PyTorch tensor sum to a Python scalar
    else:
        mask_sum = np.sum(mask)  # NumPy array sum

    x_repeat = x.unsqueeze(-1).expand(-1, -1, -1, K) #better memory [bs,num_pix,ch,K]

    # loglikelihood of full data 
    if cfg.model.distr_x in ['ber'] and cfg.dataset.threshold == 0: #TODO check this one for mnist now/ because it is binarized
        log_prob_x = -torch.nn.BCEWithLogitsLoss(reduction="none")(px_z.logits,x_repeat) #sigmoid of logit gives the mean of dist
    elif cfg.model.distr_x in ['logistic']:
        log_prob_x = px_z.log_prob(x_repeat)
    else:
        # log_prob_x = px_z.log_prob(x_repeat)
        # Create a mask for NaN values
        mask_nan = torch.isnan(x_repeat)

        # Filter out NaN values by replacing them with 0
        x_repeat[mask_nan] = 0

        # Calculate log probabilities for all values (including the replaced NaNs)
        log_prob_x = px_z.log_prob(x_repeat)

        # Create a tensor of the same shape filled with NaN
        full_log_prob_x = torch.full_like(x_repeat, float('nan'))

        # Only copy the log probabilities for positions that were NOT NaN in the original
        full_log_prob_x[~mask_nan] = log_prob_x[~mask_nan]


    # we are only interested in full_log_prob_x[mask], full_log_prob_x is full (bs, #points, ch, K) including nans
    log_prob_x_obs = full_log_prob_x.clone()
    log_prob_x_obs[mask_nan] = 0
    log_prob_x_obs[~mask] = 0
    # log_prob_x_obs = full_log_prob_x[mask,:,:].reshape(x_repeat.shape[0],-1,x_repeat.shape[2],x_repeat.shape[3]) #[bs,#points(obs),ch,K]
    # pixel_count_obs = log_prob_x_obs.shape[1]
    
    if K>1: 
        qc_probs = qc.probs.unsqueeze(-2) #[bs,num_pix,1,K], qc.probs has shape [bs,num_pix,K]
    

    log_prob_x_obs = log_prob_x_obs*qc_probs if K>1 else log_prob_x_obs
    log_prob_x_obs = log_prob_x_obs.sum(dim=(1,2,3)) #sum over pixs,ch, K
    pixel_count_obs = mask.bool().sum(dim=1)
    if mask_sum>0:
        log_prob_x_unobs = full_log_prob_x.clone()
        log_prob_x_unobs[mask_nan] = 0
        log_prob_x_unobs[mask] = 0
        log_prob_x_unobs = log_prob_x_unobs.sum(dim=(1,2,3)) #sum over pixs,ch, K and mean over bs
        pixel_count_unobs = (~mask).sum(dim=1)
    else:
        log_prob_x_unobs = 0
        pixel_count_unobs = 0

    # log_prob_x_total = log_prob_x.reshape(x_repeat.shape[0],-1,x_repeat.shape[2],x_repeat.shape[3]) #[bs,#points(obs),ch,K
    log_prob_x_total = log_prob_x_obs + log_prob_x_unobs 

    kl_cat = 0
    if K>1:
        # KL loss (c)
        kl_cat = torch.distributions.kl.kl_divergence(qc, pc).sum(-1) #[bs,pixels] and then summed over pixels
        kl_cat = kl_cat.mean()

    # KL loss (z) (KL is already mean over batch)
    kl = compute_kl(pz,qz_x)


    if cfg.model.elbo_ll_scale: 
        # reminding the shapes:
        # mask [bs, #points_full]: observed ones + nans
        # mask_nan [bs, #points_full, ch, K]: only nans

        # log_prob_x [bs, #points(obs), ch, K]: 
        # full_log_prob_x [bs, #points(obs), ch, K]: log likelihood of all pixels, where nans have been has logp nan
        # log_prob_x_obs [bs]: log likelihood of observed pixels, where nans and non-observed have been replaced with 0 then summed 
        # log_prob_x_unobs [bs]: log likelihood of non-observed pixels, where nans and observed have been replaced with 0 then summed
        mask_obs = mask & ~mask_nan[:,:,0,0]
        # mask_obs.sum(dim=1)
        # (~mask).sum(dim=1)
        # mask_nan.sum(dim=(1,2,3))

        #rescaling log likelihoods
        pixel_count = pixel_count_obs+ pixel_count_unobs
        log_prob_x_obs = log_prob_x_obs * (pixel_count/pixel_count_obs) * (1/non_nan_scaling)
        log_prob_x_unobs = log_prob_x_unobs * (pixel_count/pixel_count_unobs) *  (1/non_nan_scaling) if pixel_count_unobs.sum()>0 else 0
        log_prob_x_total = log_prob_x_total *  (1/non_nan_scaling)

    #get mean over batch
    log_prob_x_obs = log_prob_x_obs.mean()
    log_prob_x_unobs = log_prob_x_unobs.mean() if pixel_count_unobs.sum()>0 else 0
    log_prob_x_total = log_prob_x_total.mean()

    assert torch.any(torch.isinf(log_prob_x_obs))==False, "log prob obs x has infinity"
    assert torch.any(torch.isinf(kl))==False, "KL has infinity"

    
    elbo = log_prob_x_obs - (beta) * kl - (beta_c) * kl_cat 

    my_dict = {
        'elbo': elbo,
        'log_prob_x': log_prob_x_obs,
        'kl': kl,
        'log_prob_x_unobs': log_prob_x_unobs,
        'log_prob_x_total': log_prob_x_total,
        'beta_z': torch.tensor(beta),
        'kl_cat': kl_cat,
        'beta_c': torch.tensor(beta_c)
    }
    return my_dict

def compute_kl(prior, posterior):
    if cfg.dataset.task == "forecasting":
        kl = torch.distributions.kl.kl_divergence(posterior,prior).sum(-1)
        kl = kl.mean()
        return kl
    if prior.name == "normal":
        # KL loss (z)
        kl = torch.distributions.kl.kl_divergence(posterior, prior.get_prior()).sum(-1) #[bs,dim_z] and then summed over dim_z
        kl = kl.mean()
        return kl
    
    if prior.name == "nf":
        if prior.params_nf_fixed:
            kl = torch.distributions.kl.kl_divergence(posterior, prior.get_prior()).sum(-1) #[bs,dim_z] and then summed over dim_z
            kl = kl.mean()
            return kl
        else:
            z_sample =posterior.sample((2**12,)) #num_sample, bs, dim_z
            z_sample = z_sample.reshape(-1, z_sample.shape[-1]) #all, dim_z
            kl = prior.get_prior().forward_kld(z_sample) #INFO kl: [] it is a scalar, mean over all
            return kl