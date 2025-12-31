
import torch
import matplotlib.pyplot as plt

def add_custom_stats_from_loss_dict(loss_dict):
    custom_stats = {}

    for key, value in loss_dict.items():
        if key not in ["loss","z"]:
            #if value is a tensor pass
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy()
            try:
                custom_stats[key] = value.item()
            except:
                custom_stats[key] = value
    return custom_stats

def get_fwd(model, x_h, x_f, t_h, t_f, z, perm_h, perm_f, c, t_mask=None, missingness=0.0, is_train=True, split="none"):
    
    if model.name == 'tv_inr':
        if model.task == 'imputation':
            import time
            start = time.time()
            loss_dict = model(batch = (x_h, t_h, c, t_mask), missingness=missingness)        
            end = time.time()
            # print("Time taken for imputation: ", end-start)
        elif model.task == 'forecasting':
            loss_dict = model(batch = (x_h, x_f, t_h, t_f, None, perm_h, perm_f, c), missingness=missingness)
    
    elif model.name == 'timeflow':
        if model.task == 'imputation':
            loss_dict = model(batch = (x_h, t_h, z, perm_h), missingness=missingness, is_train=is_train)
        elif model.task == 'forecasting':
            loss_dict = model(batch = (x_h, x_f, t_h, t_f, z, perm_h, perm_f), missingness=missingness, is_train=is_train)
        del loss_dict['modulations']

    elif model.name == 'saits_wrapper':
        x_h = x_h.squeeze(1).permute(0, 2, 1)
        input_dict = wrapper_masking_saits(x_h, missingness)
        # inputs = {'X': x_h, 'missing_mask': missing_mask, 'missing_ratio': missingness}
        if "train" in split.lower():
            stage = 'train'
        elif "val" in split.lower():
            stage = 'val'
        elif "test" in split.lower():
            stage = 'test'
        else:
            raise ValueError("Split should be either train, val or test")

        if model.task == 'imputation':
            loss_dict = model(inputs = input_dict, stage = stage)
            #change total loss to loss
            loss_dict['loss'] = torch.tensor(0.0,device=x_h.device)
            loss_dict['reconstruction_loss'] = loss_dict['reconstruction_loss'] * model.reconstruction_loss_weight
            loss_dict['imputation_loss'] = loss_dict['imputation_loss'] * model.imputation_loss_weight
            if model.MIT:
                loss_dict['loss'] += loss_dict['reconstruction_loss']
            if model.ORT:
                loss_dict['loss'] += loss_dict['imputation_loss']
        del loss_dict['imputed_data'], loss_dict['imputed_data_recons']
    
    elif model.name == 'deeptime_wrapper':
        x_h = x_h.squeeze(1).permute(0, 2, 1)
        x_f = x_f.squeeze(1).permute(0, 2, 1)
        x_h_time = torch.empty(x_h.shape[0], x_h.shape[1], 0).to(x_h.device)
        x_f_time = torch.empty(x_f.shape[0], x_f.shape[1], 0).to(x_f.device)
        if model.task == 'forecasting':
            loss_dict = {}
            preds = model(x = x_h, x_time = x_h_time, y_time = x_f_time)
            loss_dict['loss'] = model.compute_loss(preds, x_f)
    return loss_dict


def wrapper_masking_ours(X, artificial_missing_rate):
    # Assuming X is already a PyTorch tensor
    sample_num, ch, feature_num, seq_len= X.shape

    # Flatten the tensor
    X_flat = X.reshape(-1)

    # Generate indices for artificial missing values
    indices_for_holdout = random_mask(X_flat, artificial_missing_rate)

    observed_mask = torch.ones_like(X_flat).bool()
    observed_mask[indices_for_holdout] = False

    # Reshape everything back to original dimensions
    data_dict = {
        "observed_mask": observed_mask.reshape([sample_num, 1, feature_num, seq_len])
    }

    return data_dict


def wrapper_masking_saits(X, artificial_missing_rate):
    # Assuming X is already a PyTorch tensor
    sample_num, seq_len, feature_num = X.shape

    # Flatten the tensor
    X_flat = X.reshape(-1)

    # Generate indices for artificial missing values
    indices_for_holdout = random_mask(X_flat, artificial_missing_rate)

    # Create a copy with artificial missing values
    X_hat = X_flat.clone()
    X_hat[indices_for_holdout] = torch.tensor(float('nan'))

    # Create missing mask (1 where values exist, 0 where missing)
    missing_mask = (~torch.isnan(X_hat)).float()

    # Create indicating mask for artificial missing values
    indicating_mask = ((~torch.isnan(X_hat)) ^ (~torch.isnan(X_flat))).float()


    X_flat = torch.nan_to_num(X_flat, nan=0.0, posinf=None, neginf=None, out=None)
    X_hat = torch.nan_to_num(X_hat, nan=0.0, posinf=None, neginf=None, out=None)

    # Reshape everything back to original dimensions
    data_dict = {
        "X_holdout": X_flat.reshape([sample_num, seq_len, feature_num]),
        "X": X_hat.reshape([sample_num, seq_len, feature_num]),
        "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
        "indicating_mask": indicating_mask.reshape([sample_num, seq_len, feature_num]),
    }

    return data_dict
import torch
def random_mask(vector, artificial_missing_rate):
    """Generate indices for random mask in PyTorch"""
    assert len(vector.shape) == 1
    
    # Find indices of non-NaN values
    indices = torch.where(~torch.isnan(vector))[0]
    
    # Calculate number of indices to mask
    num_to_mask = int(len(indices) * artificial_missing_rate)
    
    # Randomly select indices to mask
    perm = torch.randperm(len(indices))
    indices_to_mask = indices[perm[:num_to_mask]]
    
    return indices_to_mask