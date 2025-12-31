
import numpy as np

def save_output(batch_original=None, batch_imputed=None, batch_full=None, batch_times=None, 
                mask=None, targets=None, tau=None):
    """
    Save the output of the model in a dictionary. Handles None values and converts
    PyTorch tensors to NumPy arrays.
    
    Args:
        batch_original: Original batch data (numpy array or torch tensor or None)
        batch_imputed: Imputed batch data (numpy array or torch tensor or None)
        batch_full: Full batch data (numpy array or torch tensor or None)
        batch_times: Time points (numpy array or torch tensor or None)
        mask: Mask tensor (numpy array or torch tensor or None)
        targets: Target values (numpy array or torch tensor or None)
        tau: Tau value to be repeated (float or int or None)
    
    Returns:
        dict: Dictionary containing all processed outputs
    """
    def to_numpy(tensor):
        """Helper function to convert tensor to numpy if needed"""
        if tensor is None:
            return None
        if not isinstance(tensor, np.ndarray):
            return tensor.cpu().numpy()
        return tensor

    # Convert all inputs to numpy arrays if they exist
    batch_original = to_numpy(batch_original)
    batch_imputed = to_numpy(batch_imputed)
    batch_full = to_numpy(batch_full)
    batch_times = to_numpy(batch_times)
    mask = to_numpy(mask)
    targets = to_numpy(targets)

    # Create output dictionary
    output = {}
    
    # Only process if batch_imputed exists (needed for shape information)
    bs = batch_original.shape[0] if batch_original is not None else batch_imputed.shape[0]
    batch_shape = batch_original.shape if batch_original is not None else None

    # Add all values to output dictionary
    output['batch_original'] = batch_original if batch_original is not None else None
    output['batch_imputed'] = batch_imputed if batch_imputed is not None else None
    output['batch_times'] = batch_times if batch_times is not None else None
    output['mask'] = mask.reshape(*batch_shape) if mask is not None else None
    output['targets'] = targets if targets is not None else None
    output['tau'] = np.repeat(tau, bs).reshape(-1, 1) if tau is not None else None
    output['batch_full'] = batch_full if batch_full is not None else None

    return output


# def save_output(batch_original = None, batch_imputed = None, batch_full=None, batch_times=None, mask = None, targets= None, tau=None):
#     """
#     Save the output of the model in a dictionary
#     """
#     #check if not numpy put to cpu
#     if not isinstance(batch_original, np.ndarray):
#         batch_original = batch_original.cpu().numpy()
#     if not isinstance(batch_imputed, np.ndarray):
#         batch_imputed = batch_imputed.cpu().numpy()
#     if not isinstance(batch_times, np.ndarray):
#         batch_times = batch_times.cpu().numpy()
#     if not isinstance(mask, np.ndarray):
#         mask = mask.cpu().numpy()
#     if not isinstance(targets, np.ndarray):
#         targets = targets.cpu().numpy()
#     if not isinstance(batch_full, np.ndarray) and batch_full is not None:
#         batch_full = batch_full.cpu().numpy()
    


#     output = {}
#     bs = batch_imputed.shape[0]
#     output['batch_original'] = batch_original
#     output['batch_imputed'] = batch_imputed
#     output['batch_times'] = batch_times
#     output['mask'] = mask.reshape(*batch_imputed.shape[-2:])
#     output['targets'] = targets
#     output['tau'] = np.repeat(tau, bs).reshape(-1, 1)
#     output['batch_full'] = batch_full
#     return output