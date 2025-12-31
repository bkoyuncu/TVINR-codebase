from torch import Tensor
import torch

class ToOneHot:

    def __init__(self, n_dims=10):
        self.n_dims = n_dims


    def __call__(self, target):
        y_onehot = torch.FloatTensor(self.n_dims)
        y_onehot.zero_()
        y_onehot[target] = 1
        return  y_onehot
    def __repr__(self):
        return self.__class__.__name__ + '()'

def create_boolean_mask(time_indices, mask_length=2881):
    """
    """
    # Maximum length of the boolean mask (2880 + 1 = 2881)

    # Create a False-filled boolean mask of length 2881
    boolean_mask = torch.zeros(mask_length, dtype=torch.bool)

    # Set the values at the specified indices to True
    boolean_mask[time_indices.long()] = True

    return boolean_mask

def apply_create_boolean_mask(times_out, dim=0):
    #every of them has 203 observations but at different points? in time and some of them are padded
    #find indices of padded observations with 0s #TODO
    n = times_out.shape[0]
    l = times_out.shape[1]

    time_masks = []
    for i in range(n):
        time_masks.append(create_boolean_mask(times_out[i, :, 0], mask_length=l+1))

    return torch.stack(time_masks, dim=dim)


