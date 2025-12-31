import torch
import torch.nn.functional as F
import torchvision
import os
from imagegym.config import cfg
import numpy as np
import torch
from scipy.stats import beta


def sample_beta_distribution(a, b, size=1000, missingness_rate = 0.0):
    # Generate samples from the beta distribution
    samples = beta.rvs(a, b, size=size)
    
    # Scale samples from [0, 1] to [0.06, 1]
    scaled_samples = missingness_rate + samples * (1 - missingness_rate)
    
    return scaled_samples

def create_fixed_mask_missingness(shape:tuple):
    """
    use only for chairs dataset since it is always sampling 4096 points
    """
    # assert len(x.shape) == 5 #so this is chairs
    bs, ch, h, w, d = shape
    observed_indices = random_indices(4096,h*w*d).unsqueeze(0)
    all = torch.zeros((h*w*d),dtype=bool).flatten()
    all[observed_indices]=True
    observed_mask = all.reshape(h,w,d)

    # observed_mask = torch.zeros_like(x,dtype=bool)
    # observed_mask[:,:,observed_indices]=True
    observed_mask = np.tile(observed_mask[np.newaxis,np.newaxis], (bs,ch,1,1,1))
    observed_mask_point = observed_mask[:,0].reshape(observed_mask.shape[0],-1) #bs,h*w
    return observed_mask, observed_mask_point
    

def random_indices(num_indices, max_idx):
        """Generates a set of num_indices random indices (without replacement)
        between 0 and max_idx - 1.

        Args:
            num_indices (int): Number of indices to include.
            max_idx (int): Maximum index.
        """
        # It is wasteful to compute the entire permutation, but it looks like
        # PyTorch does not have other functions to do this
        permutation = torch.randperm(max_idx)
        # Select first num_indices indices (this will be random since permutation is
        # random)
        return permutation[:num_indices]


def create_mask_missingness(x, missingness, dist_type='uniform'):
        """
        :param x: input tensor of batch
        :param missingness: missingness value float
        :return: x with missingness applied
        :return: observed mask of non-missingness
        """
        #put x to cpu first
        device = x.device  # Store original device
    
        # Move x to CPU for mask creation
        x = x.cpu()
    

        if cfg.dataset.missing_type == 'beta':
            a, b = 1, 1.7
            missing_rate = sample_beta_distribution(a, b, size=1, missingness_rate= 1-missingness)
            missing_rate = 1-missing_rate
        elif cfg.dataset.missing_type == 'uniform':
            if missingness == 1:
                missing_rate = np.random.rand(1) * 0.9  
            elif missingness > 0:
                missing_rate = missingness
                missing_rate = np.random.uniform(low=0,high=missing_rate)
            elif missingness ==0:
                missing_rate = 0.0
        elif cfg.dataset.missing_type == 'computed':
            missing_rate = missingness
        elif cfg.dataset.missing_type == 'list':
            possible_values = [0.05, 0.3, 0.5, 0.95, 1]
            #filter values that are greater than missingness
            if cfg.model.direction_scheduler== 'max_to_min':
                possible_values = [x for x in possible_values if x<=(1-missingness+1e-5)]
            if cfg.model.direction_scheduler== 'min_to_max':
                possible_values = [x for x in possible_values if x>=(1-missingness-1e-5)]
            missing_rate = np.random.choice(possible_values)
            missing_rate = 1-missing_rate
        else: #not implemented
            raise NotImplementedError

        if len(x.shape) == 3: #shapenet
            # raise NotImplementedError
            bs,ch, h = x.shape
            #get a mask of nan values in x
            nan_mask = torch.isnan(x)
            #initialize mask indices as everything can be masked
            full_ones = torch.ones_like(x).bool()
            #total elements in batch element
            num_total_elements = full_ones.sum(dim=(1,2))
            #mask the nan values
            full_ones[nan_mask] = False
            #count the number of non nan values
            count_no_nan = full_ones.sum(dim=(1,2)).int()
            #number of elements to mask (creating missingness)
            num_to_mask = (missing_rate * count_no_nan).int()
            #initialize mask
            mask = torch.ones_like(x, dtype=bool)

            indices_to_mask = []
            for i, (count, num, num_total) in enumerate(zip(num_total_elements, num_to_mask, num_total_elements)):
                # Convert tensor values to integers for sampling
                count_int = count.item()  # Extract integer from tensor
                num_int = int(num)  # Ensure num is integer
                num_total = int(num_total)
                
                # Only sample if we have values to sample and counts to sample from
                if num_int > 0 and count_int > 0:
                    # Make sure we don't try to sample more than available
                    num_int = min(num_int, count_int)
                    # mask for indices
                    mask_flat = full_ones[i].flatten()
                    # can be masked indeces
                    possible_indices = torch.arange(num_total)[mask_flat]
                    #shuffle possible indices
                    sampled_indices = possible_indices[torch.randperm(possible_indices.shape[0])][:num_int]
                    # Store the dimension and its sampled indices
                    indices_to_mask.append((i, sampled_indices))
                    mask[i,:,sampled_indices] = False #masked ones are not observes set to zero
            observed_mask = torch.Tensor(mask).bool() #np.tile(mask[np.newaxis,np.newaxis], (bs,1,1))
            observed_mask_point = observed_mask.reshape(observed_mask.shape[0],-1) #bs,h*w

        elif len(x.shape) == 4:
            bs, ch, h, w = x.shape
            num_elements = h * w
            num_to_mask = int((missing_rate) * num_elements)
            mask = np.ones(num_elements, dtype=bool)
            indices_to_mask = np.random.choice(num_elements, num_to_mask, replace=False)
            mask[indices_to_mask] = False
            mask = mask.reshape(h,w)
            observed_mask = np.tile(mask[np.newaxis,np.newaxis], (bs,ch,1,1))
            observed_mask_point = observed_mask[:,0].reshape(observed_mask.shape[0],-1) #bs,h*w
            observed_mask = torch.from_numpy(observed_mask)
            observed_mask_point = torch.from_numpy(observed_mask_point)


        elif len(x.shape) == 5: #chairs
            observed_mask, observed_mask_point = create_fixed_mask_missingness(x.shape) 
            # bs, ch, h, w, d = x.shape
            # observed_mask_0 = (np.random.rand(h,w,d)) > missing_rate
            # observed_mask = np.tile(observed_mask_0[np.newaxis,np.newaxis], (bs,ch,1,1,1))
            # observed_mask_point= None
            
        else:
            raise NotImplementedError
            
        # observed_mask_0 = (np.random.rand(h,w)) > missing_rate
        # #TODO change this it fill fail, maybe we can use cfg to get the right dims
        # observed_mask = np.tile(observed_mask_0[np.newaxis,np.newaxis], (bs,ch,1,1))
        #observed mask indices observed (including nans)

        #add plotting if True False to plot three of them side by side for i=0, x[i], observed_mask[i], nan_mask[i]
        #make them reshape [8,48] each

        # Extract data for the specified batch index
        plot = False
        if plot:
            x_copy = x.clone()
            x_copy[~observed_mask] = np.nan
            import matplotlib.pyplot as plt
            index = 0
            x_i = x[index, 0]
            observed_i = observed_mask[index, 0]
            nan_i = nan_mask[index, 0]
            x_copy_i = x_copy[index, 0]
            
            # Reshape from (384,) to (8, 48)
            x_reshaped = x_i.reshape(8, 48)
            observed_reshaped = observed_i.reshape(8, 48)
            nan_reshaped = nan_i.reshape(8, 48)
            x_copy_reshaped = x_copy_i.reshape(8, 48)
            
            # Create figure and subplots
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            
            # Plot data
            im0 = axes[0].imshow(x_reshaped, cmap='viridis')
            im1 = axes[1].imshow(observed_reshaped, cmap='gray')
            im2 = axes[2].imshow(nan_reshaped, cmap='gray')
            im3 = axes[3].imshow(x_copy_reshaped, cmap='viridis')
            
            # Add titles
            axes[0].set_title(f'Input Data (batch {index})')
            axes[1].set_title(f'Observed Mask (batch {index})')
            axes[2].set_title(f'NaN Mask (batch {index})')
            axes[3].set_title(f'Input Data with Missingness (batch {index})')
            
            # Add colorbars
            plt.colorbar(im0, ax=axes[0])
            plt.colorbar(im1, ax=axes[1])
            plt.colorbar(im2, ax=axes[2])
            plt.colorbar(im3, ax=axes[3])
            
            # Improve layout
            plt.tight_layout()

        x = x.to(device)
        observed_mask = observed_mask.to(device)
        observed_mask_point = observed_mask_point.to(device)
        return x, observed_mask, observed_mask_point

def create_mask_missingness_frc(x, missingness, dist_type='uniform'):
        """
        :param x: input tensor of batch
        :param missingness: missingness value float
        :return: x with missingness applied
        :return: observed mask of non-missingness
        """        

        if cfg.dataset.missing_type == 'list':
            possible_values = [96,192,336,720]
            #filter values that are greater than missingness
            if cfg.model.direction_scheduler== 'max_to_min':
                possible_values = [x for x in possible_values if x>=(missingness)]
            if cfg.model.direction_scheduler== 'min_to_max':
                possible_values = [x for x in possible_values if x<=(missingness)]
            missing_rate = np.random.choice(possible_values)
        else: #not implemented
            raise NotImplementedError

        if len(x.shape) == 3: #shapenet
            # raise NotImplementedError
            bs,ch, h = x.shape
            num_elements = h
            mask = np.ones(num_elements, dtype=bool)
            mask[missing_rate:] = False
            observed_mask = np.tile(mask[np.newaxis,np.newaxis], (bs,ch,1))
            observed_mask_point = observed_mask[:,0].reshape(observed_mask.shape[0],-1) #bs,h*w

        elif len(x.shape) == 4:
            bs, ch, h, w = x.shape
            num_elements = h * w
            mask = np.ones(num_elements, dtype=bool)
            mask[missing_rate:] = False
            mask = mask.reshape(h,w)
            observed_mask = np.tile(mask[np.newaxis,np.newaxis], (bs,ch,1,1))
            observed_mask_point = observed_mask[:,0].reshape(observed_mask.shape[0],-1) #bs,h*w

        elif len(x.shape) == 5: #chairs
            observed_mask, observed_mask_point = create_fixed_mask_missingness(x.shape) 
            
        else:
            raise NotImplementedError
        
        return x, observed_mask, observed_mask_point

def mask_to_input(self, input:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:
        '''
        Args:
            input (torch.Tensor): Shape (batch_size, num_points, coordinate_dim or channel_dim).
            coor_mask (torch.Tensor): Shape (batch_size, num_points).
        Returns:
            missing_input (torch.Tensor): Shape (batch_size, num_points_not_masked,coordinate_dim or channel_dim)
        '''
        missing_input = input[mask,:].reshape(input.shape[0],-1,input.shape[2])
        return missing_input


def compute_occlusion_mask(input_size, task_type: str, occlusion_type:str, occlusion_size: int, perm=None):
    """
    Gives observed mask to be used on input data.
    Args:
        input_size (tuple): Size of the input image, WxH.
        task_type (str): Type of the task "imputation" or "forecasting".
        occlusion_type (str): Can be D or T or DT / dimension time or both
        occlusion_size (tuple):  Starting index, end index, Size of the occlusion.
    Returns:
        mask (torch.Tensor): Mask of shape (*input_size).
    """


    # w,h = input_size
    if occlusion_type is None:
        occlusion_mask = torch.ones(*input_size,dtype=bool) #bogus
        return occlusion_mask

    index_start, index_end, size = occlusion_size
    size = int(size)
    number_of_axis = len(input_size)
    occlusion_mask = torch.zeros(*input_size,dtype=bool)

    if occlusion_type == "DT":
        #input must be 1D flattened
        assert number_of_axis==1, "DT only works for 1D"
        indices = np.random.choice(index_end, size, replace=False)
        indices  = indices + index_start
        occlusion_mask[indices] = True

    elif occlusion_type == "D":
        #input must be 2D NOT flattened
        assert number_of_axis==2, "D only works for 2D"
        n_channels =  input_size[0]
        indices = np.random.choice(index_end, size, replace=False)
        occlusion_mask[indices] = True
    
    elif occlusion_type == "T":
        #input must be 2D NOT flattened
        assert number_of_axis==2, "T only works for 2D"
        n_time =  input_size[1]
        indices = np.random.choice(index_end, size, replace=False)
        # time_to_occlude = np.array([0])
        occlusion_mask[:,indices] = True
    else:
        raise NotImplementedError


    return occlusion_mask.flatten()

def apply_occlusion_mask(coordinates:torch.Tensor, features:torch.Tensor, mask: torch.Tensor):
    '''
    Args:
        coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim)
        features (torch.Tensor): Shape (batch_size, num_points, channel_dim)
        mask (torch.Tensor): Shape (*dim).
    Returns:
        coordinates (torch.Tensor): Shape (batch_size, num_points_not_masked, coordinate_dim).
        features (torch.Tensor): Shape (batch_size, num_points_not_masked, channel_dim).
    '''
    
    coors_masked = coordinates[:, mask.flatten(), :] # [bs, num_points_not_masked, coordinate_dim]
    features_masked = features[:, mask.flatten(), :] # [bs, num_points_not_masked, channel_dim]

    return coors_masked, features_masked

#NOT USED
def compute_mask_mar(batch, is_training):
    assert cfg.dataset.missing_perc>0
    bs = batch.shape[0]
    if is_training:
        if cfg.dataset.name in ["shapenet"]:
            mask = batch[0,:,[0]].expand(bs,-1,-1) #torch.Size([8, 6000, 1])
            mask_point = mask[:,:,0].reshape(mask.shape[0],-1) #bs,h*w
        elif cfg.dataset.name in ["voxels"]:
            mask = batch[0].expand(bs,-1,-1,-1,-1) #[bs,1,32,32,32]
            mask_point = mask[:,0].reshape(mask.shape[0],-1) #bs,h*w
        else:
            mask = batch[0].expand(bs,-1,-1,-1) #[bs,ch,h,w]
            mask_point = mask[:,0].reshape(mask.shape[0],-1) #bs,h*w
    else:
        if cfg.dataset.name in ["shapenet"]:
            mask = torch.ones_like(batch[0,:,[0]].expand(bs,-1,-1))
            mask_point = mask[:,:,0].reshape(mask.shape[0],-1) #bs,h*w
        elif cfg.dataset.name in ["voxels"]:
            mask = torch.ones_like(batch[0].expand(bs,-1,-1,-1,-1))#torch.Size([4, 1, 4096])
            mask_point = mask[:,0].reshape(mask.shape[0],-1) #bs,h*w
        else:
            mask = torch.ones_like(batch[0]).expand(bs,-1,-1,-1) #[bs,ch,h,w]
            mask_point = mask[:,0].reshape(mask.shape[0],-1) #bs,h*w
    return mask, mask_point


def bbox2mask(self, bbox):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [B, 1, H, W]
        """
        def npmask(bbox, ch, height, width, delta_h, delta_w):
            mask = np.zeros((1, ch, height, width), np.float32)
            # h = np.random.randint(delta_h//2+1)
            # w = np.random.randint(delta_w//2+1)
            h=delta_h
            w=delta_w
            mask[:, :, bbox[0] : bbox[0]+bbox[2],
                 bbox[1] : bbox[1]+bbox[3]] = 1.
            return mask

        img_shape =cfg.dataset.dims
        height = img_shape[1]
        width = img_shape[2]

        mask = npmask(bbox, 1, height, width, 5, 5)
        
        return torch.FloatTensor(mask)
        
def compute_neighbors(bs,K,res,pi):
                #res is the bigger one
                #res_org = (res+1)//2
                # bs = x_rec.shape[0]
                # K = x_rec.shape[-1]
                # res = x_rec.shape[-2]
                pi = pi.permute(0,2,1).reshape(bs,K,res,res)
                conv2d = torch.nn.Conv2d(in_channels=K, out_channels=K, kernel_size=3, stride=2, bias=False,groups=K)
                weight = torch.zeros((K, 1, 3, 3),dtype=torch.float).to(cfg.device)
                # print(weight)
                weight[:,:,0,0]=1
                weight[:,:,0,-1]=1
                weight[:,:,-1,0]=1
                weight[:,:,-1,-1]=1
                # weight.requires_grad=False
                # print(weight)
                conv2d.weight = torch.nn.Parameter(weight)
                conv2d.weight.requires_grad=False
                a = conv2d(pi).detach()
                centers = np.arange(1,res,2)
                pi2 = impute_findings(a,pi,centers)

                conv2d2 = torch.nn.Conv2d(in_channels=K, out_channels=K, kernel_size=3, stride=1, bias=False, groups=K, padding=1)
                weight = torch.zeros((K, 1, 3, 3),dtype=torch.float).to(cfg.device)
                # print(weight)
                weight[:,:,0,1]=1
                weight[:,:,1,0]=1
                weight[:,:,1,-1]=1
                weight[:,:,-1,1]=1
                # weight.requires_grad=False
                # print(weight)
                conv2d2.weight = torch.nn.Parameter(weight)
                conv2d2.weight.requires_grad=False
                b = conv2d2(pi2).detach()
                centers = np.arange(1,res,2)
                # print(centers)
                centers2 = np.arange(0,res,2)
                # print(centers2)
                pi3 = impute_findings2(b,pi2,centers,centers2)
                return pi3


def impute_findings(source,target,centers):
    for x in centers:
        for y in centers:
            # print(x,y)
            # print((x-1)//2,(y-1)//2)
            target[:,:,x,y] = source[:,:,(x-1)//2,(y-1)//2]/4
    return target

def impute_findings2(source,target,centers,centers2):
    for x in centers:
        for y in centers2:
            # print(x,y)
            if y==0 or y==centers2[-1]:
                dividend = 3
            else:
                dividend = 4    
            target[:,:,x,y] = source[:,:,x,y]/dividend
            target[:,:,y,x] = source[:,:,y,x]/dividend
    return target

def neighborhood_filling(centers, prior_imputed_1:torch.Tensor, scale_pixels:int, kernel_size:int=3):
    #prior_imputed_1: (bs,all,K)
    #prior_imputed_1 = reshape
    kernel = np.zeros((scale_pixels+1,scale_pixels+1))
    kernel[0,0]=1
    kernel[0,-1]=1
    kernel[-1,0]=1
    kernel[-1,-1]=1
    kernel = np.asarray(kernel,dtype=bool)

    for x in centers-1:
        for y in centers-1:
            image = prior_imputed_1[:,x-scale_pixels//2:x+scale_pixels//2+1,y-scale_pixels//2:y+scale_pixels//2+1]
            result = image[:,kernel]
            values, counts = np.unique(result, return_counts=True)
            ind = np.argmax(counts)
            prior_imputed_1[x,y] = values[ind]

    return prior_imputed_1

def neighborhood_filling_2(centers, prior_imputed_1:torch.Tensor, scale_pixels:int, kernel_size:int=3):
    kernel = np.zeros((scale_pixels+1,scale_pixels+1))
    kernel[0,0]=1
    kernel[0,-1]=1
    kernel[-1,0]=1
    kernel[-1,-1]=1
    kernel = np.asarray(kernel,dtype=bool)
    
    for x in centers-1:
        for y in centers-1:
            image = prior_imputed_1[x-scale_pixels//2:x+scale_pixels//2+1,y-scale_pixels//2:y+scale_pixels//2+1]
            result = image[kernel]
            values, counts = np.unique(result, return_counts=True)
            ind = np.argmax(counts)
            prior_imputed_1[x,y] = values[ind]

    return prior_imputed_1