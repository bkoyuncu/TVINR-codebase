import numpy as np
from imagegym.config import cfg
import torch

def z_normalize(X):
    raise NotImplementedError
    X_mean = X[:,:,0].mean(dim=1)
    X_std = X[:,:,0].std(dim=1)
    X_normalize = (X[:,:,0].transpose(1,0) - X_mean) / (X_std + 1e-7)

    return X_normalize.transpose(1,0).unsqueeze(-1)

def maxmin_normalize_out(X):

    # X_mean = X[:,:,0].mean(dim=1)
    # X_std = X[:,:,0].std(dim=1)
    # X_normalize = (X[:,:,0].transpose(1,0) - X_mean) / (X_std + 1e-7)

    # return X_normalize.transpose(1,0).unsqueeze(-1), X_mean, X_std

    X_max = X.amax(dim=(1,2)) #X[:,:,0].mean(dim=1)
    X_min = X.amin(dim=(1,2))
    X_normalize = (X - X_min[:,None,None]) / (X_max[:,None,None] - X_min[:,None,None] + 1e-7)
    return X_normalize, X_min, X_max

def maxmin_denormalize_out(X_norm, X_min, X_max):

    X = X_norm * (X_max[:,None,None] - X_min[:,None,None] + 1e-7) + X_min[:,None,None] 

    return X

def z_normalize_out(X):

    # Store the original device
    device = X.device
    # Move the tensor to the CPU for normalization
    X = X.cpu().numpy()

    if len(X.shape) == 4:
        if cfg.dataset.name in ["P12",'P12_new',"HAR"]:
            if cfg.dataset.spatial_norm == "channel_only":
                X_mean = np.nanmean(X, axis=(1,3))
                X_std = np.nanstd(X, axis=(1,3))
                X_normalize = (X - X_mean[:,None,:,None]) / (X_std[:,None,:,None] + 1e-7)
            elif cfg.dataset.spatial_norm == "all":
                X_mean = np.nanmean(X, axis=(1,2,3))
                X_std = np.nanstd(X, axis=(1,2,3))
                X_normalize = (X - X_mean[:,None,None,None]) / (X_std[:,None,None,None] + 1e-7)
            else:
                raise NotImplementedError
        
        elif cfg.dataset.spatial_norm == "all":
                X_mean = np.nanmean(X, axis=(1,2,3))
                X_std = np.nanstd(X, axis=(1,2,3))
                X_normalize = (X - X_mean[:,None,None,None]) / (X_std[:,None,None,None] + 1e-7)
    elif len(X.shape) == 3:
        X_mean = np.nanmean(X, axis=(1,2))
        X_std = np.nanstd(X, axis=(1,2))
        X_normalize = (X - X_mean[:,None,None]) / (X_std[:,None,None] + 1e-7)
    else:
        raise ValueError("Input tensor must be 3D or 4D")
    
    # Replace NaN values in the normalized tensor with zeros
    # X_normalize = np.nan_to_num(X_normalize, nan=0.0)

    #the elements in X_std smaller than 1e-3 are replaced by 1e-3
    # X_std = np.where(X_std < 1e-2, 1, X_std)

    # Convert back to torch tensor and move to original device
    X_normalize = torch.tensor(X_normalize).to(device)
    X_mean = torch.tensor(X_mean).to(device)
    X_std = torch.tensor(X_std).to(device)
    
    return X_normalize, X_mean, X_std

def z_denormalize_out(X_norm, X_mean, X_std):
    if len(X_norm.shape) == 4:
        if cfg.dataset.name in ["P12","P12_new", "HAR"]:
            if cfg.dataset.spatial_norm == "channel_only":
                X_out = X_norm * (X_std[:,None,:, None] + 1e-7) + X_mean[:,None,:, None]
            elif cfg.dataset.spatial_norm == "all":
                X_out = X_norm * (X_std[:,None,None,None] + 1e-7) + X_mean[:,None,None,None]
            else:
                raise NotImplementedError
        elif cfg.dataset.spatial_norm == "all":
                X_out = X_norm * (X_std[:,None,None,None] + 1e-7) + X_mean[:,None,None,None]
    elif len(X_norm.shape) == 3:
        X_out = X_norm * (X_std[:,None,None] + 1e-7) + X_mean[:,None,None]
    else:
        raise ValueError("Input normalized tensor must be 3D or 4D")
    
    # Handle potential NaN or infinite values
    # X_out = np.nan_to_num(X_out, nan=0.0, posinf=X_out.max(), neginf=X_out.min())

    return X_out

def z_normalize_other(X, X_mean, X_std):
    # raise NotImplementedError
    if len(X.shape) == 4:
        X_normalize = (X - X_mean[:,None,None,None]) / (X_std[:,None,None,None] + 1e-7)
    elif len(X.shape) == 3:
        X_normalize = (X - X_mean[:,None,None]) / (X_std[:,None,None] + 1e-7)

    return X_normalize


class MinMaxScalerFixed:
    def __init__(self, feature_range):
        assert feature_range[0] == 0
        assert feature_range[1] == 1
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.min_data = 0
        self.max_data = 1

    def fit(self, x):
        self.min_data = 0
        self.max_data = 1

    def transform(self, x):
        diff = self.max_data - self.min_data
        x_norm = (x - self.min_data) / diff  # [0,1]
        return x_norm

    def inverse_transform(self, x_norm):
        diff = self.max_data - self.min_data
        x = x_norm * diff + self.min_data
        return x

class MinMaxScaler:
    def __init__(self, feature_range, mode='window'):
        # assert feature_range[0] == 0
        # assert feature_range[1] == 1
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.min_data = None
        self.max_data = None

    def fit(self, x):
        min_, max_ = x.min(), x.max()
        if self.min_data is None:
            self.min_data = x.min()
            self.max_data = x.max()
        else:
            self.min_data = min(self.min_data, min_)
            self.max_data = max(self.max_data, max_)
    
    def fit_manual(self):
        self.min_data = 0
        self.max_data = 1

    def transform(self, x):
        diff = self.max_data - self.min_data
        x_norm = (x - self.min_data) / diff  # [0,1]
        return x_norm

    def inverse_transform(self, x_norm):
        diff = self.max_data - self.min_data
        x = x_norm * diff + self.min_data
        return x


class MinMaxScalerWindow(MinMaxScaler):
    def __init__(self, feature_range, max=None, min=None):
        super(MinMaxScalerWindow, self).__init__(feature_range=feature_range, mode='window')
        self.min = feature_range[0]
        self.max = feature_range[1]
    def transform(self, x):
        '''
        x : [N, C, T]
        '''

        # Store the original device
        device = x.device

        if self.max is None and self.min is None:
            max = x.max(dim=-1, keepdim=True)[0]
            min = x.min(dim=-1, keepdim=True)[0]
        else:
            max = torch.Tensor([self.max]).to(device)
            min = torch.Tensor([self.min]).to(device)
        diff = max - min
        if sum(diff)==0:
            diff = diff + 1e-7
        return (x - min) / diff
    def inverse_transform(self, x_norm):
        raise NotImplementedError
    
    def fit(self, x):
        pass
        # return super().fit(x)

class MinMaxScalerData(MinMaxScaler):
    def __init__(self, feature_range):
        super(MinMaxScalerData, self).__init__(feature_range=feature_range, mode='data')
    def transform(self, x):
        '''
        x : [N, C, T]
        '''
        max = x.max(dim=-1, keepdim=True)[0]
        min = x.min(dim=-1, keepdim=True)[0]
        diff = max - min
        return (x - min) / diff
    def inverse_transform(self, x_norm, x):
        '''
        x : [N, C, T]
        '''
        max = x.max(dim=-1, keepdim=True)[0]
        min = x.min(dim=-1, keepdim=True)[0]
        diff = max - min
        return x_norm * diff + min
        # raise NotImplementedError
    
class StandardScalerFixed:

    def __init__(self):
        self.mu_ = None
        self.scale_ = None
        if cfg.dataset.name in ["celeba"]:
            #torchvision
            # self.mu_ = np.array([0.485, 0.456, 0.406])
            # self.scale_ = np.array([0.229, 0.224, 0.225])
            #Precomputed from dataset
            self.mu_ = 0.43173495
            self.scale_ = 0.2837438
        if cfg.dataset.name in ["era5"]:
            #this take them to [-1,1]
            # self.mu_ = 0.6352
            self.mu_ = 0.5
            # self.scale_ = 0.1825
            self.scale_ = 0.5
            # self.data.shape

    def fit(self, x):
        self.mu_ = x.mean()
        self.scale_ = x.std()

    def transform(self, x):
        return (x - self.mu_) / self.scale_

    def inverse_transform(self, x_norm):
        return x_norm * self.scale_ + self.mu_

class StandardScaler:

    def __init__(self, mode="channel"):
        '''
        mode: channel, global
        '''
        self.mu_ = None
        self.scale_ = None
        self.mode = mode

    def fit(self, x):
        '''
        x: [N, C, H, W] or [N, C, T]
        '''
        assert len(x.shape) == 4 or len(x.shape) == 3
        if len(x.shape) == 4:
            data_type = 'image'
        else:
            data_type = 'time_series'

        if self.mode == "channel":
            #take mean except for channel
            if data_type == 'image':
                self.mu_ = x.mean((0, 2, 3)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                self.scale_ = x.std((0, 2, 3)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            elif data_type == 'time_series':
                self.mu_ = x.mean((0, 2)).unsqueeze(0).unsqueeze(2)
                self.scale_ = x.std((0, 2)).unsqueeze(0).unsqueeze(2)
        elif self.mode == "global":
            self.mu_ = x.mean()
            self.scale_ = x.std()
        elif self.mode in ["none","none_z","none_01","none_01_all","none_z_all"]:
            self.mu_ = 0
            self.scale_ = 1
        else:
            raise NotImplementedError

    def fit_with_loader(self, loader):
        mus, stds = [], []
        i = 0
        for batch in iter(loader):
            # print(f"i: {i} {batch[0].mean()}")
            i += 1
            mus.append(batch[0].mean())
            stds.append(batch[0].std())
            if i == 10: break

        self.mu_ = np.mean(mus)
        self.scale_ = np.mean(stds)

    def transform(self, x):
        return (x - self.mu_) / self.scale_

    def inverse_transform(self, x_norm):
        return x_norm * self.scale_ + self.mu_


class BaseScaler:

    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, x):
        self.scaler.fit(x)

    def fit_with_loader(self, loader):
        self.scaler.fit_with_loader(loader)

    def transform(self, x):
        x_uni = self.scaler.transform(x)
        return x_uni

    def inverse_transform(self, x_norm):
        x_uni = self.scaler.inverse_transform(x_norm)
        return x_uni
    

class MyStandardScaler(BaseScaler):
    def __init__(self):
        scaler = StandardScaler()
        super(MyStandardScaler, self).__init__(scaler=scaler)

class MyStandardScalerFixed(BaseScaler):
    def __init__(self):
        scaler = StandardScalerFixed()
        super(MyStandardScalerFixed, self).__init__(scaler=scaler)


class MyMinMaxScaler(BaseScaler):
    def __init__(self, feature_range):
        scaler = MinMaxScaler(feature_range=feature_range)
        super(MyMinMaxScaler, self).__init__(scaler=scaler)

class MyMinMaxScalerFixed(BaseScaler):
    def __init__(self, feature_range):
        scaler = MinMaxScalerFixed(feature_range=feature_range)
        super(MyMinMaxScalerFixed, self).__init__(scaler=scaler)


class Temporal_Scaler():
    def __init__(self):    

        assert cfg.dataset.temporal_norm in ['none', 'window', 'global']
        self.mode = cfg.dataset.temporal_norm

    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ['HAR','P12', 'P12_new', 'electricity_nips','electricity_hourly','electricity_hourly_new','electricity_hourly', 'electricity_hourly_new', 'solar-energy-10', 'solar-energy-h', 'traffic'] and self.mode in ['global']:
            times = dataset_tr.times
            min = times.min()
            max = times.max()
            scaler = MinMaxScaler(feature_range=(min, max))
            scaler.fit(times)

        elif cfg.dataset.name in ['HAR','P12','P12_new','electricity_nips','electricity_hourly','electricity_hourly_new','electricity_hourly', 'electricity_hourly_new', 'solar-energy-10', 'solar-energy-h', 'traffic'] and self.mode in ['window']:
            if cfg.dataset.name in ['HAR','P12','P12_new','electricity_hourly_new','traffic','solar-energy-10', 'solar-energy-h']:

                max_y = dataset_tr.times[:,0,:,:].max()
                min_y = dataset_tr.times[:,0,:,:].min()

                max_x = dataset_tr.times[:,1,:,:].max()
                min_x = dataset_tr.times[:,1,:,:].min()

                scaler = [MinMaxScalerWindow(feature_range=(min_y, max_y)), MinMaxScalerWindow(feature_range=(min_x, max_x))]
            else:
                max = dataset_tr.times.max()
                min = dataset_tr.times.min()
                scaler = MinMaxScalerWindow(feature_range=(0,1), max=max, min=min)
        elif cfg.dataset.name in ['HAR','P12','P12_new','electricity_nips','electricity_hourly','electricity_hourly_new','electricity_hourly', 'electricity_hourly_new', 'solar-energy-10', 'solar-energy-h', 'traffic'] and self.mode in ['none']:
            scaler = MyMinMaxScalerFixed(feature_range=(0,1))

        return scaler
