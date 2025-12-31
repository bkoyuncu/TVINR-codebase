import pathlib
import os
import numpy as np
import pandas as pd
import torch.utils.data.dataset as torch_dataset
from torch.utils.data import DataLoader
import json
from typing import Tuple, Iterator, Literal

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
import torch
from imagegym.config import cfg

import numpy as np

import numpy as np

def split_data(all_data, all_covariates, all_lag_covariates, num_dates, prediction_length, history_length, split_name):
    
    if history_length!=0:
        prediction_length = history_length+prediction_length
    output_data = []
    output_covariates = []
    output_lag_covariates = []

    len_train = all_data.shape[1]
    length_cutoff = num_dates * prediction_length

    start_idx = len_train - length_cutoff
    end_idx = len_train - length_cutoff + (num_dates) * prediction_length

    if split_name == 'train':
        start_idx = 0
        end_idx = (num_dates) * prediction_length
        length_cutoff = (num_dates) * prediction_length


    output_data.append(all_data[:, start_idx:end_idx])
    output_covariates.append(all_covariates[:, start_idx:end_idx])
    output_lag_covariates.append(all_lag_covariates[:, start_idx:end_idx])

    print(f"{split_name} start end length", start_idx,end_idx,end_idx-start_idx)

    output_data = np.array(output_data)  # [num_validation_dates, dim, history_length + prediction_length]
    output_covariates = np.array(output_covariates)  # [num_validation_dates, dim_c, history_length + prediction_length]
    output_lag_covariates = np.array(output_lag_covariates)  # [num_validation_dates, dim_c_lag, history_length + prediction_length]

    all_data = all_data[:, :-length_cutoff] if split_name != 'train' else all_data[:, length_cutoff:]  # [dim, T - num_validation_dates * prediction_length]
    all_covariates = all_covariates[:, :-length_cutoff] if split_name != 'train' else all_covariates[:, length_cutoff:]   # [dim_c, T - num_validation_dates * prediction_length]
    all_lag_covariates = all_lag_covariates[:, :-length_cutoff] if split_name != 'train' else all_lag_covariates[:, length_cutoff:]  # [dim_c_lag, T - num_validation_dates * prediction_length]

    # assert output_data.shape[-1] == history_length + prediction_length, "Data must have the correct length."
    assert output_covariates.shape[-1] == output_data.shape[-1], "Covariates and data must have the same length."
    assert output_lag_covariates.shape[-1] == output_data.shape[-1], "Lag covariates and data must have the same length."

    return output_data, output_covariates, output_lag_covariates, all_data, all_covariates, all_lag_covariates

# Example usage:
# test_data, test_covariates, test_lag_covariates, all_data, all_covariates, all_lag_covariates = process_data(all_data, all_covariates, all_lag_covariates, num_test_dates, prediction_length, history_length)




# TODO: extend this list
SUPPORTED_DATASETS = {
    'electricity_nips', # hourly
    'solar_nips', # hourly
    'wiki2000_nips', # daily
    'electricity_hourly', # hourly
    'electricity_hourly_new', # hourly
    'traffic', # hourly
    'traffic_nips', # hourly
    'solar-energy-h', # hourly
    'solar-energy-10' 
} 
# TODO: electricity_nips has constant 0s in dim = 322, remove it from the dataset?

DATASET_TEST_DATES = {
    'electricity_nips': 7,
    'electricity_hourly': None,
    'traffic': None,
    'traffic_nips': None,
    'solar-energy-10': None,
    'wiki2000_nips': 5,
    'solar-energy-h': None
}

PREDICTION_LENGTHS = {
    'electricity_nips': 24,
    'electricity_hourly': 48,
    'solar_nips': 24,
    'wiki2000_nips': 30,
    'traffic': 24,
    'traffic_nips': 24,
    'solar-energy': 24
}





def get_gluonts_multivar_dataset(
    dataset_name: str,
    dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
    regenerate: bool = False,
    prediction_length: int = None,
    print_stats: bool = True,
    combine_dims: bool = False) -> TrainDatasets:
    """Load a dataset from GluonTS repository.

    Args:
        dataset_name (str): Name of the dataset to load.
        dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
        regenerate (bool, optional): Whether to regenerate (newly download) the dataset. Defaults to False.
        prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself.
        print_stats (bool, optional): Whether to print the dataset statistics. Defaults to True.
        
    Raises:
        ValueError: If the dataset is not supported.

    Returns:
        TrainDatasets: GluonTS dataset.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # regenerate = True
    if dataset_name == 'traffic':
        prediction_length = 10000
    # dataset_name = 'electricity_hourly'
    dataset = get_dataset(dataset_name, path=dataset_path, regenerate=regenerate, prediction_length=prediction_length)
    # train_grouper = MultivariateGrouper(train_fill_rule = np.mean , test_fill_rule = lambda x: 0.0)
    test_grouper = MultivariateGrouper(num_test_dates=DATASET_TEST_DATES[dataset_name], train_fill_rule = np.mean, test_fill_rule = lambda x: 0.0)
    # train_data = train_grouper(dataset.train)
    test_data = test_grouper(dataset.test)

    dataset = TrainDatasets(metadata=dataset.metadata, train=test_data) #aim is to use all and then split the test
    
    print(f"Dataset '{dataset_name}' loaded from GluonTS.")
    return dataset


class GluonTSDataset(torch_dataset.Dataset):
    def __init__(
        self,
        dataset_name: str,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()),
        num_splits: list = [0, 0, 0],
        # num_validation_dates: int = 0,
        # num_test_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
        total_length: int = 0,
        version:int = 0
    ):
        """
        Args:
            dataset_name (str): Name of the dataset to load.
            kind (Literal['train', 'val', 'test'], optional): Train, Validation or Test set. Defaults to 'train'.
            dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
            num_validation_dates (int, optional): Number of dates to use for validation. Defaults to 0. If 0 or None, no validation set is created.
            regenerate (bool, optional): Whether to regenerate (newly download & create) the dataset. Defaults to False.
            prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself, which is set if None.
            history_length (int, optional): Conditioning length when predictiing. If None, it is set to the prediction_length. Defaults to None.
            window_offset (int, optional): Offset for the windowing of the dataset when batching. If None, it is set to the prediction_length + history_length, such that no sample overlaps. Defaults to None.
                The number of samples in the dataset will be (T - history_len - pred_len) // window_offset, where T is the length of the full training sequence.
            random_offset (bool, optional): Whether to randomly offset the windowing when __getitem__ is called. If true, the window_offset is randomly chosen between 0 and window_offset-1. Defaults to False.
        """
        assert kind in ['train', 'val', 'test'], "Kind must be 'train', 'val' or 'test'."
        assert window_offset is None or window_offset > 0, "Window offset must be positive."
        # if prediction_length is not None:
            # assert PREDICTION_LENGTHS[dataset_name] == prediction_length, f"Prediction length does not match the expected prediction length ({prediction_length} != {PREDICTION_LENGTHS[dataset_name]})."
        self.version = version
        self.num_splits = num_splits

        self.dataset_path = dataset_path
        self.dataset_name_original = dataset_name
        self.dataset_name = dataset_name

        if dataset_name == 'electricity_hourly_new':
            self.dataset_name_original = dataset_name
            self.dataset_name = 'electricity_hourly'

        self.kind = kind
        num_train_dates, num_validation_dates, num_test_dates = num_splits

        self.prediction_length = prediction_length
        self.history_length = history_length if history_length is not None else prediction_length
        self.window_offset = window_offset
        self.random_offset = random_offset
        self.dataset_extended_name = f"{dataset_name}_tmp_{prediction_length}_tmh_{history_length}_split_{num_train_dates}_{num_validation_dates}_{num_test_dates}"
        
        # sets prediction_length and history_length if not already set
        self._download_and_create_dataset( dataset_path=dataset_path, regenerate=regenerate, prediction_length=prediction_length, history_length=history_length, num_splits=num_splits, total_length=total_length)        

        assert self.prediction_length is not None, "Prediction length must be set."
        assert self.history_length is not None, "History length must be set."
        
        if random_offset and self.kind != 'train':
            print("Warning: random_offset is ignored for the test and validation sets.")
        
        if window_offset is None and self.kind == 'train':
            self.window_offset = self.prediction_length + self.history_length
        elif window_offset is not None and self.kind == 'train':
            self.window_offset = min(window_offset, self.prediction_length + self.history_length)
        elif window_offset is None and self.kind != 'train':
            self.window_offset = self.prediction_length + self.history_length
        
        self.data_scaler = GluonTSSequenceScaler()
        self.covariates_scaler = GluonTSSequenceScaler()
        
        self.data_all, self.covariates, self.lag_covariates = self._get_data(kind=self.kind) # [dim, #n, T], [dim_c, #n,  T], [dim_c_lag, #n,  T] (train) | [test_size, dim, history_length + prediction_length], [test_size, dim_c, history_length + prediction_length] [test_size, dim_c_lag, history_length + prediction_length] (test)
        
        self.dim_size, self.n_signals, _ = self.data_all.shape
        
        self.other_versions = self._get_other_versions()


    def _get_other_versions(self):
    
        if self.kind == 'train':
            other_splits = [i for i in np.arange(self.num_splits[0])]
        elif self.kind == 'val':
            other_splits = [i for i in np.arange(self.num_splits[1])]
        elif self.kind == 'test':
            other_splits = [i for i in np.arange(self.num_splits[2])]
        else:
            raise NotImplementedError
        
        # if self.version != -1:
            # other_splits.remove(self.version)

        return_list = [other_splits+[-1]]

        return return_list

    def _create_permutations(self, shape: Tuple[int, int, int, int], seed:int ) -> np.ndarray:
        """
        Create permutations for the dataset.

        Args:
            shape (Tuple[int, int, int]): Shape of the dataset.

        Returns:
            np.ndarray: Permutations for the dataset.
        """
        V_, dim, N_, t = shape
        permutations = np.zeros((V_, 1, N_, t))
        #fix seed 
        np.random.seed(seed)
        for v in range(V_):
            for n in range(N_):
                permutations[v,0,n] = np.random.permutation(t)
        
        #make sure data type is int
        permutations = permutations.astype(int)
        print("Permutations created.")
        print(permutations[0,0,:5])
        return permutations
        
    def _select_version(self, select_split: int=-1, select_user: int= -1):

        '''
        input shapes:
            data_all: (V=50, 1, N_split=321, 200)

        data_out shape: (N_signals, 1, D, T)
        times_out shape: (N_signals, 2, D, T)
        permutation_indx_out shape: (N_signals, 1, T)
        user_ids: (N_signals, 1)
        '''

        if select_split == -1:
            select_split = [i for i in range(self.data_all.shape[0])]
        else:
            if isinstance(select_split, (int, np.integer)):
                select_split = [select_split]
            if isinstance(select_split, list):
                select_split = select_split
            
        if select_user == -1:
            select_user = [i for i in range(self.data_all.shape[2])]
        else:
            if isinstance(select_split, (int, np.integer)):
                select_user = [select_user]
            if isinstance(select_user, list):
                select_user = select_user

        
        final_shape = self.total_length if self.kind == 'train' and self.model_mode =='forecasting' else self.history_length + self.prediction_length
        fake_channel = 1
        self.data = self.data_all[select_split, :, :, :][:, :, select_user, :].reshape(-1, fake_channel, self.dim_size, final_shape )
        self.length = self.data.shape[0]
        self.permutation_indx = self.permutation_indx_main[select_split, :, :, :][:, :, select_user, :].reshape(-1, 1, self.data.shape[-1])
        self.times = self.times_all.repeat(self.data.shape[0],1,1,1)
        self.user_ids = self.user_ids_all[select_user].unsqueeze(0).repeat(len(select_split),1,1).reshape(-1,1)
        
        print(f"Data shape: {self.data.shape}")
        print(f"Times shape: {self.times.shape}")
        # print(f"Targets shape: {self.targets.shape}")
        print(f"User IDs shape: {self.user_ids.shape}")
        print(f"Permutation shape: {self.permutation_indx.shape}")

        

    def create_times(self):
        times_range = torch.arange(0, self.history_length + self.prediction_length, dtype=torch.float32) * 1 # create times vector with range in Hz
        times_range_dims = torch.arange(0,self.dim_size, dtype=torch.float32)#* dim.size()[2] # create times vector with range in Hz
        if self.dim_size == 1: 
            times_range_dims = times_range_dims + 0.5
        mesh_x, mesh_y = torch.meshgrid(times_range, times_range_dims, indexing='ij')
        # Stack the two mesh grids along a new dimension to create a 2D array
        mesh_grid = torch.stack((mesh_y, mesh_x), dim=2) # torch.Size([128, 3, 2])
        
        return mesh_grid.repeat(self.data_all.shape[0],1,1,1).permute(0,3,2,1) # create timestamps tensor (N, 2, W, T)

    def create_user_ids(self):
         #(50, 1, 321, 200)
         #returns (321,1)
        return torch.arange(0, self.data_all.shape[2], dtype=torch.float32).unsqueeze(1)
        
    
    def _slice_data(self, data: np.ndarray, history_length: int, prediction_length: int, window_offset: int, random_offset: bool) -> np.ndarray:
        """
        Slice the data into sequences of length (history_length + prediction_length) with a window offset.

        Args:
            data (np.ndarray): [dim, T] Data to slice.
            history_length (int): History length.
            prediction_length (int): Prediction length.
            window_offset (int): Window offset.
            random_offset (bool): Whether to randomly offset the windowing.

        Returns:
            np.ndarray: [num_samples, dim, history_length + prediction_length] Sliced data.
        """
        
        offset = 0
        if data.shape[1] % window_offset == 0:
            num_samples = (data.shape[-1] - history_length - prediction_length) // window_offset #+ 1
        else:
            num_samples = (data.shape[-1] - history_length - prediction_length) // window_offset + 1
        
        
        sliced_data = np.zeros((num_samples, *data.shape[:-1], history_length + prediction_length))
        for i in range(num_samples):
            offset = 0

            start = i * window_offset + offset
            end = start + history_length + prediction_length
            sliced_data[i] = data[..., start:end]
        
        return sliced_data
    
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):

        ts, times_out = self.data[idx], self.times[idx]
        target_out = 1.0

        #to torch
        ts = torch.tensor(ts, dtype=torch.float32)
        times_out = torch.tensor(times_out, dtype=torch.float32)

        #ts to float
        ts = ts.float()

        #normalize ts standardize
        if cfg.dataset.spatial_norm=="none_z_all":
            ts = (ts - ts.mean()) / (ts.std() + 1e-6)
        elif cfg.dataset.spatial_norm=="none_01_all":
            ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        else:
            pass


        return ts,  torch.unsqueeze(times_out, 0), target_out



        T = self.history_length + self.prediction_length
        T = np.array([T])
        T0 = np.array([self.history_length])
        
        if self.kind != 'train':
            x, c, c_lag =  self.data[idx].T, self.covariates[idx].T, self.lag_covariates[idx].T # transpose to match our framework dimensions [T, dim], [T, dim_c] [T, dim_c_lag]
        
        else:
            offset = 0
            if self.random_offset:
                offset = np.random.randint(self.window_offset) # in [0, window_offset - 1]
            if idx == self.length - 1:
                offset += np.random.randint(self.final_offset + 1) # in [0, final_offset]
            new_idx = idx * self.window_offset + offset
            
            x = self.data[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim]
            c = self.covariates[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim_c]
            c_lag = self.lag_covariates[:, new_idx:new_idx + self.history_length + self.prediction_length].T # [T, dim_c_lag]
          
        c = np.concatenate([c, c_lag], axis=1) # [T, dim_c + dim_c_lag]  
        return x, c, T0.astype(np.int32), T.astype(np.int32)
    
    
    def create_pandas_evaluation_iterator(self, kind: Literal['train', 'val', 'test'] = 'test') -> Iterator[pd.DataFrame]:
        """
        Create an iterator for evaluation using gluonts.evaluation.MultivariateEvaluator.
        
        Args:
            kind (Literal['train', 'val', 'test']): Train, Validation or Test set. Defaults to 'test'.

        Returns:
            Iterator: Iterator for evaluation. Note that the data is not normalized.
            List[pd.Period]: List of start dates for the forecast windows in same order as the iterator.
        """       
        print(f"Creating pandas iterator for evaluation of '{self.dataset_name}'.")
        if kind == 'val':
            raise NotImplementedError("Validation set is not supported for evaluation via gluonts.evaluation.MultivariateEvaluator as it is not part of the main functionality.")
        dataset = get_gluonts_multivar_dataset(
            self.dataset_name,
            dataset_path = self.dataset_path,
            prediction_length = self.prediction_length,
            print_stats = False
        )
        test_data = dataset.train if kind == 'train' else dataset.test
        data_list = []
        start_dates = []
        for entry in test_data:
            origianl_length = entry["target"].shape[1]
            truncated_data = entry["target"][:, - (self.history_length + self.prediction_length):] # [dim, history_length + prediction_length]
            new_start = entry["start"] + origianl_length - truncated_data.shape[1]
            data_list.append(
                pd.DataFrame(
                    data=truncated_data.T, # [history_length + prediction_length, dim]
                    index=pd.date_range(start=new_start.to_timestamp(), periods=truncated_data.shape[1], freq=new_start.freqstr).to_period(),
                ) 
            )
            start_dates.append(new_start + self.history_length)
        return iter(data_list), start_dates
        
       
    def _get_data(self, kind: Literal['train', 'test', 'val']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data and covariates from the dataset and scale them.
        
        Args:
            kind (Literal['train', 'test', 'val']): Train, Test or Validation set.
        """
        loaded_data = np.load(self.dataset_path / self.dataset_name_original / self.dataset_extended_name / kind /f"{kind}_split.npy", allow_pickle=True).item()
        data = loaded_data["data"] # [dim, T] | [test_size, dim, history_length + prediction_length]
        covariates = loaded_data["covariates"] # [dim_c, T] | [test_size, dim_c, history_length + prediction_length]
        lag_covariates = loaded_data["lag_covariates"] # [dim_c_lag, T] | [test_size, dim_c_lag, history_length + prediction_length]
            
        return data.astype(np.float32), covariates.astype(np.float32), lag_covariates.astype(np.float32)
    
    
    def _check_exist_gluonts_dataset(self):
        if (os.path.exists(self.dataset_path / self.dataset_name)
            and os.path.exists(self.dataset_path / self.dataset_name / "metadata.json")):
            return True
        else:
            return False
        
    def _check_source_dataset(self):
        if (os.path.exists(self.dataset_path / self.dataset_name)
            and os.path.exists(self.dataset_path / self.dataset_name / 'train' / "data.json.gz")
            and os.path.exists(self.dataset_path / self.dataset_name / 'test' / "data.json.gz")):
            return True
        else:
            return False   
     
    def _check_exist_dataset(self):
        if (os.path.exists(self.dataset_path / self.dataset_name)
            and os.path.exists(self.dataset_path / self.dataset_name / self.dataset_extended_name / "train_split.npy")
            and os.path.exists(self.dataset_path / self.dataset_name / self.dataset_extended_name / "test_split.npy")):
            return True
        else:
            return False
        

    def _create_covariates(self, length: int, start: pd.Period) -> np.ndarray:
        """Create covariates for the dataset.

        Args:
            length (int): Length of the sequence
            start (pd.Period): Start date of the sequence with the sample frequency.

        Returns:
            np.ndarray: [dim_c, length] Covariates for the sequence.
        """
        
        # TODO: Gluonts uses sine and cosine functions for the time covariates.
        if start.freqstr == 'H':
            covariates = [[timestamp.month, timestamp.day, timestamp.hour] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='H')]
            covariates = np.array(covariates).T # [3, length]
        elif start.freqstr == 'D':
            covariates = [[timestamp.month, timestamp.day] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='D')]
            covariates = np.array(covariates).T # [2, length]
        elif start.freqstr == '10T':
            covariates = [[timestamp.month, timestamp.day, timestamp.hour, timestamp.minute] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='10T')]
            covariates = np.array(covariates).T
        else: 
            raise NotImplementedError("Only hourly frequency is supported.")
        return covariates
    
    
    def _create_lag_covariates(self, length: int, start: pd.Period) -> np.ndarray:
        """Create lag covariates for the dataset.

        Args:
            length (int): Length of the sequence
            start (pd.Period): Start date of the sequence with the sample frequency.

        Returns:
            np.ndarray: [dim_c_lag, length] Covariates for the sequence.
        """
        if start.freqstr == 'H':
            lag_covariates = [[168, 24, 1] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='H')]
            lag_covariates = np.array(lag_covariates).T # [3, length]
        elif start.freqstr == 'D':
            lag_covariates = [[7, 1] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='D')]
            lag_covariates = np.array(lag_covariates).T # [2, length]
        elif start.freqstr == '10T':
            lag_covariates = [[168*6, 144, 6, 1] for timestamp in pd.date_range(start=start.to_timestamp(), periods=length, freq='10T')]
            lag_covariates = np.array(lag_covariates).T # [4, length]
        else:
            raise NotImplementedError("Only hourly frequency is supported.")
        return lag_covariates
        
            
    def _download_and_create_dataset(
        self,
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()),
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        create_val_set: bool = False,
        num_splits: list = [0, 0, 0],
        add_ind_to_covariates: bool = False,
        total_length: int = 0
    ):
        """
        This function downloads a GluonTS dataset and creates a PyTorch dataset as numpy arrays from a GluonTS dataset, and stores it in the dataset folder.
        The dataset is stored in the dataset_path/dataset_name folder with train_split.npy and test_split.npy files.
        
        Args:
            dataset_path (pathlib.Path, optional): Path where to save the downloaded dataset. Defaults to pathlib.Path(os.getcwd())/"data_gluonts".
            regenerate (bool, optional): Whether to regenerate (newly download & create) the dataset. Defaults to False.
            prediction_length (int, optional): Prediction length of the dataset. Defaults to None. Is usually given by the dataset itself, which is set if None.
            history_length (int, optional): Conditioning length when predictiing. If None, it is set to the prediction_length. Defaults to None.
        """

        # need to download/load the dataset from GluonTS and create the dataset for PyTorch
        if (regenerate or (not self._check_source_dataset())) and self.dataset_name != 'electricity_hourly_new':
            print(f"Creating dataset '{self.dataset_name}' from GluonTS repository.")
            dataset = get_gluonts_multivar_dataset(self.dataset_name, dataset_path=dataset_path, regenerate=True, prediction_length=total_length)

        #check if they are alrady there
        #TODO change this
        if not os.path.exists(dataset_path / self.dataset_name_original / self.dataset_extended_name / 'train') or not os.path.exists(dataset_path / self.dataset_name_original / self.dataset_extended_name / 'test') or not os.path.exists(dataset_path / self.dataset_name_original / self.dataset_extended_name / 'val'):
            dataset = get_gluonts_multivar_dataset(self.dataset_name, dataset_path=dataset_path, regenerate=False, prediction_length=None)

            if self.dataset_name_original == 'electricity_hourly_new':
                print("Dataset is electricity_hourly_new")
                data_OW = np.load(dataset_path / self.dataset_name_original / 'electricity_hourly.npy')
                dataset.train[0]['target'] = data_OW[...,0]
                self.dataset_name = self.dataset_name_original



            assert prediction_length is not None, "Prediction length must be set."
            assert history_length is not None, "History length must be set."
            
            self.num_train_dates, self.num_validation_dates, self.num_test_dates = num_splits 

            train_data = []
            train_covariates = []
            train_lag_covariates = []
            val_data = []
            val_covariates = []
            val_lag_covariates = []
            test_data = []
            test_covariates = []
            test_lag_covariates = []
            
            for i, entry in enumerate(dataset.train):
                if i == 0:
                    data = entry["target"] # [dim, T]         
                    covariates = self._create_covariates(data.shape[1], entry["start"])
                    lag_covariates = self._create_lag_covariates(data.shape[1], entry["start"])
                    assert covariates.shape[1] == data.shape[1], "Covariates and data must have the same length."
                    assert lag_covariates.shape[1] == data.shape[1], "Lag covariates and data must have the same length."
                    all_data = data
                    all_covariates = covariates
                    all_lag_covariates = lag_covariates
                else:
                    raise ValueError("Expecting only one entry in the train dataset.")
                

            if self.num_test_dates not in [0, None]:
                test_data, test_covariates, test_lag_covariates, all_data, all_covariates, all_lag_covariates =split_data(all_data, all_covariates, all_lag_covariates, self.num_test_dates, prediction_length, history_length, 'test')
            else:
                test_data = np.array([])
                test_covariates = np.array([])
                test_lag_covariates = np.array([])

            if self.num_validation_dates not in [0, None]:
                val_data, val_covariates, val_lag_covariates, all_data, all_covariates, all_lag_covariates = split_data(all_data, all_covariates, all_lag_covariates, self.num_validation_dates, prediction_length, history_length, 'val')
            else:
                val_data = np.array([])
                val_covariates = np.array([])
                val_lag_covariates = np.array([])
            
            if self.num_train_dates not in [0, None]:
                train_data, train_covariates, train_lag_covariates, all_data, all_covariates, all_lag_covariates = split_data(all_data, all_covariates, all_lag_covariates, self.num_train_dates, total_length, history_length, 'train')

            # save the statistics in the dataset folder as json
            paths = [dataset_path / self.dataset_name / self.dataset_extended_name, dataset_path / self.dataset_name / self.dataset_extended_name / 'train', dataset_path / self.dataset_name / self.dataset_extended_name / 'test', dataset_path / self.dataset_name / self.dataset_extended_name / 'val']
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p, exist_ok=True)

            
            with open(dataset_path / self.dataset_name / self.dataset_extended_name/"stats.json", "w") as f:
                json.dump({
                    "prediction_length": self.prediction_length, 
                    "history_length": self.history_length,
                    "data_dim": train_data.shape[0],
                    "train_data_length": train_data.shape[-1],
                    "val_data_length": val_data.shape[-1],
                    "test_data_length": test_data.shape[-1],
                    "covariates_dim": train_covariates.shape[0], 
                    "lag_covariates_dim": train_lag_covariates.shape[0],
                    "split_dataset": num_splits,
                    }, f)
                
            print(f"Dataset '{self.dataset_name}' created and saved in '{dataset_path}'.")
            print(f"Train data shape: {train_data.shape}, Train covariates shape: {train_covariates.shape}, Train lag covariates shape: {train_lag_covariates.shape}.")
            if self.num_validation_dates not in [0, None]:
                print(f"Validation data shape: {val_data.shape}, Validation covariates shape: {val_covariates.shape}, Validation lag covariates shape: {val_lag_covariates.shape}.")
            else:
                print("No validation set created.")
            print(f"Test data shape: {test_data.shape}, Test covariates shape: {test_covariates.shape}, Test lag covariates shape: {test_lag_covariates.shape}.")
                
            # save the dataset as numpy arrays
            train_data = {"data": train_data, "covariates": train_covariates, "lag_covariates": train_lag_covariates}
            test_data = {"data": test_data, "covariates": test_covariates, "lag_covariates": test_lag_covariates}
            # val_data = {"data": val_data, "covariates": val_covariates, "lag_covariates": val_lag_covariates}

            np.save(dataset_path / self.dataset_name / self.dataset_extended_name/ 'train'/ "train_split.npy", train_data)
            np.save(dataset_path / self.dataset_name / self.dataset_extended_name/ 'test'/ "test_split.npy", test_data)
            # np.save(dataset_path / self.dataset_name / self.dataset_extended_name/"val_split.npy", val_data)
            
            if self.num_validation_dates not in [0, None]:
                val_data = {"data": val_data, "covariates": val_covariates, "lag_covariates": val_lag_covariates}
                np.save(dataset_path / self.dataset_name / self.dataset_extended_name / 'val'/ "val_split.npy", val_data)
            
        else:
            # pass
            with open(dataset_path / self.dataset_name_original / self.dataset_extended_name / "stats.json") as f:
                stats = json.load(f)
                assert stats["prediction_length"] == self.prediction_length
                assert stats["history_length"] == self.history_length
                assert stats["split_dataset"] == num_splits    
        
class ElectricityNIPS(GluonTSDataset):
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(ElectricityNIPS, self).__init__(dataset_name="electricity_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates, 
                                              regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                              window_offset=window_offset, random_offset=random_offset)

    
class SolarNIPS(GluonTSDataset): 
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(SolarNIPS, self).__init__(dataset_name="solar_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates,
                                        regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                        window_offset=window_offset, random_offset=random_offset)
        
        
class Wiki2000NIPS(GluonTSDataset):
    def __init__(
        self,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
    ): 
        super(Wiki2000NIPS, self).__init__(dataset_name="wiki2000_nips", kind=kind, dataset_path=dataset_path, num_validation_dates=num_validation_dates,
                                           regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                           window_offset=window_offset, random_offset=random_offset)

class ElectricityBaseline (GluonTSDataset):
    def __init__(
        self,
        dataset_name: str,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        num_test_dates: int = 0,
        num_splits: list = [0,0,0],
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        total_length: int = None,
        interest_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
        draw_ratio = 0.5,
        version:int = -1,
        mode: str = 'imputation'
    ): 
        
        self.draw_ratio = draw_ratio
        # self.max_length = prediction_length + history_length
        self.full_length = False
        self.z = None
        self.model_type = cfg.model.type
        self.interest_length = interest_length
        self.total_length = total_length
        self.prediction_length = prediction_length
        self.history_length = history_length
        self.window_size = prediction_length + history_length
        self.model_mode = mode




        super(ElectricityBaseline, self).__init__(dataset_name=dataset_name, kind=kind, dataset_path=dataset_path, num_splits=num_splits,
                                              regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                              window_offset=self.total_length, random_offset=random_offset, version=version, total_length=self.total_length)

        self.data_all = self._slice_data(self.data_all, history_length= history_length, prediction_length=prediction_length, window_offset=history_length+prediction_length, random_offset=random_offset)
        self.times_all = self.create_times()[[0]]
        self.user_ids_all = self.create_user_ids()
        self.targets_all = None


        self.data_all_safe = self.data_all.copy() #data all has shape (50, 1, 321, 200)
        seeds_dict = {'train': 0, 'val': 1, 'test': 2}
        self.permutation_indx_main = self._create_permutations(self.data_all.shape, seed = seeds_dict[kind])
        self.version_len = num_splits[['train', 'val', 'test'].index(kind)]
        self._select_version(version)
        self.split = kind

        if self.model_type== 'timeflow':
            self.z = torch.zeros(np.prod(self.data_all.shape[0:3]), cfg.inr.latent_dim)
    

    def __getitem__(self, idx, full_length=False):
        """
        Args:
            idx (int): Index
        Input shapes:
            self.data.shape: (16050, 1, 1, 200)
            self.times.shape: (16050, 2, 1, 200)
            self.user_ids.shape: (16050, 1)
        Returns:
            ts.shape: (1, 3, 128)
            times_out.shape: (2, 3, 128)
            perm_idx.shape: (1, 128)
            user_ids_out.shape: (1,)

        ts shape is [1, dim, T]
        times_out shape is [2, 1, T]
        perm shape is [1,  T]
        """

        #TODO we need to write a mask to get indices with correct dims in case the signal is multi channel

        ts, times_out, user_ids_out  = self.data[idx], self.times[idx], self.user_ids[idx]
        z = self.z[idx] if self.z is not None else None
        perm_idx = self.permutation_indx[idx]
        user_ids_out = user_ids_out.reshape(-1)
        
        if self.full_length or self.draw_ratio == 1.0:
            ts = ts
            times_out = times_out
            perm_idx = perm_idx
        else:
            perm_idx = perm_idx[:, :int(self.draw_ratio*ts.shape[-1])]
            ts = ts[:, :, perm_idx[0]]
            times_out = times_out[:, :, perm_idx[0]]
        
        #to torch
        ts = torch.tensor(ts, dtype=torch.float32)
        times_out = torch.tensor(times_out, dtype=torch.float32)

        #ts to float
        ts = ts.float()

        #normalize ts standardize
        if cfg.dataset.spatial_norm=="none_z_all":
            ts = (ts - ts.mean()) / (ts.std() + 1e-6)
        elif cfg.dataset.spatial_norm=="none_01_all":
            ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        else:
            pass
        
        if self.model_type == 'timeflow': #no need for second time axis only linear
            return ts, torch.Tensor(0), times_out[[1]], torch.Tensor(0), z, perm_idx, torch.Tensor(0), torch.Tensor(0), torch.Tensor(0), torch.Tensor(0)
        else:
            return ts, torch.Tensor(0), times_out, torch.Tensor(0), torch.Tensor(0), perm_idx, torch.Tensor(0), user_ids_out, torch.Tensor(0), torch.Tensor(0)
        

class ElectricityBaselineForecast (GluonTSDataset):
    def __init__(
        self,
        dataset_name: str,
        kind: Literal['train', 'val', 'test'] = 'train',
        dataset_path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
        num_validation_dates: int = 0,
        num_test_dates: int = 0,
        num_splits: list = [0,0,0],
        regenerate: bool = False,
        prediction_length: int = None,
        history_length: int = None,
        total_length: int = None,
        interest_length: int = None,
        window_offset: int = None,
        random_offset: bool = False,
        draw_ratio = 0.5,
        version:int = -1,
        mode: str = 'imputation',
    ): 
        
        self.draw_ratio = draw_ratio
        # self.max_length = prediction_length + history_length
        self.full_length = False
        self.z = None
        self.model_type = cfg.model.type
        self.interest_length = interest_length
        self.total_length = total_length
        self.prediction_length = prediction_length
        self.history_length = history_length
        self.window_size = prediction_length + history_length
        self.model_mode = mode

        if self.total_length == 0:
            self.total_length = prediction_length + history_length
    

        super(ElectricityBaselineForecast, self).__init__(dataset_name=dataset_name, kind=kind, dataset_path=dataset_path, num_splits=num_splits,
                                              regenerate=regenerate, prediction_length=prediction_length, history_length=history_length,
                                              window_offset=self.total_length, random_offset=random_offset, version=version, total_length=self.total_length-self.history_length)
        if dataset_name == "electricity_hourly_new":
            self.data_all = np.delete(self.data_all, 106, axis=1)
        if dataset_name == "traffic":
            self.data_all = np.delete(self.data_all, 743, axis=1)
            
        if self.kind == 'train':
            history_length = total_length
            self.data_all = self._slice_data(self.data_all, history_length= history_length, prediction_length=0, window_offset=history_length+0, random_offset=random_offset)
        else:
            self.data_all = self._slice_data(self.data_all, history_length= history_length, prediction_length=prediction_length, window_offset=history_length+prediction_length, random_offset=random_offset)
        
        self.times_all = self.create_times()[[0]]
        self.user_ids_all = self.create_user_ids()
        self.targets_all = None

        self.data_all_safe = self.data_all.copy()
        seeds_dict = {'train': 0, 'val': 1, 'test': 2}

        self.permutation_indx_main = self._create_permutations(self.data_all.shape, seed = seeds_dict[kind])

        self.version_len = num_splits[['train', 'val', 'test'].index(kind)]

        self._select_version(version)
        if self.kind == 'train':
            self.pseudo_days = self.data_all.shape[-1] // (720+512)
            self.other_versions = [[i for i in range(0, self.pseudo_days)] + [-1]]
            self.start_indices = np.linspace(0, self.total_length - self.window_size, self.pseudo_days, dtype=int)
        # if self.model_type== 'timeflow':
        self.z = torch.zeros(np.prod(self.data_all.shape[0:3]), cfg.inr.latent_dim)

        self.split = kind


    def __getitem__(self, idx, full_length=False):

        """
        Args:
            idx (int): Index
        Input shapes:
            self.data.shape: (16050, 1, 1, 200)
            self.times.shape: (16050, 2, 1, 200)
            self.user_ids.shape: (16050, 1)
        Returns:
            ts.shape: (1, 3, 128)
            times_out.shape: (2, 3, 128)
            perm_idx.shape: (1, 128)
            user_ids_out.shape: (1,)
        """
        # print("idx:",idx)
        #TODO we need to write a mask to get indices with correct dims in case the signal is multi channel

        if self.kind == 'train' :
            if not self.full_length:
                end_point = torch.randint(self.window_size, self.interest_length, (1,))
                start_point = end_point - self.window_size
            else:
                start_point= self.start_indices[self.version]
                # start_point = self.interest_length
                end_point = start_point + self.window_size
                assert end_point <= self.total_length, "The endpoint is greater than total length"
        else:
            start_point = 0
            end_point = self.window_size

        if cfg.dataset.use_number_batch == 1:
            start_point = 0
            end_point = self.window_size


        ts, times_out, user_ids_out = self.data[idx], self.times[idx], self.user_ids[idx]
        z = self.z[idx] if self.z is not None else None
        perm_idx = self.permutation_indx[idx]
        user_ids_out = user_ids_out.reshape(-1)

        
        sample_ts = ts[:, :, start_point:end_point]
        assert self.history_length + self.prediction_length == sample_ts.shape[-1], "The sample length is not correct"
        
        ts_passed = sample_ts[:, :, :self.history_length]
        ts_horizon = sample_ts[:, :, -self.prediction_length:]

        times_passed = times_out[:, :, :self.history_length]
        times_horizon = times_out[:, :,  -self.prediction_length:]



        # target_out = 1.0

        #permute and get the draw_ratio amount of random data
        if self.full_length or self.draw_ratio == 1.0:
            ts_passed = ts_passed
            ts_horizon = ts_horizon
            times_passed = times_passed
            times_horizon = times_horizon
            perm_idx_passed = torch.arange(self.history_length)[None, :]
            perm_idx_horizon = torch.arange(self.prediction_length)[None, :]
        else:
            # ts = ts[:, np.random.choice(ts.shape[1], int(self.draw_ratio*ts.shape[1]), replace=False)]
            # times_out = times_out[:,np.random.choice(times_out.shape[-1], int(self.draw_ratio*times_out.shape[-1]), replace=False)]
            # perm_idx = perm_idx[:, :int(self.draw_ratio*ts.shape[1])]
            perm_idx_passed = torch.randperm(self.history_length)[None, :]
            perm_idx_horizon = torch.randperm(self.prediction_length)[None, :]
            perm_idx_passed = perm_idx_passed[:, :int(self.draw_ratio*ts_passed.shape[-1])]
            perm_idx_horizon = perm_idx_horizon[:, :int(self.draw_ratio*ts_horizon.shape[-1])]
            
            ts_passed = ts_passed[:, :, perm_idx_passed[0]]
            ts_horizon = ts_horizon[:, :, perm_idx_horizon[0]]

            times_passed = times_passed[:, :, perm_idx_passed[0]]
            times_horizon = times_horizon[:, :, perm_idx_horizon[0]]
        
        #to torch
        # ts = torch.tensor(ts, dtype=torch.float32)
        # times_out = torch.tensor(times_out, dtype=torch.float32)

        #ts to float
        # ts = ts.float()

        ts_passed = torch.tensor(ts_passed, dtype=torch.float32).float()
        ts_horizon = torch.tensor(ts_horizon, dtype=torch.float32).float()

        times_passed = torch.tensor(times_passed, dtype=torch.float32)
        times_horizon = torch.tensor(times_horizon, dtype=torch.float32)



        #normalize ts standardize
        if cfg.dataset.spatial_norm=="none_z_all":
            mean = ts_passed.mean()
            std = ts_passed.std()
            ts_passed = (ts_passed - mean) / (std + 1e-6)
            ts_horizon = (ts_horizon - mean) / (std + 1e-6)
        elif cfg.dataset.spatial_norm=="none_01_all":
            max_val = ts_passed.max()
            min_val = ts_passed.min()
            ts_passed = (ts_passed - min_val) / (max_val - min_val + 1e-6)
            ts_horizon = (ts_horizon - min_val) /  (max_val - min_val + 1e-6)
        else:
            pass
        
        if self.model_type == 'timeflow': #no need for second time axis only linear
            return ts_passed, ts_horizon, times_passed[[1]], times_horizon[[1]], z, perm_idx_passed, perm_idx_horizon, user_ids_out, torch.Tensor(0), torch.Tensor(0)
        else:
            return ts_passed, ts_horizon, times_passed, times_horizon, z, perm_idx_passed, perm_idx_horizon, user_ids_out, torch.Tensor(0), torch.Tensor(0)
        

class GluonTSSequenceScaler:
    """
    A Scaler class for sequences loaded from GluonTS datasets.

    """

    def __call__(self, sequences):
        return self.transform(sequences)

    def fit(self, data: np.array, eps=1e-7):
        """Save the means and stds of a training set.
        Note: Assumes that the data is already cleaned (does not contain NaNs or infs).
        
        Args:
            data (np.array): training set of sequences (N, dim, T) or (dim, T)
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        """
        if data.ndim == 2:
            sequences = [data]
        elif data.ndim == 3:
            sequences = [seq for seq in data]
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        
        sequences = np.hstack(sequences) # [dim, T] | [dim, T * N]
        means = np.mean(sequences, axis=1) # [dim]
        stds = np.std(sequences, axis=1) # [dim]
        
        print(f"Sequences fit with avg. mean: {means.mean()} and avg. stds: {stds.mean()}. (Avg. is over dimensions)")
        self.means = means.reshape(1, -1) # [1, dim]
        self.stds = stds.reshape(1, -1) # [1, dim]
        self.eps = eps

    # note that we change T and dim to stay consistent with our framework
    def transform(self, sequences: np.ndarray):
        """Apply the normalization

        Args:
            sequences (np.ndarray): single sequence or batch of sequences  (T, dim) | (N, T, dim)

        Returns:
            (np.ndarray): normalized single sequence or batch of sequences   (T, dim) | (N, T, dim)
        """
        if sequences.ndim == 2:
            means, stds = self.means, self.stds
        elif sequences.ndim == 3:
            means, stds = np.expand_dims(self.means, axis=1), np.expand_dims(self.stds, axis=1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        # broadcasts automatically
        return (sequences - means) / (stds + self.eps)
            
    def fit_transform(self, sequences):
        """Fit the scaler and transform in one step

        Args:
            sequences (np.array): train_set (dim, T) | (N, dim, T)

        Returns:
            (np.array): normalized batch of sequences (dim, T) | (N, dim, T) 
        """
        self.fit(sequences)
        if sequences.ndim == 2:
            return self.transform(sequences.T).T
        elif sequences.ndim == 3:
            return self.transform(sequences.transpose(0, 2, 1)).transpose(0, 2, 1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        
    # again note that we change T and dim to stay consistent with our framework
    def inverse_transform(self, sequences_normalized):
        """Apply the DEnormalization

        Args:
            sequences_normalized (np.array): single sequence or batch of normalized sequences (T, dim) | (N, T, dim)

        Returns:
            (np.array): single sequence or batch of denormalized sequences (T, dim) | (N, T, dim)
        """
        if sequences_normalized.ndim == 2:
            means, stds = self.means, self.stds
        elif sequences_normalized.ndim == 3:
            means, stds = np.expand_dims(self.means, axis=1), np.expand_dims(self.stds, axis=1)
        else:
            raise NotImplementedError("Data must be either 2D or 3D.")
        return sequences_normalized * (stds + self.eps) + means

    

# ====================== Data Loaders ======================
def get_gluonts_data_loader(
    dataset_name: str, 
    split='train', 
    path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
    num_validation_dates: int = 0,
    regenerate: bool = False,
    prediction_length: int = None,
    history_length: int = None,
    window_offset: int = None,
    random_offset: bool = False,
    batch_size=64, 
    num_workers=4, 
    shuffling= True, 
    persistent_workers = False, 
    **kwargs
) -> DataLoader:
    if dataset_name == "electricity_nips":
        data = ElectricityNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                                  prediction_length=prediction_length, history_length=history_length, 
                                  window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    elif dataset_name == "solar_nips":
        data = SolarNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                         prediction_length=prediction_length, history_length=history_length, 
                         window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    elif dataset_name == "wiki2000_nips":
        data = Wiki2000NIPS(kind=split, dataset_path=path, regenerate=regenerate, num_validation_dates=num_validation_dates,
                            prediction_length=prediction_length, history_length=history_length, 
                            window_offset=window_offset, random_offset=random_offset)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffling, num_workers=num_workers, persistent_workers=persistent_workers)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return data_loader



# ====================== DataSets ======================
def get_gluonts_dataset(
    dataset_name: str, 
    split='train', 
    path: pathlib.Path = pathlib.Path(os.getcwd()) / "data_gluonts",
    num_splits: list = [0, 0, 0],
    regenerate: bool = False,
    prediction_length: int = None,
    history_length: int = None,
    total_length: int = None,
    interest_length: int = None,
    window_offset: int = None,
    random_offset: bool = False,
    draw_ratio: float = 1.0,
    version: int = -1,
    mode: str = 'forecasting',
    
    **kwargs
) -> DataLoader:
    # if dataset_name == "electricity_nips":
    #     data = ElectricityNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_splits=num_splits,
    #                               prediction_length=prediction_length, history_length=history_length, 
    #                               window_offset=window_offset, random_offset=random_offset)
    # # elif dataset_name == "solar_nips":
    # #     data = SolarNIPS(kind=split, dataset_path=path, regenerate=regenerate, num_splits=num_splits,
    # #                      prediction_length=prediction_length, history_length=history_length, 
    # #                      window_offset=window_offset, random_offset=random_offset)
    # elif dataset_name == "wiki2000_nips":
    #     data = Wiki2000NIPS(kind=split, dataset_path=path, regenerate=regenerate, num_splits=num_splits,
    #                         prediction_length=prediction_length, history_length=history_length, 
    #                         window_offset=window_offset, random_offset=random_offset)
    if dataset_name in  ["electricity_hourly", "electricity_hourly_new", "traffic", "solar-energy-h", "solar-energy-10"]:
        if mode == 'forecasting':
            data = ElectricityBaselineForecast(dataset_name = dataset_name, kind=split, dataset_path=path, regenerate=regenerate, num_splits=num_splits,
                                  prediction_length=prediction_length, history_length=history_length, 
                                  window_offset=window_offset, random_offset=random_offset, draw_ratio=draw_ratio, version=version, total_length=total_length, interest_length=interest_length,mode=mode)
        elif mode == 'imputation':
            data = ElectricityBaseline(dataset_name = dataset_name, kind=split, dataset_path=path, regenerate=regenerate, num_splits=num_splits,
                                  prediction_length=prediction_length, history_length=history_length, 
                                  window_offset=window_offset, random_offset=random_offset, draw_ratio=draw_ratio, version=version, total_length=total_length, interest_length=interest_length, mode=mode)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return data

"""
def test():
    data_loader = get_gluonts_data_loader("electricity_nips", split='train', batch_size=64, num_workers=4, shuffling=True, persistent_workers=True, random_offset=True)
    for k in range(200):
        print(f"k = {k}")
        for i, (x, c, T0, T) in enumerate(data_loader):
            if k < 1:
                print(f"batch {i}: x.shape = {x.shape}, c.shape = {c.shape}, T0[0] = {T0[0]}, T[0] = {T[0]}")
    
if __name__ == "__main__":     
    test()
"""

"""
def test2():
    data = ElectricityNIPS(
        kind='train', 
        regenerate=False, 
        prediction_length=24, 
        history_length=24, 
        window_offset=None, 
        random_offset=True)
    
    from gluonts.model.forecast import SampleForecast
    from gluonts.evaluation import MultivariateEvaluator
    evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})
        
    t, _ = data.create_pandas_evaluation_iterator(kind='test')
    test_forecasts = [
        SampleForecast(
            samples= np.random.randn(100, 24, 370) * 100,
            start_date=pd.Period("2014-08-31 01:00", freq='H') + (i+1) * 24,
        ) for i in range(7)
    ]
    it_2 = iter(test_forecasts)
    agg_metric, ts_wise_metrics = evaluator(t, it_2)
    metrics = {
            "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),             # same as agg_metric['mean_absolute_QuantileLoss'] / agg_metric['abs_target_sum']
            "ND": agg_metric.get("ND", float("nan")),
            "NRMSE": agg_metric.get("NRMSE", float("nan")),
            "MSE": agg_metric.get("MSE", float("nan")),
            "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),   # same as agg_metric['m_sum_mean_absolute_QuantileLoss'] / agg_metric['m_sum_abs_target_sum']
            "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
            "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
            "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan"))
        }
    pass
    
    
    from gluonts.evaluation import make_evaluation_predictions
    
test2()
"""

"""
def test3():
    # data = get_gluonts_multivar_dataset("wiki2000_nips", regenerate=True, print_stats=False)
    data = Wiki2000NIPS(train=True, regenerate=True)
test3()
"""
"""
def test4():
    data = ElectricityNIPS(
        kind='val',
        dataset_path=pathlib.Path(os.getcwd()) / "testing_datasets",
        num_validation_dates=2,
        regenerate=True,
        random_offset=True
    )

    
test4()
"""