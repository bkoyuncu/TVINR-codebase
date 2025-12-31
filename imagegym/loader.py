import torch
import torchvision
import os
from imagegym.config import cfg
from imagegym.contrib.utils.random import get_permutation
from torch.utils.data import DataLoader
from imagegym.datasets.datasets_gluonts import get_gluonts_data_loader, GluonTSDataset, get_gluonts_dataset


def dataset_checker(datasets, dataset_splits):
    assert len(datasets) == len(dataset_splits)
    print('Dataset Checker')
    print('Datasets Total: ', len(datasets))
    for i, dataset in enumerate(datasets):
        print(f'Dataset split {dataset_splits[i]} {i}')
        print('Dataset Length: ', dataset.__len__())
        print('Dataset All: ', dataset.data_all_safe.shape)
        print('Dataset Shape: (N, D, T)', dataset.data.shape)
        #get an batch of data
        for batch in dataset:
            print('Batch Length: ', len(batch))
            print([b.shape if b is not None else '-' for b in batch])
            break



def compute_split_idx(original_len, split_sizes, random=True):
    all_idx = torch.arange(original_len)
    if random:
        perm = get_permutation(original_len=original_len)
        all_idx = all_idx[perm]

    start_idx, end_idx = 0, None
    all_idx_splits = []

    num_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        assert isinstance(size, float)
        assert 0 < size
        assert 1 > size
        new_len = int(size * original_len)
        end_idx = new_len + start_idx
        if i == (num_splits - 1):
            all_idx_splits.append(all_idx[start_idx:])
        else:
            all_idx_splits.append(all_idx[start_idx:end_idx])
        start_idx = end_idx

    return all_idx_splits


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''

    return datasets


def load_torch(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = os.path.join(dataset_dir, name)  # './datasets/MNIST'
    print(dataset_dir)
    datasets = []
    if name in ['PolyMNIST']:
        from imagegym.datasets.my_polymnist import MyPolyMNIST
        from torchvision import transforms
        
        dataset_train = MyPolyMNIST(root=dataset_dir, train=True,
                                    download=False,
                                    missing_perc=cfg.dataset.missing_perc,
                                    use_one_hot=cfg.dataset.use_one_hot,
                                    modality=cfg.dataset.modality)

        setattr(dataset_train, 'num_classes', 10)
        dataset_test = MyPolyMNIST(root=dataset_dir, train=False,
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                   modality=cfg.dataset.modality)

        cfg.dataset.dims = [3, 28, 28]
        cfg.dataset.label_dim = 10
        cfg.dataset.coordinate_dim = 2

        datasets.append(dataset_train)
        datasets.append(dataset_test)

    elif name in ['HAR']:
        from imagegym.datasets.my_har import HAR
        mode = cfg.dataset.task
        import pathlib
        tmh = cfg.model.tmh
        tmf = cfg.model.tmp
        tmt = cfg.model.tmt
        tm_interest = cfg.model.tm_interest
        window_offset = (tmh + tmf)
        num_train_dates = cfg.dataset.num_train_dates
        num_validation_dates = cfg.dataset.num_validation_dates
        num_test_dates = cfg.dataset.num_test_dates
        num_splits = [num_train_dates, num_validation_dates, num_test_dates]
        version = cfg.dataset.version
        dataset_splits = []
        dataset_dict = {}
        model_type = cfg.model.type

        if num_train_dates > 0:
            dataset_train = HAR(root=dataset_dir, split="train",
                                    download=False,
                                    missing_perc=cfg.dataset.missing_perc,
                                    use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                    )
            datasets.append(dataset_train)
            dataset_splits.append('train')
            dataset_dict['train'] = dataset_train
        if num_validation_dates > 0:
            dataset_valid = HAR(root=dataset_dir, split="val",
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                   )
            datasets.append(dataset_valid)
            dataset_splits.append('val')
            dataset_dict['val'] = dataset_valid
        if num_test_dates > 0:
            dataset_test = HAR(root=dataset_dir, split="test",
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                    )
            datasets.append(dataset_test)
            dataset_splits.append('test')
            dataset_dict['test'] = dataset_test


        cfg.dataset.dims = [1, 3, 128] # [dim+timestamp,T]
        cfg.dataset.dims_draw = [1,int((128)*cfg.dataset.draw_ratio)]
        cfg.dataset.dims_c = 30
        cfg.dataset.dims_target = 6
        cfg.dataset.label_dim = 1
        cfg.dataset.coordinate_dim = 2 #linear bc we are using the same function for images
        cfg.dataset.num_splits = num_splits

    elif name in ['P12', 'P12_new']:
        from imagegym.datasets.my_p12 import P12
        mode = cfg.dataset.task
        import pathlib
        tmh = cfg.model.tmh
        tmf = cfg.model.tmp
        tmt = cfg.model.tmt
        tm_interest = cfg.model.tm_interest
        window_offset = (tmh + tmf)
        num_train_dates = cfg.dataset.num_train_dates
        num_validation_dates = cfg.dataset.num_validation_dates
        num_test_dates = cfg.dataset.num_test_dates
        num_splits = [num_train_dates, num_validation_dates, num_test_dates]
        version = cfg.dataset.version
        dataset_splits = []
        dataset_dict = {}
        model_type = cfg.model.type

        if num_train_dates > 0:
            dataset_train = P12(root=dataset_dir, split="train",
                                    download=False,
                                    missing_perc=cfg.dataset.missing_perc,
                                    use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                    )
            datasets.append(dataset_train)
            dataset_splits.append('train')
            dataset_dict['train'] = dataset_train
        if num_validation_dates > 0:
            dataset_valid = P12(root=dataset_dir, split="val",
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                   )
            datasets.append(dataset_valid)
            dataset_splits.append('val')
            dataset_dict['val'] = dataset_valid
        if num_test_dates > 0:
            dataset_test = P12(root=dataset_dir, split="test",
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                    num_splits=num_splits,
                                    draw_ratio=cfg.dataset.draw_ratio,
                                    version = version,
                                    mode = mode,
                                    prediction_length=tmf,
                                    history_length=tmh,
                                    window_offset=window_offset,
                                    total_length=tmt,
                                    interest_length=tm_interest,
                                    model_type=model_type,
                                    cond_type=cfg.dataset.cond_type
                                    )
            datasets.append(dataset_test)
            dataset_splits.append('test')
            dataset_dict['test'] = dataset_test

        dim = dataset_train.dim_size
        cfg.dataset.dims = [1, dim, window_offset] # [dim+timestamp,T]
        cfg.dataset.dims_draw = [1,int((window_offset)*cfg.dataset.draw_ratio)]
        cfg.dataset.dims_c = dataset_train.condition_dim #TODO ?
        cfg.dataset.dims_target = 2 #survival class
        cfg.dataset.label_dim = 1
        cfg.dataset.coordinate_dim = 2 #linear bc we are using the same function for images
        cfg.dataset.num_splits = num_splits

    elif name in ['electricity_hourly', 'electricity_hourly_new', 'solar-energy-10', 'solar-energy-h', 'traffic']:
        from imagegym.datasets.datasets_gluonts import get_gluonts_dataset, GluonTSDataset
        mode = cfg.dataset.task
        import pathlib
        tmh = cfg.model.tmh
        tmf = cfg.model.tmp
        tmt = cfg.model.tmt
        tm_interest = cfg.model.tm_interest
        window_offset = (tmh + tmf)
        num_train_dates = cfg.dataset.num_train_dates
        num_validation_dates = cfg.dataset.num_validation_dates
        num_test_dates = cfg.dataset.num_test_dates
        num_splits = [num_train_dates, num_validation_dates, num_test_dates]
        version = cfg.dataset.version
        dataset_dict = {}
        dataset_splits = []
        if num_train_dates > 0:
            dataset_train = get_gluonts_dataset(
            name,
            split='train',
            prediction_length=tmf,
            history_length=tmh,
            window_offset=window_offset,
            random_offset=True,
            batch_size=None,
            num_workers=1, # change to more if needed
            shuffling=True,
            persistent_workers=True,
            path=pathlib.Path(cfg.dataset.dir),
            num_splits=num_splits,
            regenerate=False,
            draw_ratio=cfg.dataset.draw_ratio,
            version = version,
            mode = mode,
            total_length=tmt,
            interest_length=tm_interest
            )   
            datasets.append(dataset_train)
            dataset_splits.append('train')
            dataset_dict['train'] = dataset_train
        if num_validation_dates > 0:
            dataset_valid = get_gluonts_dataset(
            name,
            split='val',
            prediction_length=tmf,
            history_length=tmh,
            window_offset=window_offset,
            random_offset=True,
            batch_size=None,
            num_workers=1, # change to more if needed
            shuffling=True,
            persistent_workers=True,
            path=pathlib.Path(cfg.dataset.dir),
            num_splits=num_splits,
            regenerate=False,
            draw_ratio=cfg.dataset.draw_ratio,
            version = -1,
            mode = mode,
            total_length=tmt,
            interest_length=tm_interest
            )
            datasets.append(dataset_valid)
            dataset_splits.append('val')
            dataset_dict['val'] = dataset_valid
        if num_test_dates > 0:
            dataset_test = get_gluonts_dataset(
            name,
            split='test',
            prediction_length=tmf,
            history_length=tmh,
            window_offset=None,
            random_offset=False,
            batch_size=None,
            num_workers=1,
            shuffling=False,
            persistent_workers=True,
            path=pathlib.Path(cfg.dataset.dir),
            num_splits=num_splits,
            regenerate=False,
            draw_ratio=cfg.dataset.draw_ratio,
            version = -1,
            mode = mode,
            total_length=tmt,
            interest_length=tm_interest
            )
            datasets.append(dataset_test)
            dataset_splits.append('test')
            dataset_dict['test'] = dataset_test

        cfg.dataset.dims = [1,1,tmh+tmf]
        cfg.dataset.dims_draw = [1,1,int((tmh+tmf)*cfg.dataset.draw_ratio)]
        cfg.dataset.label_dim = 0
        cfg.dataset.coordinate_dim = 2 #linear bc we are using the same function for images
        cfg.dataset.num_splits = num_splits

    else:
        raise ValueError('{} not support'.format(name))

    assert cfg.dataset.dims is not None
    assert cfg.dataset.label_dim is not None

    # dataset_checker(datasets, dataset_splits)

    return dataset_dict


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format  # torch
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir  # './datasets'
    # Load from Pytorch Geometric dataset
    if format == 'torch':
        dataset_dict = load_torch(name, dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))

    return dataset_dict


def filter_samples(datasets):
    return datasets


def create_dataset():
    ## Load dataset
    datasets  = load_dataset()
    # datasets = transform_after_split(datasets)  # empty
    return datasets


def create_loader(datasets):
    
    total_split_names = ['train_main', 'train', 'val', 'test']

    if cfg.dataset.use_train_as_valid:
        train_shuffle = False
    else:
        train_shuffle = True

    # if cfg.dataset.check_data or cfg.train.mode == "sample":
    #     train_shuffle = False

    loader_train = DataLoader(datasets['train'],
                              batch_size=cfg.train.batch_size, shuffle=train_shuffle,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    loaders = [loader_train]

    loader_train = DataLoader(datasets['train'],
                              batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    loaders.append(loader_train)

    split_names = ["train_main", "train"]


    for i in [1,2]:
        split_name = total_split_names[i+1]

        if split_name == "val" and cfg.dataset.use_train_as_valid:
            loaders.append(DataLoader(datasets['train'],
                                      batch_size=cfg.train.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=cfg.pin_memory))
            split_names.append(split_name)
    
        elif cfg.dataset.num_splits[i]>0:
            loaders.append(DataLoader(datasets[split_name],
                                      batch_size=cfg.train.batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_workers,
                                      pin_memory=cfg.pin_memory))
            split_names.append(split_name)
            
    cfg.dataset.split_names = split_names
    return loaders
