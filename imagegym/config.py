import logging
import os
from yacs.config import CfgNode as CN

from imagegym.utils.io import makedirs_rm_exist

from imagegym.contrib.config import *
import imagegym.register as register

# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in imagegym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Set print destination: stdout / file / both
    cfg.print = 'both'

    cfg.gpu = 0

    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'cpu'

    # Output directory
    cfg.out_dir = 'results'

    # Config destination (in OUT_DIR)
    cfg.cfg_dest = 'config.yaml'

    # Random seed
    cfg.seed = 1

    # Print rounding
    cfg.round = 4
    
    # Additional num of worker for data loading
    cfg.num_workers = 2

    # For data loader
    cfg.pin_memory = False

    # Max threads used by PyTorch
    cfg.num_threads = 6

    cfg.params = None
    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = 'MNIST'

    #threshold #if this is zero no threshold.
    cfg.dataset.threshold = 0.0

    # if PyG: look for it in Pytorch Geometric dataset
    # if NetworkX/nx: load data in NetworkX format
    cfg.dataset.format = 'torch'

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = './datasets'

    # Dimension for node feature, edge feature. Updated by the real dim of the dataset
    cfg.dataset.dims = None

    cfg.dataset.label_dim = None

    # Resizing purposes of image data
    cfg.dataset.size = 64

    # Use only a percentage of the dataset (0, 1]
    cfg.dataset.use_subset = 1.0

    # Use only limited number of batch (if 0 omitted)
    cfg.dataset.use_number_batch = 0

    # Use input batch normaliation
    cfg.dataset.use_bn_initial = True

    #  Percentage of missing concepts
    cfg.dataset.missing_perc = 0.0 #if it is 1 we use uniform missingness

    cfg.dataset.missing_perc_min = 0.0

    # missingness types
    cfg.dataset.missing_type = "uniform"

    cfg.dataset.sparsity = "DT"

    # Use one hot encoding for label
    cfg.dataset.use_one_hot = False

    # Fourier implementation
    cfg.dataset.fourier = False

    # Use training data in validation
    cfg.dataset.use_train_as_valid = False
    
    #config for PolyMNIST modality
    cfg.dataset.modality = "m1"

    #check data or not?
    cfg.dataset.check_data = False

    cfg.dataset.spatial_norm = None

    cfg.dataset.temporal_norm =  None

    cfg.dataset.val_perc = 0.0

    cfg.dataset.num_train_dates = 1

    cfg.dataset.num_validation_dates = 1

    cfg.dataset.num_test_dates = 1

    cfg.dataset.draw_ratio = 1.0

    cfg.dataset.version = -1

    cfg.dataset.coordinate_dim = None

    cfg.dataset.dims_draw = None

    cfg.dataset.task = "forecasting" #imputation, forecasting

    cfg.dataset.num_splits = None

    cfg.dataset.split_names = None
    
    cfg.dataset.cond_type = None

    cfg.dataset.dims_c = None

    cfg.dataset.dims_target = None


    # Type of task: imputation, forecasting
    # classification_multi
    # cfg.dataset.task_type = 'forecasting'
    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()

    # Training (and validation) pipeline mode
    cfg.train.mode = 'standard'

    cfg.train.accumulation_steps = None

    # Total graph mini-batch size
    cfg.train.batch_size = 16

    # Evaluate model on test data every eval period epochs
    cfg.train.eval_period = 10

    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 100

    # Resume training from the latest checkpoint in the output directory
    cfg.train.auto_resume = False

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True

    # Number of iterations per epoch (for sampling based loaders only)
    cfg.train.iter_per_epoch = 32

    # Number of iterations per epoch (for sampling based loaders only)
    cfg.train.clip = 0.0

    cfg.train.resume_training = False
    # ------------------------------------------------------------------------ #
    # Validation options
    # ------------------------------------------------------------------------ #
    cfg.val = CN()


    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    # Model type to use
    cfg.model.type = 'tv_inr'

    cfg.model.conditional = False

    cfg.model.use_same_label = False

    cfg.model.elbo_mode = 'observed' #full

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'elbo'

    # Dimension for latent space
    cfg.model.dim_z = 8

    # Distribution name for latent space (prior)
    cfg.model.distr_z = 'normal' #nf for normalizing flow

    # Beta coefficient for the latent kl
    cfg.model.beta_z= 1.0
    
    # Beta coefficient for the cat  kl
    cfg.model.beta_c= 0.00001

    # Type of scheduler
    cfg.model.beta_c_scheduler = "Linear"

    # Type of scheduler
    cfg.model.beta_z_scheduler = "Linear"

    # Type of scheduler
    cfg.model.beta_missing_perc_scheduler = "Linear"

    cfg.model.direction_scheduler = "min_to_max"

    # Epoch to start scheduler
    cfg.model.start_scheduler = 0

    # Epoch to end scheduler
    cfg.model.end_scheduler = 0
    
    # Dimension for cond space
    cfg.model.dim_cond = 0

    # Distribution name for cond space
    cfg.model.distr_cond= None

    # Beta coefficient for the cond kl
    cfg.model.beta_cond= 1

    # Distribution name for x
    cfg.model.distr_x = 'logistic'

    # Log scale for logistic dist
    cfg.model.distr_x_logscales = -1

    #Fix or learn log scales
    cfg.model.distr_x_logscales_learn = False
    
    #Scale LL in the ELBO with length
    cfg.model.elbo_ll_scale = False

    # Use the real cond during training
    cfg.model.name_encoding = None

    # Select encoder type
    cfg.model.encoder_type = "pointconv"

    # Select if you want to use mixture
    cfg.model.use_k_mixture= True

    # U
    cfg.model.drop_input = 0.0

    #
    cfg.model.dropout = 0.0

    #two step training

    cfg.model.two_step_training = False

    cfg.model.first_step_ratio = 0.0

    cfg.model.scenerio = []

    #residual posterior
    cfg.model.learn_residual_posterior = False

    #if z in the cat_post
    cfg.model.post_cat_has_z = True

    # if we use the simple version of the categorical
    cfg.model.simple_cat = False

    # if we use encoder for x to porject first
    cfg.model.cat_encoder_x = False

    # if using a uniform categorical prior
    cfg.model.fix_categorical_prior = False

    cfg.model.scenerio_start =  []
    
    cfg.model.scenerio_end = []

    cfg.model.temporal_grid_norm = None

    cfg.model.tmh = None

    cfg.model.tmp = None

    cfg.model.tmt = None

    cfg.model.tm_interest = None
    # ------------------------------------------------------------------------ #
    # NF options
    # ------------------------------------------------------------------------ #

    cfg.params_nf = CN()

    #nf type
    cfg.params_nf.type = "planar"

    #nf #layer
    cfg.params_nf.L = 10

    #nf activation
    cfg.params_nf.act = "leaky_relu"


    # ------------------------------------------------------------------------ #
    #  Hypernet options
    # ------------------------------------------------------------------------ #
    cfg.params_hyper = CN()

    
    cfg.params_hyper.name = "transformer"
    # Type of graph conv: dens
    cfg.params_hyper.dim_inner = 32

    # Type of graph conv: dens
    cfg.params_hyper.num_layers = 2

    # Whether use batch norm
    # cfg.params_hyper.batchnorm = False

    # Activation
    cfg.params_hyper.act = 'relu'

    # Dropout
    cfg.params_hyper.dropout = 0.0

    # # Coordinate dim if it is 2 it is concatenatened with the latent to be fed into decoder # if this is bigger than 0 
    cfg.params_hyper.coords_dim=0

    cfg.params_hyper.depth = 2

    cfg.params_hyper.n_heads = 2

    cfg.params_hyper.head_dim = 8

    cfg.params_hyper.ff_dim = 512

    cfg.params_hyper.dropout = 0.1

    cfg.params_hyper.agg = 'repeat'

    cfg.params_hyper.n_patches = 1

    cfg.params_hyper.n_groups = 1

    cfg.params_hyper.decoder_norm = ["linear","output"]




    # ------------------------------------------------------------------------ #
    #  FeatureRepresentation options
    # ------------------------------------------------------------------------ #
    cfg.params_fnrep = CN()

    # Type of graph conv: dens
    cfg.params_fnrep.dim_inner = 32

    # Type of graph conv: dens
    cfg.params_fnrep.num_layers = 2

    # Activation
    cfg.params_fnrep.act = 'relu'

    # Dropout
    cfg.params_fnrep.dropout = 0.0

    cfg.params_fnrep.shared_layer_indx = []

    cfg.params_fnrep.mode = "wb"




    # ------------------------------------------------------------------------ #
    #  PointConv options
    # ------------------------------------------------------------------------ #

    cfg.params_pointconvnet = CN()

    cfg.params_pointconvnet.coordinate_dim = 0 #coordinates
    cfg.params_pointconvnet.feature_dim = 0 #features

    cfg.params_pointconvnet.same_coordinates = "none"
    cfg.params_pointconvnet.deterministic = True
    cfg.params_pointconvnet.add_batchnorm = True
    cfg.params_pointconvnet.add_weightnet_batchnorm=False
    cfg.params_pointconvnet.linear_layer_sizes = []
    
    cfg.params_pointconvnet.layer_configs = []
    cfg.params_pointconvnet.out_channels = []
    cfg.params_pointconvnet.num_output_points = []
    cfg.params_pointconvnet.num_neighbors = []
    cfg.params_pointconvnet.mid_channels = []
    
    cfg.params_pointconvnet.avg_pooling_num_output_points = []
    cfg.params_pointconvnet.avg_pooling_num_neighbors = []
    cfg.params_pointconvnet.add_sigmoid = False
    #norm order?

    cfg.params_pointconvnet.use_encoded_coors = False

    # ------------------------------------------------------------------------ #
        #  Transformer options
    # ------------------------------------------------------------------------ #

    cfg.params_transformer = CN()

    cfg.params_transformer.coordinate_dim = 0 #coordinates
    cfg.params_transformer.feature_dim = 0 #features

    cfg.params_transformer.d_model = 0
    cfg.params_transformer.attention_layers = 0
    cfg.params_transformer.add_sigmoid = False
    cfg.params_transformer.n_heads = 0
    cfg.params_transformer.causal = False
    cfg.params_transformer.dropout = 0.0
    cfg.params_transformer.linear_layer_sizes = []
    cfg.params_transformer.max_len = 500



    # ------------------------------------------------------------------------ #
    #  K mixture options
    # ------------------------------------------------------------------------ #

    cfg.params_k_mixture = CN()

    cfg.params_k_mixture.K = 1

    # ------------------------------------------------------------------------ #
    #  Categorical X (Encoder) options
    # ------------------------------------------------------------------------ #

    cfg.params_cat_x = CN()

    cfg.params_cat_x.layers = [5,5]

    cfg.params_cat_x.batchnorm = False

    cfg.params_cat_x.l2norm = False

    cfg.params_cat_x.act = 'softmax'

    cfg.params_cat_x.dropout = 0.0

    cfg.params_cat_x.output_dropout = 0.0

    # ------------------------------------------------------------------------ #
    #  Categorical Prior (Encoder) options NOT USED
    # ------------------------------------------------------------------------ #

    cfg.params_cat_prior = CN()
    
    cfg.params_cat_prior.layers = [5,5]

    cfg.params_cat_prior.batchnorm = False

    cfg.params_cat_prior.l2norm = False

    cfg.params_cat_prior.act = 'softmax'

    cfg.params_cat_prior.dropout = 0.0

    cfg.params_cat_prior.output_dropout = 0.0


    # ------------------------------------------------------------------------ #
    #  Categorical Posterior (Encoder) options NOT USED
    # ------------------------------------------------------------------------ #

    cfg.params_cat_post = CN()
    
    cfg.params_cat_post.layers = [5,5]

    cfg.params_cat_post.batchnorm = False

    cfg.params_cat_post.l2norm = False

    cfg.params_cat_post.act = 'softmax'

    cfg.params_cat_post.dropout = 0.0

    cfg.params_cat_post.output_dropout = 0.0


    # ------------------------------------------------------------------------ #
    #  Categorical Condition Encoder options
    # ------------------------------------------------------------------------ #

    cfg.params_cat_cond = CN()

    cfg.params_cat_cond.dim_z_c = 4

    cfg.params_cat_cond.one_hot = True
    
    cfg.params_cat_cond.layers = [5,5]

    cfg.params_cat_cond.batchnorm = False

    cfg.params_cat_cond.l2norm = False

    cfg.params_cat_cond.act = 'prelu'

    cfg.params_cat_cond.dropout = 0.0

    cfg.params_cat_cond.output_dropout = 0.0

    # ------------------------------------------------------------------------ #
    # Features Encoder options
    # ------------------------------------------------------------------------ #
    cfg.params_encoding = CN()

    # Number of frequencies
    cfg.params_encoding.num_frequencies = 10

    cfg.params_encoding.num_frequencies_enc = 11

    # Init standard dev. of the frequency matrix
    cfg.params_encoding.std = 0.1

    # Whether use learn the  frequency matrix or not
    cfg.params_encoding.learn_feats = False

    # Whether logscale with frequency band
    cfg.params_encoding.log_scale = False

    cfg.params_encoding.base_freq = 2.0

    cfg.params_encoding.use_pi = False

    cfg.params_encoding.min_freq_log2 = 0.0

    cfg.params_encoding.max_freq_log2 = 10.0

    # ------------------------------------------------------------------------ #
    # Plotting Figures options
    # ------------------------------------------------------------------------ #

    cfg.plotting =CN()

    # plot the super res size images per epoch 
    cfg.plotting.super_res_epoch = 50

    # plot the same size images per epoch 
    cfg.plotting.res_epoch = 50

    cfg.plotting.use_neighbors = False

    cfg.plotting.figure_type = ".png"

    # ------------------------------------------------------------------------ #
    # Optimizer options
    # ------------------------------------------------------------------------ #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'
    
    # Use scheduler
    cfg.optim.use_scheduler = True

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    #For scheduler
    cfg.optim.gamma = 0.99

    # Maximal number of epochs
    cfg.optim.max_epoch = 200

    #checks last X values of the val loss, should be decided with eval period#
    cfg.optim.early_stop = 2 

    cfg.optim.patience = 10

    cfg.optim.metric = 'loss'

    # ------------------------------------------------------------------------ #
    # Batch norm options
    # ------------------------------------------------------------------------ #
    cfg.bn = CN()

    # BN epsilon
    cfg.bn.eps = 1e-5

    # BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
    cfg.bn.mom = 0.1

    # ------------------------------------------------------------------------ #
    # Memory options
    # ------------------------------------------------------------------------ #
    cfg.mem = CN()

    # Perform ReLU inplace
    cfg.mem.inplace = False

    
    # ------------------------------------------------------------------------ #
    # Note options
    # ------------------------------------------------------------------------ #

    cfg.note = ""

    # ------------------------------------------------------------------------ #
    # Inference options
    # ------------------------------------------------------------------------ #

    cfg.inference = CN()
    
    cfg.inference.save_samples = False

    cfg.inference.reconstruct = False

    cfg.inference.ckpt_numbers = [49]

    cfg.inference.first_batch = False

    cfg.inference.mode = None
    
    cfg.inference.wandb_ckpt_name = None

    cfg.inference.wandb_run_id = None
    
    cfg.inference.wandb_run_dir = None


    # ------------------------------------------------------------------------ #
    # Set user customized cfgs
    # ------------------------------------------------------------------------ #

    cfg.wandb = CN()
    cfg.wandb.use = True
    cfg.wandb.project_name = None
    cfg.wandb.entity = None
    cfg.wandb.mode = 'offline'
    cfg.wandb.create_cfgdir = False
    cfg.wandb.log_dir = None
    cfg.wandb.logger = None
    cfg.wandb.project = None
    cfg.wandb.enable = True
    cfg.wandb.dir = None
    cfg.wandb.resume = None
    cfg.wandb.gpoup = None

    cfg.inr = CN()
    cfg.inr.model_type = "fourier_features"
    cfg.inr.latent_dim = 128
    cfg.inr.depth = 5
    cfg.inr.hidden_dim = 256
    cfg.inr.num_frequencies = 64
    cfg.inr.modulate_scale = False
    cfg.inr.modulate_shift = True
    cfg.inr.frequency_embedding = "nerf"
    cfg.inr.max_frequencies = 10
    cfg.inr.min_frequencies = 0.0
    cfg.inr.base_frequency = 2
    cfg.inr.include_input = True
    cfg.inr.scale = 5
    cfg.inr.w_passed = 0.5
    cfg.inr.w_futur = 0.5
    cfg.inr.passed_ratio = 0.3
    cfg.inr.horizon_ratio = 0.3
    cfg.inr.log_sampling = True
    cfg.inr.lr_code = 0.01
    cfg.inr.meta_lr_code = 0.0
    cfg.inr.weight_decay_code = 0.0


    cfg.saits = CN()
    cfg.saits.model_name = "Electricity_SAITS_best"
    cfg.saits.input_with_mask = True
    cfg.saits.model_type = "SAITS"
    cfg.saits.n_groups = 1
    cfg.saits.n_group_inner_layers = 1
    cfg.saits.param_sharing_strategy = "inner_group"
    cfg.saits.d_model = 1024
    cfg.saits.d_inner = 128
    cfg.saits.n_head = 8
    cfg.saits.d_k = 128
    cfg.saits.d_v = 128
    cfg.saits.dropout = 0.2
    cfg.saits.diagonal_attention_mask = True
    cfg.saits.MIT = True
    cfg.saits.ORT = True
    cfg.saits.reconstruction_loss_weight = 1
    cfg.saits.imputation_loss_weight = 1

    cfg.deeptime = CN()
    cfg.deeptime.model_name = "DeepTime_model"
    cfg.deeptime.datetime_feats = 0
    cfg.deeptime.model_type = "deeptime_wrapper"
    cfg.deeptime.layer_size = 256
    cfg.deeptime.inr_layers = 5
    cfg.deeptime.n_fourier_feats = 4096
    cfg.deeptime.scales = [0.01, 0.1, 1, 5, 10, 20, 50, 100]
    cfg.deeptime.d_model = 256
    cfg.deeptime.d_inner = 256
    cfg.deeptime.input_with_mask = False
    cfg.deeptime.warmup_epochs = 5
    cfg.deeptime.loss_name = "mse"

    #### Set user customized cfgs
    for func in register.config_dict.values():
        func(cfg)


def assert_cfg(cfg):
    """Checks config values invariants."""

    if cfg.dataset.name in ['shapes3d','shapes3d_10','PolyMNIST','celebahq256']:
        cfg.dataset.task_type = 'generative'
        cfg.dataset.task = 'image'
    
    if cfg.dataset.name in ['era5']:
        cfg.dataset.task_type = 'polar'
        cfg.dataset.task = 'era5_polar'

    if cfg.dataset.name in ['voxels']:
        cfg.dataset.task_type = 'chairs'
        cfg.dataset.task = 'voxels_chairs'




def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def update_out_dir(out_dir, args, wandb_path=None):
    if wandb_path is not None:
        wandb_id = wandb_path.split('/')[-2]
        cfg.out_dir = os.path.join(out_dir, 'local_logs', wandb_id)
    else:
        fname = args.cfg_file
        fname = os.path.splitext(args.cfg_file.split('/')[-1])[0]
        cfg.out_dir = os.path.join(out_dir, fname, str(cfg.seed))

    if args.local: #run from a saved folder
        cfg.out_dir = args.cfg_file.split('config')[0]
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)


def get_parent_dir(out_dir, fname):
    fname = fname.split('/')[-1][:-5]
    return os.path.join(out_dir, fname)


def rm_parent_dir(out_dir, fname):
    fname = fname.split('/')[-1][:-5]
    makedirs_rm_exist(os.path.join(out_dir, fname))


set_cfg(cfg)
