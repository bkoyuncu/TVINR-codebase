# Based on https://github.com/EmilienDupont/neural-function-distributions
import torch
import torch.nn as nn
from imagegym.models.layer.our_mlp import MLP
from imagegym.models.layer.mlp_inr import MLPInr
from imagegym.models.layer.trans_inr import TransInr


def process_hypernet_params(input_dim, fn_representation, hyper_params):
    assert 'dim_inner' in hyper_params
    assert 'num_layers' in hyper_params
    assert 'coords_dim' in hyper_params
    assert hyper_params["coords_dim"] == 0 

    c_dim_list = [input_dim]
    dim_inner = hyper_params['dim_inner']
    layers_encoder = hyper_params['num_layers']
    c_dim_list.extend([dim_inner*(i+1) for i in range(layers_encoder)])
    hyper_params['c_dim_list'] = c_dim_list
    hyper_params['function_representation'] = fn_representation


class HyperNetwork(nn.Module):
    """Hypernetwork that outputs the weights of a function representation.

    Args:
        function_representation (models.function_representation.FunctionRepresentation):
        latent_dim (int): Dimension of latent vectors.
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        non_linearity (torch.nn.Module):
    """
    def __init__(self, function_representation,
                 c_dim_list,
                 dropout=0.0,
                 act='prelu',
                 coords_dim=0,
                 encoding=None,
                 **kwargs):
        super(HyperNetwork, self).__init__()
        assert function_representation is not None
        assert isinstance(c_dim_list, list)
        self.function_representation = function_representation
        self.c_dim_list = c_dim_list
        self.dropout = dropout
        self.act = act
        self.coords_dim = coords_dim
        self.mlp = None

        self.depth = kwargs["depth"]
        self.n_heads = kwargs["n_heads"]
        self.head_dim = kwargs["head_dim"]
        self.ff_dim = kwargs["ff_dim"]
        self.agg = kwargs["agg"]
        self.n_patches = kwargs["n_patches"]
        self.n_groups = kwargs["n_groups"]
        self.decoder_norm = kwargs["decoder_norm"]

        self.dropout = dropout


        # assert encoding is not None
        # self.encoding = function_representation['encoding']
        # self._infer_output_shapes()
        name = kwargs["name"]

        if name == "transformer":
            self._init_transformer_net()
        elif name == "mlp":
            self._init_neural_net()

    def _infer_output_shapes(self):
        """Uses function representation to infer correct output shapes for
        hypernetwork (i.e. so dimension matches size of weights in function
        representation) network."""
        self.weight_shapes, self.bias_shapes, self.weight_names, self.bias_names = self.function_representation.get_weight_shapes()

        # Calculate output dimension
        self.output_dim = 0
        for i in range(len(self.weight_names)):
            self.output_dim += self.weight_shapes[i][0] * self.weight_shapes[i][1]
        for i in range(len(self.bias_names)):
            self.output_dim += self.bias_shapes[i][0]

        # Calculate partition of output of network, so that output network can
        # be reshaped into weights of the function representation network
        # Partition first part of output into weight matrices
        start_index = 0
        self.weight_partition = []
        for i in range(len(self.weight_shapes)):
            weight_size = self.weight_shapes[i][0] * self.weight_shapes[i][1]
            self.weight_partition.append((start_index, start_index + weight_size))
            start_index += weight_size

        # Partition second part of output into bias matrices
        self.bias_partition = []
        for i in range(len(self.bias_shapes)):
            bias_size = self.bias_shapes[i][0]
            self.bias_partition.append((start_index, start_index + bias_size))
            start_index += bias_size

    def _init_neural_net(self):
        """Initializes weights of hypernetwork."""

        c_dim_list = [*self.c_dim_list]
        if self.coords_dim > 0:
            c_dim_list[0] += self.coords_dim
        decoder_params = {'c_dim_list': self.c_dim_list,
                            'act': self.act,
                            'batchnorm': False,
                            'dropout': self.dropout,
                            'l2norm': False,
                            'dims': None}
        # self.mlp = MLP(c_dim_list=c_dim_list,
        #                act=self.act,
        #                batchnorm=False,
        #                dropout=self.dropout,
        #                l2norm=False,
        #                dims=None)
        
        self.function_representation['batchnorm']=False
        self.function_representation['l2norm']=False
        self.function_representation['dims']=None
                          
        self.hypernetwork = MLPInr(inr_params = self.function_representation, decoder_params=decoder_params)
    
    def _init_transformer_net(self):
        """Initializes weights of hypernetwork."""
        self.transfomer_cfg = {'dim': self.c_dim_list[0]//self.n_patches, 
                               'depth': self.depth, 'n_head': self.n_heads,
                                'head_dim': self.head_dim, 'ff_dim': self.ff_dim,
                                'dropout': self.dropout}# {"args" :{'dim': self.c_dim_list[0], 'depth': self.depth, 'n_head': self.n_head, 'head_dim': self.head_dim, 'ff_dim': self.ff_dim, 'dropout': self.dropout}}

        
        self.function_representation['batchnorm']=False
        self.function_representation['l2norm']=False
        self.function_representation['dims']=None


        self.hypernetwork = TransInr(
            inr_params = self.function_representation,
            decoder_params=self.transfomer_cfg,
            n_groups = self.n_groups,
            n_patches= self.n_patches,
            agg = self.agg,
            decoder_norm = self.decoder_norm
        )        
    def output_to_weights(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.

        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).

        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        all_weights = {}
        all_biases = {}
        # Compute weights and biases separately for each element in batch
        for i in range(output.shape[0]):
            weights = []
            biases = []
            # Add weight matrices
            for j, (start_index, end_index) in enumerate(self.weight_partition):
                weight = output[i, start_index:end_index]
                weights.append(weight.view(*self.weight_shapes[j]))
            # Add bias vectors
            for j, (start_index, end_index) in enumerate(self.bias_partition):
                bias = output[i, start_index:end_index]
                biases.append(bias.view(*self.bias_shapes[j]))
            # Add weights and biases for this function representation to batch
            all_weights[i] = weights
            all_biases[i] = biases
        return all_weights, all_biases

    def output_to_weights_2(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.

        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).

        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        # Compute weights and biases separately for each element in batch
        bs = output.shape[0]
        weights = []
        biases = []
        ws_and_bs= []

        for j in range(len(self.weight_partition)):
            start_index, end_index = self.weight_partition[j]
            weight = output[:, start_index:end_index]
            weights.append(weight.reshape((bs,*self.weight_shapes[j])))

        for j in range(len(self.bias_partition)):
            start_index, end_index = self.bias_partition[j]
            bias = output[:, start_index:end_index]
            biases.append(bias.reshape((bs,*self.bias_shapes[j])))
    
        return weights, biases
    
    def forward(self, latents, coords=None):
        """Compute weights of function representations from latent vectors.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim).
            coord (torch.Tensor): Shape (batch_size, coord_dim).
        """
        # if self.coords_dim > 0:
        #     # print("input z and coordinates are concat")
        #     input_ = torch.cat((latents, coords), dim=1)

        
        input_ = latents
        
        output =self.hypernetwork(input_) # 
        # output_reshaped = self.output_to_weights(output)
        return output