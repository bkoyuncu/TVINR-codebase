import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imagegym.models.layer.our_mlp import MLP

def deactivate(model, layer_indices):
    for idx, layer in enumerate(model.layers):
        if idx in layer_indices:
            for module in layer.modules():# print(module)
                if isinstance(module, (nn.Linear, nn.BatchNorm1d)): #lets check this  #TODO
                    for param in module.parameters():
                        param.requires_grad = False



def process_fnrep_params(encoding, feature_dim , fnrep_params):
    assert 'dim_inner' in fnrep_params
    assert 'num_layers' in fnrep_params
    c_dim_list = [encoding.feature_dim]
    dim_inner = fnrep_params['dim_inner']
    layers_encoder = fnrep_params['num_layers']
    c_dim_list.extend([dim_inner for i in range(layers_encoder)])
    c_dim_list.append(feature_dim)
    fnrep_params['c_dim_list'] = c_dim_list
    fnrep_params['encoding'] = encoding
    fnrep_params['shared_layer_indx'] = fnrep_params['shared_layer_indx']
    fnrep_params['mode'] = fnrep_params['mode']

class FunctionRepresentation(nn.Module):
    """Function to represent a single datapoint. For example this could be a
    function that takes pixel coordinates as input and returns RGB values, i.e.
    f(x, y) = (r, g, b).

    Args:
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """

    def __init__(self, c_dim_list,
                 encoding,
                 dropout=0.0,
                 act='prelu',
                 shared_layer_indx=[],
                 mode = "wb",
                 **kwargs):
        super(FunctionRepresentation, self).__init__()

        assert act not in ['prelu']
        self.c_dim_list = c_dim_list
        self.dropout = dropout
        self.act = act
        self.encoding = encoding

        if not isinstance(self.encoding, nn.Identity):
            assert self.encoding.feature_dim == c_dim_list[0]

        self.mlp = MLP(c_dim_list,
                       act=act,
                       batchnorm=False,
                       dropout=dropout,
                       l2norm=False,
                        shared_layers=shared_layer_indx,
                       dims=None)

        list_of_layers = np.arange(self.mlp.n_layers)  # List of layers
        self.mlp_shared_layers = self.mlp.shared_layers  # Shared layers
        self.mlp_non_shared_layers = self.mlp.non_shared_layers  # Non-shared layers

        # Assertion
        assert all(layer in list_of_layers for layer in self.mlp_shared_layers), "Not all shared layers are in list_of_layers"
        assert all(layer in list_of_layers for layer in self.mlp_non_shared_layers), "Not all non-shared layers are in list_of_layers"

        self.mlp_layers = self.mlp.all_layers
        self.layer_names = [n for n, p in self.mlp.named_parameters()]
        self.shared_layer_names = []
        self.mode = mode

        for name in self.layer_names:
            layer_indx = int(name.split('.')[1])
            if layer_indx in self.mlp_shared_layers:
                if '1.weight' not in name and '1.bias' not in name:
                    if 'weight' in name and "w" in self.mode:
                        self.shared_layer_names.append(name)
                    if 'bias' in name and "b" in self.mode:
                        self.shared_layer_names.append(name)

        self.non_shared_layer_names = [name for name in self.layer_names if name not in self.shared_layer_names]
        print(self.shared_layer_names)



        # deactivate(self.mlp, self.mlp_non_shared_layers)
    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        weight_names = []
        bias_names = []
        
        # Iterate over named parameters to also get the name of each parameter
        # for name, param in self.mlp.named_parameters():
        for name,param in self.mlp.named_parameters():
            print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")
            # Check if this parameter belongs to a batch normalization layer
            if name in self.non_shared_layer_names:
                if '1.weight' not in name and '1.bias' not in name:
                    if len(param.shape) == 1 and "bias" in name:
                        bias_shapes.append(param.shape)
                        bias_names.append(name)
                    if len(param.shape) == 2 and "weight" in name:
                        weight_shapes.append(param.shape)
                        weight_names.append(name)
        return weight_shapes, bias_shapes, weight_names, bias_names
    

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                weights.append(param)
            elif 'bias' in name:
                biases.append(param)
        return weights, biases
    
    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.

        Args:
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):

        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.mlp.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def duplicate(self):
        """Returns a FunctionRepresentation instance with random weights."""
        # Extract device
        device = next(self.parameters()).device
        # Create new function representation and put it on same device
        return FunctionRepresentation(c_dim_list=self.c_dim_list,
                                      encoding=self.encoding,
                                      dropout=self.dropout,
                                      act=self.act).to(device)


    # def batch_forward(self, coordinates, weights, biases):
    #     """Stateless forward pass for multiple function representations.

    #     Args:
    #         coordinates (torch.Tensor): Batch of coordinates of shape
    #             (batch_size, num_points, coordinate_dim).
    #         weights (dict of list of torch.Tensor): Batch of list of tensors
    #             containing weights of linear layers for each neural network.
    #         biases (dict of list of torch.Tensor): Batch of list of tensors
    #             containing biases of linear layers for each neural network.

    #     Return:
    #         Returns a tensor of shape (batch_size, num_points, feature_dim).
    #     """
    #     assert  len(coordinates.shape) == 3
    #     features = []
    #     bs = coordinates.shape[0]

    #     for i in range(bs):
    #         self.set_weights_and_biases(weights[i], biases[i])
    #         co = coordinates[i]
    #         features.append(
    #             self(co, weights[i], biases[i]).unsqueeze(0)
    #         )
    #     return torch.cat(features, dim=0)



    def stateless_forward(self, coordinates, weights, biases):

        # Apply MLP
        for i, layer in enumerate(self.mlp.layers):
            for j, module in enumerate(layer.children()):
                # if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                    # print(module)
                    # for param in module.parameters():
                    #     print(param.requires_grad)
                if j==0 and i in self.mlp_non_shared_layers:#if j == 0:  # Linear layer
                    indx = self.mlp_non_shared_layers.index(i)
                    hidden = F.linear(hidden, weights[indx], biases[indx]) #TODO we can add weight normalization here. 
                else:
                    hidden = module(hidden)
        return hidden
    
    def batch_stateless_forward(self, coordinates, all_weights, all_biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape (batch_size, num_points, coordinate_dim).
            all_weights (list of list of torch.Tensor): Batch of list of tensors containing weights for each neural network.
            all_biases (list of list of torch.Tensor): Batch of list of tensors containing biases for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """
        features = []
        bs = coordinates.shape[0]

        for i in range(bs):
            co = coordinates[i]
            features.append(
                self.stateless_forward(co, all_weights[i], all_biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)

    def position_encoding(self, coordinates):
        """Transform coordinates using the given position embedding
        Args:
            coordinates (torch.Tensor): Tensor of shape (num_points, coordinate_dim).
        Return:
            Returns a tensor of shape (num_points, feature_dim)
        """
        # Positional encoding is first layer of function representation
        # model, so apply this transformation to coordinates
        hidden = self.encoding(coordinates)
        return hidden

# from imagegym.models.layer.function_representation import FunctionRepresentation, FourierFeatures, IdentityFeatures, process_fnrep_params, FourierFeaturesT
import math
class FourierFeaturesT(nn.Module):
    """Temporal Fourier features.
    # https://github.com/bmild/nerf/issues/12 People are some cases omitting pi* in the computation
    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
        learnable_features (bool): If True, fourier features are learnable, otherwise they are fixed. 
    """

    def __init__(self, frequency_matrix, learnable_features=False, use_time=False, use_pi = True):
        super(FourierFeaturesT, self).__init__()
        self.use_time = use_time
        self.use_pi = use_pi
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        # self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

        if use_time: #add real time coordinate in the features
            self.feature_dim += frequency_matrix.shape[1] #TODO check with others

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2*math.pi * prefeatures if self.use_pi else prefeatures)
        sin_features = torch.sin(2*math.pi * prefeatures if self.use_pi else prefeatures)
        # Concatenate sine and cosine features
        features = torch.cat((sin_features, cos_features), dim=-1)
        #TODO implement
        if self.use_time:
            features = torch.cat((coordinates, features), dim=-1)

        return features
    
    def compute_periodicity(self):
        # Define the function to compute periodicity
        periodicities = []
        L =  self.frequency_matrix.shape[0]
        for i in range(L):
            if self.use_pi: #frequency gives f 
                period = 1 / self.frequency_matrix[i] 
            if not self.use_pi: #frequency gives 2*pi*f  = omega
                period = (2 * math.pi) / self.frequency_matrix[i]
            periodicities.append(period)
        return periodicities
        

class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """

    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)




class IdentityFeatures(nn.Module):
    """Itendity features.

    Args:
        coordinate_dim (int): Dimension of the coordinates
    """

    def __init__(self, coordinate_dim):
        super(IdentityFeatures, self).__init__()
        self.feature_dim = coordinate_dim

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        return  coordinates
