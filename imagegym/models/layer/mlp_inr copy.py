import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from imagegym.models.layer.our_mlp import MLP

class MLPInr(nn.Module):
    def __init__(self, inr_params, decoder_params):
        super().__init__()
        self.inr = MLP(**inr_params)
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_shapes = dict()
        self.wtoken_rng = dict()
        for name, shape in self.inr.param_shapes.items(): 
            if name in self.inr.shared_layer_names:
                continue
            g = shape[0]*shape[1]
            self.wtoken_shapes[name] = (shape)
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        #get weight shapes aka wtoken_rng total
        output_shape = n_wtokens
        decoder_params['c_dim_list'] = decoder_params['c_dim_list'] + [output_shape]
        self.decoder = MLP(**decoder_params)

    def forward(self, data):
        dtokens = data 
        trans_out = self.decoder(dtokens)
        params = dict()
        for name, shape in self.inr.param_shapes.items():
            if name in self.inr.shared_layer_names:
                continue
            l, r = self.wtoken_rng[name]
            x = trans_out[:, l: r]
            params[name] = x.reshape(-1, *self.wtoken_shapes[name])
        return params
        self.hyponet.set_params(params)
        return self.hyponet
