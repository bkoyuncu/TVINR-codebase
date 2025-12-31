import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
# import models
from imagegym.models.layer.transformer_encoder_new import TransformerEncoder
from imagegym.models.layer.our_mlp import MLP

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

class TransInr(nn.Module):

    def __init__(self, inr_params, n_groups, decoder_params, n_patches=1, agg='repeat', decoder_norm = []):
        super().__init__()
        self.dim = decoder_params['dim']
        self.n_patches = n_patches
        self.n_groups = n_groups
        self.agg = agg
        self.decoder_norm = decoder_norm

        self.inr = MLP(**inr_params)
        self.decoder = TransformerEncoder(**decoder_params)
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.inr.param_shapes.items(): 
            if name in self.inr.shared_layer_names:
                continue
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(self.dim) if 'linear' in decoder_norm else nn.Identity(),
                nn.Linear(self.dim, shape[0]),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, self.dim))

        # if self.agg == 'repeat':
        #     self.posemb = nn.Parameter(torch.randn(n_wtokens, self.dim))
        if self.n_patches >= 1:
            self.posemb = nn.Parameter(torch.randn(n_patches, self.dim))


    def forward(self, data):
        dtokens = data #self.tokenizer(data) #[2, 400, 768]) (bs, number of patches, dim)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B) # (B, n_wtokens, dim) 259 for weight tokens
        if self.agg == 'repeat':
            dtokens = einops.repeat(dtokens, 'b d -> b n d', n=self.wtokens.shape[0]) # (B, n_wtokens, dim)
            # dtokens = dtokens + self.posemb
        elif self.agg == 'slice' and self.n_patches >= 1:
            dtokens = dtokens.reshape(B, self.n_patches, self.dim) # (B, n_patches, dim)
            dtokens = dtokens + self.posemb
        else:
            raise NotImplementedError

        trans_out = self.decoder(torch.cat([dtokens, wtokens], dim=1)) # (B, n_pathecs + n_wtokens, dim) both input and output
        trans_out = trans_out[:, -len(self.wtokens):, :] 

        params = dict()
        for name, shape in self.inr.param_shapes.items():
            if name in self.inr.shared_layer_names:
                continue
            w = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            out = w * x.repeat(1, 1, w.shape[2] // x.shape[2])
            if 'w' in name and 'output' in self.decoder_norm:
                out = F.normalize(out, dim=1)
            params[name] = out
        return params
        self.hyponet.set_params(params)
        return self.hyponet
