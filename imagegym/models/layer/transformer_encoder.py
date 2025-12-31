import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from imagegym.config import cfg

def process_transformer_params(transformer_params=None):
    transformer_params["linear_layer_sizes"].append(cfg.model.dim_z*2)

class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation:
    https://github.com/BorgwardtLab/Set_Functions_for_Time_Series 

    TODO: adapt for correct dimensions
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        # create a timescale of all times from 0-1
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        # make a tensor to hold the time embeddings
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        # scale the timepoints according to the 0-1 scale
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        # pe = pe.cuda()
        return pe
    

def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor

    TODO: make sure dimensions selected are correct...
    """
    if mask is not None: 
        # eliminate all values learned from nonexistant timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
        datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value

    maxed = torch.max(datatensor, 1)[0]

    return maxed 

def last_hidden_pooling(datatensor, mask):
    """
    select final weighted time point embedding as z
    Usage: used for causal models only (use masked_max_pooling for imputation)

    TODO: make sure dimensions selected are correct...
    TODO: make sure no mask case functions
    """
    if mask is not None: 
        last_valid_indices = mask.sum(dim=1).long() - 1
    else: 
        last_valid_indices = -1
    last_hidden = datatensor[torch.arange(datatensor.shape[0]), last_valid_indices]
    return last_hidden



class TransformerEncoder(nn.Module):
    """
    Args:
        layer_configs (list of dicts): List of dictionaries, each specifying a pointconv
            layer. Must contain keys "out_channels", "num_output_points", "num_neighbors"
            and "mid_channels" if PointConv layer. Should *not* contain key "in_channels" 
            as this is predetermined. If AvgPool layer, should not contain key 
            "out_channels".
        linear_layer_sizes (list of ints): Specify size of hidden layers in linear layers
            applied after pointconv. Note the last element of this list must be 1 (since
            discriminator outputs a single scalar value).
        
        (create timepoint embeddings, add positional encoding (based on coordinates), perform attention between timepoints, 
        pool into a single "z" representation of whatever size he'd like)
    """
    def __init__(self, coordinate_dim, feature_dim, d_model,
                 attention_layers=1, non_linearity=nn.LeakyReLU(0.2), add_sigmoid=True, 
                 n_heads=1, causal=False, dropout=0.0, pos_encoder=None, max_len=500, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.add_sigmoid = add_sigmoid
        self.causal = causal
        self.num_layers = attention_layers

        self.same_coordinates = None # had to add this to debug, but there might be a mistake in other files by carrying this around?

        self.non_linearity = non_linearity
        self.d_model = d_model # number of input features
        self.n_heads = n_heads # number of heads
        self.dim_feedforward = d_model*4 # size of feedforward network model

        self.embedding = nn.Linear(in_features=self.feature_dim *2, out_features=self.d_model)
        # self.embedding = nn.LazyLinear(self.d_model) # create d-model-sized embedding
        self.max_len = max_len
                
        # self.pos_encoder = pos_encoder
        self.pos_encoder = PositionalEncodingTF(self.d_model, self.max_len) # create positional encoding
        self.layer = torch.nn.TransformerEncoderLayer(self.d_model, n_heads, self.dim_feedforward, dropout=dropout, activation=non_linearity, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.layer, num_layers=self.num_layers)
        self.pooling = masked_max_pooling if not self.causal else last_hidden_pooling

        self.linear_layer_sizes = kwargs['linear_layer_sizes']
        
        if len(self.linear_layer_sizes):
            prev_num_units = self.d_model
            linear_layers = []
            for i, num_units in enumerate(self.linear_layer_sizes):
                linear_layers.append(nn.Linear(prev_num_units, num_units))
                # If not last layer, apply non linearity to features
                if i != len(self.linear_layer_sizes) - 1:
                    linear_layers.append(non_linearity)
                prev_num_units = num_units
            self.linear_layers = nn.Sequential(*linear_layers)
        else:
            self.linear_layers = nn.Identity()
        if len(self.linear_layer_sizes):
            self.output_dim = self.linear_layer_sizes[-1]


        # self.dense_output = torch.nn.Sequential(
        #     torch.nn.LazyLinear(self.d_model *2),
        #     non_linearity,
        #     torch.nn.LazyLinear(cfg.model.dim_z *2) # question: is feature dim the desired output size? #TODO input
        # )
        

    def forward(self, x_h, t_h, observed_mask, nan_mask):
        """
        Args:
            Shape of x: [bs, 1, dim_x, T]
            Shape of t: [bs, 2, dim_x, T]

            Shape of nan_mask: [bs, T, dim_x] indicates nan values
            Shape of observed_mask: [bs, T, dim_x] indicates observed values (we may not observe all values at all times) includes nan values as not observed (nans are already in this mask)

            Motivation: NaN values are always inherently missing, but observed values may be missing due to sensor failure or other reasons or simply for robust training

            coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim). (N, T, D)??
            features (torch.Tensor): Shape (batch_size, num_points, in_channels). (N, T, D)???

        
        TODO: fix lazy linear because parameter counts breaks 
        TODO: output mu, log var (how? currently just duplicating output)
        Question: is there an indicator mask somewhere? (have made indicator mask below in case no)
        Question: is implementation is correct/attention is done as # expects?

        """
        # reshape representations
        #batch_size_x, psudo_i_x, d_x, t_x = x_h.shape
        _, _, _, t = t_h.shape
        x_h = x_h.squeeze(1)
        
        observed_mask = observed_mask.to(x_h.device)
        nan_mask = nan_mask.to(x_h.device)
        observed_mask = observed_mask.reshape(*x_h.shape).permute(0,2,1)


        x_h = x_h.permute(0,2,1) # [bs, T, dim_x]
        x_h = torch.nan_to_num(x_h) #makes nan to 0
        x_h[~observed_mask] = 0 # makes missing values 0
        x_h = torch.concat([x_h, observed_mask], -1) # add indicator mask to channels
        t_h_sliced = t_h[0, 1, 0, :].view(1,t) # they should have the same times for every sensor, yes? yes

        # make embedding
        data_embedding = self.embedding(x_h)

        # add positional encoding
        times_vector = t_h_sliced #not normalized
        pe = self.pos_encoder(times_vector).to(data_embedding.device)
        # pe = self.pos_encoder(t_h[:,:,0].permute(0,2,1)).to(data_embedding.device)
        data_embedding_pe = torch.add(data_embedding, pe)

        if self.causal:
            mask = torch.triu(torch.ones(t, t, device=data_embedding.device) * float('-inf'), diagonal=1)
        else:
            mask = None

        # make timepoint mask
        timepoint_mask = observed_mask.sum(-1) # make missing values 1 and present values 0 with TORCH.WHERE
        timepoint_mask = torch.where(timepoint_mask == 0, True, False)

        # perform attention 
        # print(f"data_embedding_pe before: {data_embedding_pe}")
        data_embedding_pe_output = self.encoder(data_embedding_pe, mask=mask, src_key_padding_mask=timepoint_mask) 
        # print(f"data_embedding_pe after attention: {data_embedding_pe_output}")

        # pool for z
        features = self.pooling(data_embedding_pe_output, mask=None)

        # final linear layer/activation fxn?
        features = self.linear_layers(features)
        # print(features.shape)
        # print(features)
        #features = features.reshape(batch_size, t, v, 2)

        if self.add_sigmoid:
            return torch.sigmoid(features)
        else:
            return features