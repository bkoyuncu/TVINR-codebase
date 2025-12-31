import torch
import torch.nn as nn
import torch.nn.functional as F
from imagegym.models.act import act_dict

class TransformerBlock(nn.Module):
    def __init__(self, c_in, num_heads, feedforward_dim, dropout=0.0, batchnorm=True, act=None):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(c_in, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(c_in) if batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(c_in, feedforward_dim),
            act,
            nn.Linear(feedforward_dim, c_in)
        )
        self.norm2 = nn.LayerNorm(c_in) if batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Attention block
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feedforward block
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    def __init__(self, latent_dim=32, num_heads=8, feedforward_dim=2048, L=8, output_dim=64,
                 act='relu', batchnorm=True, dropout=0.0, n_layers= 1, l2norm=False, **kwargs):
        super(TransformerModel, self).__init__()
        
        self.L = L  # Number of chunks
        self.dim_per_chunk = latent_dim // L  # Dimension of each chunk
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.dim_per_chunk, num_heads, feedforward_dim, dropout, batchnorm, act_dict[act])
            for _ in range(n_layers)  # Number of transformer layers
        ])
        # Linear projection to output_dim
        self.output_projection = nn.Linear(self.dim_per_chunk, output_dim)

        self.has_l2norm = l2norm

    def forward(self, z):
        batch_size = z.size(0)

        # Reshape latent vector z (batch_size, latent_dim) to (batch_size, L, dim_per_chunk)
        x = z.view(batch_size, self.L, self.dim_per_chunk)  # Treat as sequence of L chunks

        # Permute to match the input format expected by nn.MultiheadAttention (seq_len, batch_size, dim)
        x = x.permute(1, 0, 2)  # (L, batch_size, dim_per_chunk)

        # Pass through each transformer block
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Project each chunk's dimension from dim_per_chunk to output_dim
        x = self.output_projection(x)

        # (L, batch_size, output_dim) -> (batch_size, L, output_dim)
        x = x.permute(1, 0, 2)

        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=-1)

        return x.reshape(batch_size,-1) # Final shape: (batch_size, L, output_dim)
