import torch
import torch.nn as nn


class MHABlock(nn.Module):
    """Multi-Head self-Attention block with pre-norm and residual connection."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (N, T, embed_dim)
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        return residual + x


class AttentionBlock(nn.Module):
    """Wrapper that reshapes higher-D tensors down to 3-D for MHA."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = MHABlock(embed_dim, num_heads, dropout)

    def forward(self, x):
        # x: (N, T, embed_dim)  — or higher-D, collapsed along trailing dims
        in_shape = x.shape
        if x.dim() > 3:
            x = x.reshape(in_shape[0], in_shape[1], -1)
        out = self.mha(x)
        if len(in_shape) > 3:
            out = out.reshape(in_shape)
        return out
