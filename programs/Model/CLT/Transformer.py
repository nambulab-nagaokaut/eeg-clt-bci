import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,drop_p):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads,dropout=drop_p, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadSelfAttention(emb_size,num_heads,drop_p=0.5),
                nn.Dropout(drop_p)
            )),
            RMSNorm(dim=emb_size),
            # nn.LayerNorm(emb_size),
            
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),    
            )),
            RMSNorm(dim=emb_size)
            # nn.LayerNorm(emb_size),
            
            )


class TransformerEncoder(nn.Sequential):
    def __init__(self,embed_dim:int, numheads:int = 4, depth:int = 6):
        """Transformer Module. 
        As described in https://arxiv.org/abs/1706.03762.
        Args:
            embed_dim (int): Total dimension of the module. (The number of features output of the LSTM Module)
            numheads (int, optional): Number of parallel attention heads. Defaults to 4.
            depth (int, optional): The number of Identical Encoder layers. Defaults to 6.
        """
        super().__init__(*[TransformerEncoderBlock(emb_size=embed_dim,num_heads=numheads) for _ in range(depth)])
