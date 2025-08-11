import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.c_attn = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)  # Renamed
        self.c_proj = nn.Linear(d_out, d_out)  # Renamed
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.c_attn(x)  # [batch, tokens, 3*emb_dim]
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, tokens, head_dim]
        queries, keys, values = qkv
        
        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        return self.c_proj(context_vec)
 