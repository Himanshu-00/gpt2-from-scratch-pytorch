import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # Rename to checkpoint standard names
        self.c_attn = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.c_proj = nn.Linear(d_out, d_out)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Optional: GPT checkpoints do not save mask buffer usually, so omit or add accordingly
        # self.register_buffer('bias', torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # Compute QKV in one go
        qkv = self.c_attn(x)  # shape (b, t, 3*d_out)

        # Reshape and permute to separate query, key, value
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, num_heads, num_tokens, head_dim)

        queries, keys, values = qkv[0], qkv[1], qkv[2]  # (b, n_head, seq_len, head_dim)

        # Use PyTorch causal scaled dot product attention
        # use dropout only during training as per your dropout settings
        attn_dropout_p = self.attn_dropout.p if self.training else 0.0

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            dropout_p=attn_dropout_p,
            is_causal=True
        )  # shape: (b, n_head, seq_len, head_dim)

        # Rearrange so heads are merged
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Output projection + residual dropout
        context_vec = self.resid_dropout(self.c_proj(context_vec))

        return context_vec
