from attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(emb_dim)) 
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * norm_x + self.bias

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])  
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])  
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg["emb_dim"])  
        self.attn = MultiHeadAttention(  
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ln_2 = LayerNorm(cfg["emb_dim"])  
        self.mlp = FeedForward(cfg)  
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Self-attention
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = shortcut + x

        # Feedforward
        shortcut = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.drop_resid(x)
        return shortcut + x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.wpe = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.h = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.ln_f = LayerNorm(cfg["emb_dim"])  

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        pos = torch.arange(seq_len, device=in_idx.device).unsqueeze(0)
        tok_emb = self.wte(in_idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:  # Updated
            x = block(x)
            
        x = self.ln_f(x)  # Updated
        # logits = torch.matmul(x, self.wte.weight.T)  # Weight tying
        logits = x @ self.wte.weight.T  # Weight tying
        return logits



