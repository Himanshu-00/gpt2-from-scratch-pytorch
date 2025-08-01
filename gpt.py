from attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


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
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            # context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
       
        self.mlp = FeedForward(cfg)
        self.ln_1 = LayerNorm(cfg["emb_dim"])                     # rename 'norm1' -> 'ln_1'
        self.ln_2 = LayerNorm(cfg["emb_dim"])                     # rename 'norm2' -> 'ln_2'
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.ln_2(x)
        x = self.mlp(x)

        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])          # rename to 'wte'
        self.wpe = nn.Embedding(cfg["context_length"], cfg["emb_dim"])      # rename to 'wpe'
        self.drop = nn.Dropout(cfg["drop_rate"])                            # rename to 'drop'


        self.h = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # change to ModuleList, 'h'

        self.ln_f = LayerNorm(cfg["emb_dim"])                               # rename to 'ln_f'


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        pos = torch.arange(seq_len, device=in_idx.device).unsqueeze(0)
        x = self.wte(in_idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T                                      # tie output to embedding
       
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
