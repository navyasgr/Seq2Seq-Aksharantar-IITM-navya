import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, encoder_outputs, dec_hidden, mask=None):
        enc_proj = self.W_enc(encoder_outputs)
        dec_proj = self.W_dec(dec_hidden).unsqueeze(1)
        scores = torch.tanh(enc_proj + dec_proj)
        scores = self.v(scores).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
