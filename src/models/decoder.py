from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, output_dim: int, embed_dim: int, hidden_dim: int, rnn_type: str = "LSTM", num_layers: int =1, dropout: float=0.1, pad_idx: int=0, use_attention: bool=False, attn_dim: int=128):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
        rnn_input = embed_dim + (hidden_dim if use_attention else 0)
        rnn_kwargs = dict(input_size=rnn_input, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        rnn_type = rnn_type.upper()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(nonlinearity="tanh", **rnn_kwargs)
        if use_attention:
            self.attention = BahdanauAttention(enc_dim=hidden_dim, dec_dim=hidden_dim, attn_dim=attn_dim)
        else:
            self.attention = None
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_step, hidden, encoder_outputs=None, encoder_mask=None):
        emb = self.embedding(input_step).unsqueeze(1)
        context = None
        attn = None
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("encoder_outputs required when using attention")
            if isinstance(hidden, tuple):
                dec_h = hidden[0][-1]
            else:
                dec_h = hidden[-1] if hidden.dim()==3 else hidden
            context, attn = self.attention(encoder_outputs, dec_h, mask=encoder_mask)
            emb = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.rnn(emb, hidden)
        logits = self.out(out.squeeze(1))
        return logits, hidden, attn
