from typing import Optional, Tuple
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, rnn_type: str = "LSTM", num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        rnn_kwargs = dict(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        rnn_type = rnn_type.upper()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(nonlinearity="tanh", **rnn_kwargs)

    def forward(self, src, src_lens=None):
        emb = self.embedding(src)
        if src_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            outputs, hidden = self.rnn(emb)
        return outputs, hidden
