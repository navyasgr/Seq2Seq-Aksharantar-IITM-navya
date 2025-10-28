from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # ✅ Corrected LSTM input dimension to match actual concatenation
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim)

        # ✅ Output combines output + weighted + embedded (same as concat in forward)
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        # attention weights
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs.transpose(0, 1))  # [batch_size, 1, hid_dim]
        weighted = weighted.transpose(0, 1)  # [1, batch_size, hid_dim]

        # ✅ concat = emb_dim + hid_dim = 256 + 512 = 768
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, 768]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # ✅ output for FC: hid_dim (output) + hid_dim (weighted) + emb_dim
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(0))

        return prediction, hidden, cell, a



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
