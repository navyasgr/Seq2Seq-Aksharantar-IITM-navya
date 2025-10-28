from typing import Optional
import torch, random, torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, sos_idx:int, eos_idx:int, device:str='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def _build_encoder_mask(self, src, src_lens):
        if src_lens is None:
            return torch.ones(src.size(0), src.size(1), dtype=torch.bool, device=src.device)
        ranges = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        return ranges < src_lens.unsqueeze(1)

    def forward(self, src, src_lens=None, trg=None, teacher_forcing_ratio:float=0.5, max_len:Optional[int]=None):
        batch_size = src.size(0)
        enc_outputs, enc_hidden = self.encoder(src, src_lens)
        enc_mask = self._build_encoder_mask(src, src_lens)
        dec_hidden = enc_hidden
        if trg is not None:
            T_out = trg.size(1)
        else:
            if max_len is None:
                raise ValueError("Provide trg or max_len")
            T_out = max_len
        V = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, T_out, V, device=self.device)
        input_tokens = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
        for t in range(T_out):
            try:
                logits, dec_hidden, attn = self.decoder(input_tokens, dec_hidden, encoder_outputs=enc_outputs, encoder_mask=enc_mask)
            except TypeError:
                logits, dec_hidden = self.decoder(input_tokens, dec_hidden)
                attn = None
            outputs[:, t, :] = logits
            if trg is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    input_tokens = trg[:, t].to(self.device)
                else:
                    input_tokens = logits.argmax(dim=1)
            else:
                input_tokens = logits.argmax(dim=1)
        return outputs
