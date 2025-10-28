import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from .tokenizer import VocabBuilder, CharacterTokenizer

class TransliterationDataset(Dataset):
    def __init__(self, csv_path, src_tokenizer: CharacterTokenizer, tgt_tokenizer: CharacterTokenizer, max_len=64):
        df = pd.read_csv(csv_path, header=None)
        # allow both tab or comma separated; expect two columns: src, tgt
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
        df = df.dropna()
        df.columns = ["src", "tgt"]
        self.pairs = df.to_records(index=False).tolist()
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def pad(self, seq, pad_idx):
        if len(seq) < self.max_len:
            return seq + [pad_idx] * (self.max_len - len(seq))
        return seq[:self.max_len]

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        s = str(s)
        t = str(t)
        s_ids = self.src_tok.encode(s)
        t_ids = self.tgt_tok.encode(t)
        s_ids = self.pad(s_ids, self.src_tok.pad_idx)
        t_ids = self.pad(t_ids, self.tgt_tok.pad_idx)
        return torch.tensor(s_ids, dtype=torch.long), torch.tensor(len(s_ids), dtype=torch.long), torch.tensor(t_ids, dtype=torch.long), torch.tensor(len(t_ids), dtype=torch.long)

def build_tokenizers_from_csv(train_csv, save_dir="data/processed"):
    df = pd.read_csv(train_csv, header=None)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df = df.dropna()
    df.columns = ["src", "tgt"]
    src_texts = df["src"].astype(str).tolist()
    tgt_texts = df["tgt"].astype(str).tolist()
    src_vocab = VocabBuilder(src_texts)
    tgt_vocab = VocabBuilder(tgt_texts)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    src_vocab.save(f"{save_dir}/src_vocab.json")
    tgt_vocab.save(f"{save_dir}/tgt_vocab.json")
    src_tok = CharacterTokenizer(src_vocab)
    tgt_tok = CharacterTokenizer(tgt_vocab)
    return src_tok, tgt_tok
