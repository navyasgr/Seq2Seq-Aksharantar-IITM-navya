import json
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class VocabBuilder:
    def __init__(self, texts=None, special_tokens=None):
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        texts = texts or []
        chars = set()
        for t in texts:
            for ch in t:
                chars.add(ch)
        # sort for deterministic ordering
        chars = sorted(list(chars))
        self.itos = list(self.special_tokens) + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vb = cls([])
        vb.itos = data["itos"]
        vb.stoi = {ch: i for i, ch in enumerate(vb.itos)}
        return vb

class CharacterTokenizer:
    def __init__(self, vocab: VocabBuilder):
        self.vocab = vocab
        self.pad_idx = self.vocab.stoi.get("<pad>")
        self.sos_idx = self.vocab.stoi.get("<sos>")
        self.eos_idx = self.vocab.stoi.get("<eos>")
        self.unk_idx = self.vocab.stoi.get("<unk>")

    def encode(self, text, add_sos_eos=True):
        ids = [self.vocab.stoi.get(ch, self.unk_idx) for ch in text]
        if add_sos_eos:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, indices, join_char=""):
        chars = []
        for i in indices:
            if i == self.eos_idx:
                break
            if i in (self.pad_idx, self.sos_idx):
                continue
            chars.append(self.vocab.itos[i])
        return join_char.join(chars)
