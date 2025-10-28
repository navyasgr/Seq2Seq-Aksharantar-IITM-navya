# ==============================================================
# TRAIN.PY — Seq2Seq Transliteration Model (LSTM + Attention)
# IIT Madras — Aksharantar Project
# ==============================================================

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# ==============================================================
# 1️⃣  DEVICE SETUP
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ==============================================================
# 2️⃣  SEED FIXING FOR REPRODUCIBILITY
# ==============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==============================================================
# 3️⃣  HYPERPARAMETERS
# ==============================================================
INPUT_DIM = 100   # size of source vocabulary
OUTPUT_DIM = 100  # size of target vocabulary
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 15
BATCH_SIZE = 32

# ==============================================================
# 4️⃣  DEFINE ATTENTION, ENCODER, DECODER, SEQ2SEQ CLASSES
# ==============================================================

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs.transpose(0,1)), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # ✅ FIX: input size should be (emb_dim + hid_dim), not (hid_dim*2 + emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim)
        
        # ✅ Output combines output + weighted + embedded
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        # attention over encoder outputs
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs.transpose(0, 1))  # [batch_size, 1, hid_dim]
        weighted = weighted.transpose(0, 1)  # [1, batch_size, hid_dim]

        # ✅ concat: emb_dim + hid_dim = 256 + 512 = 768
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, 768]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # output layer
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(0))

        return prediction, hidden, cell, a


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]  # first token <sos>

        for t in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs

# ==============================================================
# 5️⃣  INITIALIZE MODEL, OPTIMIZER, LOSS
# ==============================================================

attn = BahdanauAttention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ==============================================================
# 6️⃣  DUMMY DATASET (for demonstration)
# ==============================================================

class DummyDataset(Dataset):
    def __init__(self, n=1000):
        self.src = torch.randint(1, INPUT_DIM, (n, 10))
        self.trg = torch.randint(1, OUTPUT_DIM, (n, 10))
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

train_loader = DataLoader(DummyDataset(1000), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DummyDataset(200), batch_size=BATCH_SIZE)

# ==============================================================
# 7️⃣  TRAINING FUNCTION
# ==============================================================

def train(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for src, trg in iterator:
        src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        acc = (output.argmax(1) == trg).float().mean().item()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
            output = model(src, trg, 0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            acc = (output.argmax(1) == trg).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ==============================================================
# 8️⃣  TRAINING LOOP WITH VISUALIZATION
# ==============================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/visualizations", exist_ok=True)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("💾 Saved best model!")

# ==============================================================
# 9️⃣  SAVE VISUALIZATION OUTPUTS
# ==============================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("checkpoints/visualizations/loss_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("checkpoints/visualizations/accuracy_curve.png")
plt.close()

print("✅ Visualization saved at checkpoints/visualizations/")
print("🎯 Training completed successfully!")
