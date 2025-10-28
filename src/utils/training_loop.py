import torch, torch.nn as nn
from tqdm import tqdm

def masked_cross_entropy(logits, targets, lengths, ignore_index):
    B, T, V = logits.shape
    logits_flat = logits.view(B*T, V)
    targets_flat = targets.contiguous().view(B*T)
    loss_f = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    loss = loss_f(logits_flat, targets_flat)
    total_tokens = lengths.sum().item()
    if total_tokens == 0:
        return torch.tensor(0., device=logits.device)
    return loss / total_tokens

def train_epoch(model, dataloader, optimizer, device, pad_idx, teacher_forcing_ratio=0.5, clip=1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for src, src_lens, tgt, tgt_lens in tqdm(dataloader, leave=False):
        src = src.to(device); tgt = tgt.to(device); src_lens = src_lens.to(device); tgt_lens = tgt_lens.to(device)
        optimizer.zero_grad()
        outputs = model(src, src_lens, trg=tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        T_out = outputs.size(1)
        targets = tgt[:, :T_out]
        loss = masked_cross_entropy(outputs, targets, tgt_lens.clamp(max=T_out), ignore_index=pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_tokens += tgt_lens.clamp(max=T_out).sum().item()
    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss

def eval_epoch(model, dataloader, device, pad_idx, src_tok=None, tgt_tok=None, max_eval_samples=200):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    examples = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in tqdm(dataloader, leave=False):
            src = src.to(device); tgt = tgt.to(device); src_lens = src_lens.to(device); tgt_lens = tgt_lens.to(device)
            outputs = model(src, src_lens, trg=None, teacher_forcing_ratio=0.0, max_len=tgt.size(1))
            T_out = outputs.size(1)
            targets = tgt[:, :T_out]
            loss = masked_cross_entropy(outputs, targets, tgt_lens.clamp(max=T_out), ignore_index=pad_idx)
            total_loss += loss.item() * targets.size(0)
            total_tokens += tgt_lens.clamp(max=T_out).sum().item()
            preds = outputs.argmax(dim=-1).cpu().tolist()
            for i in range(min(len(preds), 4)):
                if len(examples) < max_eval_samples and src_tok is not None and tgt_tok is not None:
                    src_str = src_tok.decode(src[i].cpu().tolist())
                    pred_str = tgt_tok.decode(preds[i])
                    tgt_str = tgt_tok.decode(targets[i].cpu().tolist())
                    examples.append((src_str, pred_str, tgt_str))
    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, examples
