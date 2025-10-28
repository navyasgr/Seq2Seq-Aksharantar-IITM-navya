import argparse, yaml, os, random, time
import torch, json
from torch import optim
from torch.utils.data import DataLoader
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.data.dataset import TransliterationDataset, build_tokenizers_from_csv
from src.utils.training_loop import train_epoch, eval_epoch

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(cfg.get('seed',1337)); torch.manual_seed(cfg.get('seed',1337))
    train_csv = cfg['data'].get('train_path')
    val_csv = cfg['data'].get('val_path')
    # build tokenizers
    src_tok, tgt_tok = build_tokenizers_from_csv(train_csv, save_dir='data/processed')
    # datasets
    train_ds = TransliterationDataset(train_csv, src_tok, tgt_tok, max_len=cfg['data'].get('max_len',64))
    val_ds = TransliterationDataset(val_csv, src_tok, tgt_tok, max_len=cfg['data'].get('max_len',64))
    train_loader = DataLoader(train_ds, batch_size=cfg['training'].get('batch_size',32), shuffle=True, collate_fn=None)
    val_loader = DataLoader(val_ds, batch_size=cfg['training'].get('batch_size',32), collate_fn=None)
    encoder = Encoder(input_dim=len(src_tok.vocab.itos), embed_dim=cfg['model']['embed_dim'], hidden_dim=cfg['model']['hidden_dim'], rnn_type=cfg['model'].get('rnn_type','LSTM'), num_layers=cfg['model'].get('num_layers',1), dropout=cfg['model'].get('dropout',0.1), pad_idx=src_tok.pad_idx)
    decoder = Decoder(output_dim=len(tgt_tok.vocab.itos), embed_dim=cfg['model']['embed_dim'], hidden_dim=cfg['model']['hidden_dim'], rnn_type=cfg['model'].get('rnn_type','LSTM'), num_layers=cfg['model'].get('num_layers',1), dropout=cfg['model'].get('dropout',0.1), pad_idx=tgt_tok.pad_idx, use_attention=cfg['model'].get('use_attention',True), attn_dim=cfg['model'].get('attn_dim',128))
    model = Seq2Seq(encoder, decoder, sos_idx=tgt_tok.sos_idx, eos_idx=tgt_tok.eos_idx, device=str(device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training'].get('lr',1e-3))
    out_dir = args.out_dir or 'checkpoints'
    os.makedirs(out_dir, exist_ok=True)
    best_cer = float('inf')
    epochs = cfg['training'].get('epochs',3)
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, pad_idx=src_tok.pad_idx, teacher_forcing_ratio=cfg['training'].get('teacher_forcing',0.5))
        val_loss, examples = eval_epoch(model, val_loader, device, pad_idx=src_tok.pad_idx, src_tok=src_tok, tgt_tok=tgt_tok)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={elapsed:.1f}s")
        # save checkpoint
        ckpt = {'epoch': epoch+1, 'model_state': model.state_dict(), 'cfg': cfg}
        torch.save(ckpt, os.path.join(out_dir, f'ckpt_epoch{epoch+1}.pt'))
        # compute CER on collected examples
        from src.utils.metrics import cer as cer_fn
        val_cer = 0.0; n=0
        for s,p,t in examples:
            val_cer += cer_fn(p,t); n+=1
        if n>0:
            val_cer = val_cer / n
            print("Sample val CER (approx):", val_cer)
            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(ckpt, os.path.join(out_dir, 'best.pt'))
    print("Training finished. Best sample CER:", best_cer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    parser.add_argument('--out_dir', default='checkpoints')
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()
    main(args)
