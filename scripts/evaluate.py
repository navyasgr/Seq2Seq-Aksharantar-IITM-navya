import argparse, yaml, os, json, torch
from torch.utils.data import DataLoader
from src.data.dataset import TransliterationDataset, build_tokenizers_from_csv
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build tokenizers from processed if present
    proc = 'data/processed'
    with open(os.path.join(proc,'src_itos.json'),'r',encoding='utf-8') as f:
        src_itos = json.load(f)
    with open(os.path.join(proc,'tgt_itos.json'),'r',encoding='utf-8') as f:
        tgt_itos = json.load(f)
    from src.data.tokenizer import VocabBuilder, CharacterTokenizer
    src_vocab = VocabBuilder([]); src_vocab.itos = src_itos; src_vocab.stoi = {c:i for i,c in enumerate(src_itos)}
    tgt_vocab = VocabBuilder([]); tgt_vocab.itos = tgt_itos; tgt_vocab.stoi = {c:i for i,c in enumerate(tgt_itos)}
    src_tok = CharacterTokenizer(src_vocab); tgt_tok = CharacterTokenizer(tgt_vocab)
    test_csv = cfg['data'].get('test_path')
    ds = TransliterationDataset(test_csv, src_tok, tgt_tok, max_len=cfg['data'].get('max_len',64))
    loader = DataLoader(ds, batch_size=32)
    # load model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_used = ckpt.get('cfg', cfg)
    encoder = Encoder(input_dim=len(src_tok.vocab.itos), embed_dim=cfg_used['model']['embed_dim'], hidden_dim=cfg_used['model']['hidden_dim'], rnn_type=cfg_used['model'].get('rnn_type','LSTM'), num_layers=cfg_used['model'].get('num_layers',1))
    decoder = Decoder(output_dim=len(tgt_tok.vocab.itos), embed_dim=cfg_used['model']['embed_dim'], hidden_dim=cfg_used['model']['hidden_dim'], rnn_type=cfg_used['model'].get('rnn_type','LSTM'), num_layers=cfg_used['model'].get('num_layers',1), use_attention=cfg_used['model'].get('use_attention',True))
    model = Seq2Seq(encoder, decoder, sos_idx=tgt_tok.sos_idx, eos_idx=tgt_tok.eos_idx, device=str(device))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    # run inference and print a few examples
    with torch.no_grad():
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
            outputs = model(src, src_lens, trg=None, teacher_forcing_ratio=0.0, max_len=tgt.size(1))
            preds = outputs.argmax(dim=-1).cpu().tolist()
            for j in range(min(5, len(preds))):
                print('SRC:', src[i][j].tolist())
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args)
