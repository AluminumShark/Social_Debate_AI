from torch import load
from pathlib import Path

_ckpt = load(str(Path('data/models/gnn_social.pt')), map_location='cpu')
_EMB = _ckpt['emb']; _IDX = _ckpt['node2idx']
def social_encoder(author, k=8):
    idx = _IDX.get(author)
    return "UNK" if idx is None else " ".join(f"{x:.4f}" for x in _EMB[idx][:k])