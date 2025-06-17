import torch
import torch_geometric.utils as tgu
from torch_geometric.data import InMemoryDataset
try:
    from .build_graph import build_graph
except ImportError:
    from build_graph import build_graph

NUM_FEATURES = 6

def to_pyg_data(G):
    data = tgu.from_networkx(G)
    # assemble x
    feats = []
    for _, d in G.nodes(data=True):
        f = [
            d.get('replies', 0),
            d.get('success_count', 0),
            d.get('success_rate', 0.0),
            d.get('average_body_len', 0.0),
            d.get('std_body_len', 0.0),
            d.get('avg_score', 0.0),
        ]
        feats.append(torch.tensor(f, dtype=torch.float))
    data.x = torch.stack(feats)
    # edge weight
    data.edge_weight = data.weight.float()
    return data

def get_pyg_data(pairs_path='data/raw/pairs.jsonl'):
    """構建圖並轉換為 PyTorch Geometric 格式"""
    G = build_graph(pairs_path)
    data = to_pyg_data(G)
    return data, G