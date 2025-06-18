"""
CMV (Change My View) 數據集載入器
用於 GNN 訓練
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from pathlib import Path
import sys

# 添加父目錄到路徑
sys.path.append(str(Path(__file__).parent))

from build_graph import build_graph

# 特徵維度（使用 BERT 嵌入維度）
NUM_FEATURES = 768

def get_pyg_data(pairs_path='data/raw/pairs.jsonl'):
    """
    載入 CMV 數據並轉換為 PyTorch Geometric 格式
    
    Returns:
        data: PyG Data 對象
        G: NetworkX 圖對象
    """
    print("構建社交網絡圖...")
    G = build_graph(pairs_path)
    print(f"圖構建完成: {G.number_of_nodes()} 個節點, {G.number_of_edges()} 條邊")
    
    # 轉換為 PyG 格式
    print("轉換為 PyTorch Geometric 格式...")
    
    # 為每個節點創建特徵向量
    node_features = []
    node_mapping = {}
    
    for idx, (node, attrs) in enumerate(G.nodes(data=True)):
        node_mapping[node] = idx
        
        # 創建節點特徵（基於統計信息）
        features = np.zeros(NUM_FEATURES)
        
        # 使用節點屬性作為特徵的一部分
        features[0] = attrs.get('average_body_len', 0.0) / 1000.0  # 標準化
        features[1] = attrs.get('std_body_len', 0.0) / 1000.0
        features[2] = attrs.get('avg_score', 0.0) / 100.0
        features[3] = attrs.get('success_rate', 0.0)
        features[4] = attrs.get('replies', 0.0) / 100.0
        features[5] = attrs.get('success_count', 0.0) / 10.0
        
        # 添加一些隨機性（模擬文本嵌入）
        np.random.seed(hash(node) % 2**32)
        features[6:] = np.random.randn(NUM_FEATURES - 6) * 0.1
        
        node_features.append(features)
    
    # 創建邊索引
    edge_index = []
    edge_attr = []
    
    for u, v, attrs in G.edges(data=True):
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        
        # 無向圖，添加雙向邊
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])
        
        weight = attrs.get('weight', 1.0)
        edge_attr.append(weight)
        edge_attr.append(weight)
    
    # 轉換為張量
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    
    # 創建 PyG Data 對象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    print(f"PyG 數據創建完成:")
    print(f"  節點特徵: {data.x.shape}")
    print(f"  邊索引: {data.edge_index.shape}")
    print(f"  邊屬性: {data.edge_attr.shape}")
    
    return data, G

if __name__ == "__main__":
    # 測試數據載入
    data, G = get_pyg_data()
    print("\n數據載入測試成功！") 