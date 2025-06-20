"""
GNN 社會編碼器
提供社會向量編碼功能和說服力預測
"""

from torch import load
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn

# 重新定義模型架構（與訓練時一致）
class PersuasionGNN(nn.Module):
    """說服力預測 GNN 模型"""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_strategies=4):
        super().__init__()
        
        # 圖卷積層（使用 GraphSAGE）
        self.conv1 = tgnn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = tgnn.SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = tgnn.SAGEConv(hidden_dim, hidden_dim // 2)
        
        # 注意力機制
        self.attention = tgnn.GATConv(hidden_dim // 2, hidden_dim // 2, heads=4, concat=False)
        
        # 任務特定的預測頭
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # 二元分類：是否為 delta
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # 回歸：說服力分數
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_strategies)  # 多分類：最佳策略
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # 圖卷積
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # 注意力層
        x = self.attention(x, edge_index)
        x = self.dropout(x)
        
        # 如果有批次信息，進行全局池化
        if batch is not None:
            x = tgnn.global_mean_pool(x, batch)
        
        # 多任務預測
        delta_pred = self.delta_head(x)
        quality_pred = self.quality_head(x)
        strategy_pred = self.strategy_head(x)
        
        return {
            'delta': delta_pred,
            'quality': quality_pred,
            'strategy': strategy_pred,
            'embeddings': x
        }

# 載入預訓練模型
_ckpt_path = Path('data/models/gnn_social.pt')
_persuasion_path = Path('data/models/gnn_persuasion.pt')

# 優先使用監督式模型
if _persuasion_path.exists():
    print("✅ 載入監督式 GNN 模型")
    _persuasion_ckpt = load(str(_persuasion_path), map_location='cpu')
    _PERSUASION_MODEL = PersuasionGNN(
        input_dim=_persuasion_ckpt['config']['input_dim'],
        hidden_dim=_persuasion_ckpt['config']['hidden_dim']
    )
    _PERSUASION_MODEL.load_state_dict(_persuasion_ckpt['model_state'])
    _PERSUASION_MODEL.eval()
    _NODE_TO_IDX = _persuasion_ckpt['node_to_idx']
    print(f"  模型性能: Delta 準確率={_persuasion_ckpt['performance']['delta_acc']:.3f}")
else:
    _PERSUASION_MODEL = None
    _NODE_TO_IDX = None

# 舊模型作為備份（可選）
if _ckpt_path.exists():
    _ckpt = load(str(_ckpt_path), map_location='cpu')
    _EMB = _ckpt['emb']
    _IDX = _ckpt['node2idx']
else:
    # 舊模型不存在，但如果有監督式模型就沒關係
    _EMB = None
    _IDX = None

# 策略名稱映射
STRATEGY_NAMES = {
    0: 'aggressive',
    1: 'defensive',
    2: 'analytical',
    3: 'empathetic'
}

def social_encoder(author, k=8):
    """獲取作者的社會編碼（字串格式）"""
    if _IDX is None or _EMB is None:
        return "UNK"
    
    idx = _IDX.get(author)
    return "UNK" if idx is None else " ".join(f"{x:.4f}" for x in _EMB[idx][:k])

def social_vec(agent_id: str) -> list:
    """
    獲取 Agent 的社會向量
    
    Args:
        agent_id: Agent 識別符（如 'Agent_A'）
        
    Returns:
        128 維的社會向量
    """
    # 如果有監督式模型，使用其嵌入
    if _PERSUASION_MODEL is not None and _NODE_TO_IDX is not None:
        idx = _NODE_TO_IDX.get(agent_id)
        if idx is not None:
            # 使用模型的嵌入層輸出
            with torch.no_grad():
                # 創建虛擬輸入（單節點）
                x = torch.zeros(1, _PERSUASION_MODEL.conv1.in_channels)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                outputs = _PERSUASION_MODEL(x, edge_index)
                embedding = outputs['embeddings'][0].numpy()
                
                # 確保是 128 維
                if len(embedding) >= 128:
                    return embedding[:128].tolist()
                else:
                    # 填充到 128 維
                    padded = np.zeros(128)
                    padded[:len(embedding)] = embedding
                    return padded.tolist()
    
    # 使用舊模型
    if _IDX is not None and _EMB is not None:
        idx = _IDX.get(agent_id)
        if idx is not None and idx < len(_EMB):
            embedding = _EMB[idx].numpy() if hasattr(_EMB[idx], 'numpy') else _EMB[idx]
            if len(embedding) >= 128:
                return embedding[:128].tolist()
            else:
                padded = np.zeros(128)
                padded[:len(embedding)] = embedding
                return padded.tolist()
    
    # 如果找不到，使用基於 agent_id 的確定性隨機向量
    np.random.seed(hash(agent_id) % 2**32)
    return np.random.rand(128).tolist()

def predict_persuasion(text_features: np.ndarray, agent_id: str = None) -> dict:
    """
    預測說服力相關指標
    
    Args:
        text_features: 文本特徵向量 (768維)
        agent_id: Agent ID（可選）
        
    Returns:
        包含預測結果的字典
    """
    if _PERSUASION_MODEL is None:
        return {
            'delta_probability': 0.5,
            'quality_score': 0.5,
            'best_strategy': 'analytical',
            'strategy_scores': {'aggressive': 0.25, 'defensive': 0.25, 
                              'analytical': 0.25, 'empathetic': 0.25}
        }
    
    with torch.no_grad():
        # 準備輸入 - 添加論證特徵的佔位符
        # 模型期望 770 維輸入 (768 文本特徵 + 2 論證特徵)
        if len(text_features) == 768:
            # 添加預設的論證特徵
            text_features = np.concatenate([text_features, [0.1, 15.0]])  # 預設值
        
        x = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0)
        
        # 如果需要圖結構，創建單節點圖
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # 預測
        outputs = _PERSUASION_MODEL(x, edge_index)
        
        # 處理輸出
        delta_prob = torch.sigmoid(outputs['delta']).item()
        quality_score = outputs['quality'].item()
        strategy_logits = outputs['strategy'][0]
        strategy_probs = F.softmax(strategy_logits, dim=0)
        
        # 找出最佳策略
        best_strategy_idx = torch.argmax(strategy_probs).item()
        best_strategy = STRATEGY_NAMES[best_strategy_idx]
        
        # 策略分數字典
        strategy_scores = {
            STRATEGY_NAMES[i]: strategy_probs[i].item() 
            for i in range(len(STRATEGY_NAMES))
        }
        
        return {
            'delta_probability': delta_prob,
            'quality_score': quality_score,
            'best_strategy': best_strategy,
            'strategy_scores': strategy_scores
        }

def get_social_influence_score(agent_id: str) -> float:
    """
    計算社會影響力分數
    
    基於監督式學習的結果，而不是簡單的向量求和
    """
    if _PERSUASION_MODEL is not None:
        # 使用模型預測的說服力作為影響力指標
        vec = social_vec(agent_id)
        text_features = np.zeros(768)  # 虛擬文本特徵
        predictions = predict_persuasion(text_features, agent_id)
        
        # 綜合考慮 delta 概率和品質分數
        influence = predictions['delta_probability'] * 0.7 + predictions['quality_score'] * 0.3
        return min(1.0, max(0.0, influence))
    else:
        # 使用舊方法
        vec = social_vec(agent_id)
        return sum(vec[:10]) / 10