"""
GNN 社會編碼器
提供社會向量編碼功能
"""

from torch import load
from pathlib import Path
import numpy as np

# 載入預訓練模型
_ckpt_path = Path('data/models/gnn_social.pt')
if _ckpt_path.exists():
    _ckpt = load(str(_ckpt_path), map_location='cpu')
    _EMB = _ckpt['emb']
    _IDX = _ckpt['node2idx']
else:
    print("⚠️ GNN 模型未找到，使用隨機向量")
    _EMB = None
    _IDX = None

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
    # 如果模型未載入，返回隨機向量
    if _IDX is None or _EMB is None:
        np.random.seed(hash(agent_id) % 2**32)
        return np.random.rand(128).tolist()
    
    # 嘗試從索引中獲取
    idx = _IDX.get(agent_id)
    
    if idx is not None and idx < len(_EMB):
        # 獲取嵌入向量
        embedding = _EMB[idx].numpy() if hasattr(_EMB[idx], 'numpy') else _EMB[idx]
        
        # 確保是 128 維
        if len(embedding) >= 128:
            return embedding[:128].tolist()
        else:
            # 填充到 128 維
            padded = np.zeros(128)
            padded[:len(embedding)] = embedding
            return padded.tolist()
    else:
        # 如果找不到，使用基於 agent_id 的確定性隨機向量
        np.random.seed(hash(agent_id) % 2**32)
        return np.random.rand(128).tolist()