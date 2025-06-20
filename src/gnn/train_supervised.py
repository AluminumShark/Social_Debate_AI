"""
監督式 GNN 訓練
預測說服成功率和最佳說服策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import networkx as nx
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel

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

class PersuasionDataset:
    """說服力數據集"""
    
    def __init__(self, pairs_path='data/raw/pairs.jsonl'):
        self.pairs_path = Path(pairs_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PersuasionDataset 使用設備: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.encoder = self.encoder.to(self.device)  # 將模型移到 GPU
        self.encoder.eval()
        
        # 策略映射
        self.strategies = {
            'aggressive': 0,
            'defensive': 1,
            'analytical': 2,
            'empathetic': 3
        }
        
    def encode_text(self, text: str) -> np.ndarray:
        """編碼文本為向量"""
        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, max_length=512, 
                                  padding=True, return_tensors='pt')
            # 將輸入移到 GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs)
            # 使用 [CLS] token 的表示，並移回 CPU
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    
    def encode_texts_batch(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
        """批次編碼文本為向量（提高 GPU 利用率）"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, truncation=True, max_length=512, 
                                      padding=True, return_tensors='pt')
                # 將輸入移到 GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.encoder(**inputs)
                # 使用 [CLS] token 的表示
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def extract_argument_features(self, text: str) -> Dict[str, float]:
        """提取論證特徵"""
        features = {}
        
        # 論證指標詞
        argument_markers = {
            'causal': ['because', 'therefore', 'thus', 'hence', 'consequently'],
            'contrast': ['however', 'but', 'although', 'despite', 'nevertheless'],
            'evidence': ['studies show', 'research indicates', 'data suggests', 'according to'],
            'example': ['for example', 'for instance', 'such as', 'like'],
            'emphasis': ['indeed', 'in fact', 'clearly', 'obviously', 'certainly']
        }
        
        text_lower = text.lower()
        
        # 計算各類論證標記的出現次數
        for category, markers in argument_markers.items():
            count = sum(1 for marker in markers if marker in text_lower)
            features[f'arg_{category}'] = count
        
        # 計算論證結構分數
        total_markers = sum(features.values())
        features['argument_density'] = total_markers / (len(text.split()) + 1)
        
        # 句子複雜度
        sentences = text.split('.')
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        return features
    
    def determine_strategy(self, text: str, is_delta: bool) -> int:
        """根據文本內容判斷說服策略"""
        text_lower = text.lower()
        
        # 策略指標詞
        aggressive_markers = ['wrong', 'incorrect', 'flawed', 'ridiculous', 'absurd']
        defensive_markers = ['defend', 'maintain', 'stand by', 'believe', 'position']
        analytical_markers = ['analyze', 'consider', 'examine', 'evaluate', 'assess']
        empathetic_markers = ['understand', 'appreciate', 'see your point', 'relate', 'feel']
        
        scores = {
            'aggressive': sum(1 for m in aggressive_markers if m in text_lower),
            'defensive': sum(1 for m in defensive_markers if m in text_lower),
            'analytical': sum(1 for m in analytical_markers if m in text_lower),
            'empathetic': sum(1 for m in empathetic_markers if m in text_lower)
        }
        
        # 如果是成功的說服（delta），傾向於分析型或同理型
        if is_delta:
            scores['analytical'] += 2
            scores['empathetic'] += 1
        
        # 選擇得分最高的策略
        best_strategy = max(scores.keys(), key=lambda k: scores[k])
        return self.strategies[best_strategy]
    
    def build_interaction_graph(self) -> Tuple[Data, Dict]:
        """構建互動圖"""
        print("構建說服互動圖...")
        
        # 收集所有互動數據
        interactions = []
        texts_to_encode = []
        text_metadata = []  # 存儲文本對應的元數據
        
        print("收集文本數據...")
        with open(self.pairs_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="掃描數據"):
                try:
                    data = json.loads(line)
                    submission = data.get('submission', {})
                    
                    # 處理 delta 評論
                    if 'delta_comment' in data and data['delta_comment']:
                        delta_data = data['delta_comment']
                        if 'comments' in delta_data and delta_data['comments']:
                            for comment in delta_data['comments']:
                                if comment.get('body') and len(comment['body']) > 50:
                                    author = comment.get('author', 'unknown')
                                    if author and author != '[deleted]' and author != 'unknown':
                                        texts_to_encode.append(comment['body'][:1000])
                                        text_metadata.append({
                                            'author': author,
                                            'is_delta': True,
                                            'score': comment.get('score', 0),
                                            'body': comment['body'],
                                            'op_author': submission.get('author', 'unknown')
                                        })
                    
                    # 處理 non-delta 評論
                    if 'nodelta_comment' in data and data['nodelta_comment']:
                        nodelta_data = data['nodelta_comment']
                        if 'comments' in nodelta_data and nodelta_data['comments']:
                            for comment in nodelta_data['comments']:
                                if comment.get('body') and len(comment['body']) > 50:
                                    author = comment.get('author', 'unknown')
                                    if author and author != '[deleted]' and author != 'unknown':
                                        texts_to_encode.append(comment['body'][:1000])
                                        text_metadata.append({
                                            'author': author,
                                            'is_delta': False,
                                            'score': comment.get('score', 0),
                                            'body': comment['body'],
                                            'op_author': submission.get('author', 'unknown')
                                        })
                
                except Exception as e:
                    continue
        
        print(f"收集到 {len(texts_to_encode)} 條文本")
        
        # 批次編碼所有文本
        print("批次編碼文本（使用 GPU）...")
        text_embeddings = self.encode_texts_batch(texts_to_encode, batch_size=32)
        
        # 構建節點特徵和標籤
        node_features = {}
        node_labels = {}
        
        print("構建節點特徵...")
        for embedding, metadata in zip(text_embeddings, text_metadata):
            author = metadata['author']
            
            # 提取論證特徵
            arg_features = self.extract_argument_features(metadata['body'])
            
            # 組合特徵
            features = np.concatenate([
                embedding,
                [arg_features.get('argument_density', 0),
                 arg_features.get('avg_sentence_length', 0)]
            ])
            
            node_features[author] = features
            node_labels[author] = {
                'is_delta': metadata['is_delta'],
                'score': metadata['score'],
                'strategy': self.determine_strategy(metadata['body'], metadata['is_delta'])
            }
            
            # 添加互動邊
            op_author = metadata['op_author']
            if op_author and op_author != '[deleted]' and op_author != author:
                weight = 3.0 if metadata['is_delta'] else 1.0
                interactions.append((op_author, author, {'weight': weight}))
        
        # 構建圖
        G = nx.Graph()
        G.add_edges_from([(u, v) for u, v, _ in interactions])
        
        # 創建節點索引映射
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        # 準備 PyG 數據
        x_list = []
        y_delta = []
        y_score = []
        y_strategy = []
        
        for node in G.nodes():
            if node in node_features:
                x_list.append(node_features[node])
                labels = node_labels[node]
                y_delta.append(1 if labels['is_delta'] else 0)
                y_score.append(labels['score'] / 100.0)  # 標準化分數
                y_strategy.append(labels['strategy'])
            else:
                # 對於沒有特徵的節點，使用零向量
                x_list.append(np.zeros(770))  # 768 + 2
                y_delta.append(0)
                y_score.append(0.0)
                y_strategy.append(2)  # 預設為 analytical
        
        # 轉換為張量 - 先轉換為 numpy array 再轉為張量
        x = torch.FloatTensor(np.array(x_list))
        edge_index = torch.LongTensor([[node_to_idx[u], node_to_idx[v]] 
                                      for u, v in G.edges()]).t()
        
        # 創建 PyG Data 對象
        data = Data(
            x=x,
            edge_index=edge_index,
            y_delta=torch.FloatTensor(y_delta),
            y_score=torch.FloatTensor(y_score),
            y_strategy=torch.LongTensor(y_strategy)
        )
        
        print(f"圖構建完成：{len(G.nodes())} 個節點，{len(G.edges())} 條邊")
        print(f"Delta 節點：{sum(y_delta)} 個")
        
        return data, node_to_idx

def train_supervised_gnn(
    epochs=50,
    hidden_dim=256,
    learning_rate=0.001,
    output_path='data/models/gnn_persuasion.pt'
):
    """訓練監督式 GNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 準備數據
    dataset = PersuasionDataset()
    data, node_to_idx = dataset.build_interaction_graph()
    
    # 檢查是否有足夠的數據
    if data.x.size(0) == 0:
        print("❌ 錯誤：沒有載入到任何數據！")
        print("請確認 data/raw/pairs.jsonl 檔案存在且格式正確。")
        return
    
    if data.x.size(0) < 10:
        print(f"⚠️ 警告：只有 {data.x.size(0)} 個節點，可能不足以進行有效訓練。")
    
    data = data.to(device)
    
    # 劃分訓練集和測試集
    num_nodes = data.x.size(0)
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # 初始化模型
    model = PersuasionGNN(
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練
    print(f"\n開始訓練 {epochs} 個 epochs...")
    best_val_acc = 0
    
    for epoch in range(epochs):
        # 訓練模式
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data.x, data.edge_index)
        
        # 計算多任務損失
        # 1. Delta 預測損失（二元交叉熵）
        delta_loss = F.binary_cross_entropy_with_logits(
            outputs['delta'][train_mask].squeeze(),
            data.y_delta[train_mask]
        )
        
        # 2. 品質分數損失（MSE）
        quality_loss = F.mse_loss(
            outputs['quality'][train_mask].squeeze(),
            data.y_score[train_mask]
        )
        
        # 3. 策略預測損失（交叉熵）
        strategy_loss = F.cross_entropy(
            outputs['strategy'][train_mask],
            data.y_strategy[train_mask]
        )
        
        # 總損失（加權組合）
        total_loss = 0.5 * delta_loss + 0.3 * quality_loss + 0.2 * strategy_loss
        
        total_loss.backward()
        optimizer.step()
        
        # 評估
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(data.x, data.edge_index)
                
                # Delta 預測準確率
                delta_pred = (outputs['delta'][test_mask].squeeze() > 0).float()
                delta_acc = accuracy_score(
                    data.y_delta[test_mask].cpu(),
                    delta_pred.cpu()
                )
                
                # 策略預測準確率
                strategy_pred = outputs['strategy'][test_mask].argmax(dim=1)
                strategy_acc = accuracy_score(
                    data.y_strategy[test_mask].cpu(),
                    strategy_pred.cpu()
                )
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  訓練損失: {total_loss.item():.4f}")
                print(f"  Delta 準確率: {delta_acc:.4f}")
                print(f"  策略準確率: {strategy_acc:.4f}")
                
                # 保存最佳模型
                if delta_acc > best_val_acc:
                    best_val_acc = delta_acc
                    
                    # 確保輸出目錄存在
                    output_dir = Path(output_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'model_state': model.state_dict(),
                        'node_to_idx': node_to_idx,
                        'config': {
                            'hidden_dim': hidden_dim,
                            'input_dim': data.x.size(1)
                        },
                        'performance': {
                            'delta_acc': delta_acc,
                            'strategy_acc': strategy_acc
                        }
                    }, output_path)
                    print(f"  💾 保存最佳模型 (Delta 準確率: {delta_acc:.4f})")
    
    print(f"\n✅ 訓練完成！最佳 Delta 準確率: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_supervised_gnn() 