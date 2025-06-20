"""
RL Policy Network for Debate Strategy
使用 PPO 訓練的策略網路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 導入 PPO 相關類
from .ppo_trainer import DebatePPONetwork

class PolicyNetwork:
    """辯論策略網路（使用 PPO）"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PolicyNetwork 使用設備: {self.device}")
        
        # 載入 tokenizer 和 encoder
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # 策略列表
        self.strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
        
        # 載入 PPO 模型
        self.policy_model = DebatePPONetwork(
            state_dim=768,
            hidden_dim=256,
            num_actions=4
        ).to(self.device)
        
        # 如果提供了模型路徑，載入預訓練權重
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("⚠️ 未找到預訓練 PPO 模型，使用隨機初始化")
    
    def load_model(self, model_path: str):
        """載入預訓練模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_model.eval()
            print(f"✅ 載入 PPO 模型: {model_path}")
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
    
    def encode_state(self, 
                    query: str, 
                    context: str = "", 
                    social_context: Optional[List[float]] = None,
                    debate_history: Optional[List[Dict]] = None) -> torch.Tensor:
        """編碼辯論狀態"""
        # 編碼文本
        text = f"{query} {context}".strip()
        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, max_length=512, 
                                  padding=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        
        # 創建狀態向量
        state = np.zeros(768)
        state[:len(text_features)] = text_features
        
        # 添加社會背景信息
        if social_context and len(social_context) >= 2:
            state[0] = social_context[0]  # 當前立場
            state[1] = social_context[1]  # 對手立場
        
        # 添加歷史信息
        if debate_history:
            recent_strategies = [h.get('strategy', 'analytical') for h in debate_history[-5:]]
            for i, strategy in enumerate(recent_strategies):
                if strategy in self.strategies:
                    state[10 + i] = self.strategies.index(strategy) / 4.0
        
        return torch.tensor(state, dtype=torch.float32)
    
    def select_strategy(self, 
                       query: str, 
                       context: str = "", 
                       social_context: Optional[List[float]] = None,
                       debate_history: Optional[List[Dict]] = None,
                       deterministic: bool = False) -> str:
        """選擇辯論策略"""
        # 編碼狀態
        state = self.encode_state(query, context, social_context, debate_history)
        state = state.unsqueeze(0).to(self.device)
        
        # 使用 PPO 模型選擇策略
        with torch.no_grad():
            action, _, _ = self.policy_model.get_action(state, deterministic=deterministic)
        
        selected_strategy = self.strategies[action]
        print(f"🎯 PPO 選擇策略: {selected_strategy}")
        
        return selected_strategy
    
    def get_strategy_distribution(self, 
                                 query: str, 
                                 context: str = "", 
                                 social_context: Optional[List[float]] = None,
                                 debate_history: Optional[List[Dict]] = None) -> Dict[str, float]:
        """獲取策略概率分布"""
        # 編碼狀態
        state = self.encode_state(query, context, social_context, debate_history)
        state = state.unsqueeze(0).to(self.device)
        
        # 獲取動作概率
        with torch.no_grad():
            action_logits, _ = self.policy_model(state)
            action_probs = F.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
        
        # 創建策略分布字典
        strategy_dist = {
            strategy: float(prob) 
            for strategy, prob in zip(self.strategies, action_probs)
        }
        
        return strategy_dist
    
    def predict_quality(self, text: str) -> float:
        """預測文本品質（使用價值函數）"""
        # 編碼狀態
        state = self.encode_state(text)
        state = state.unsqueeze(0).to(self.device)
        
        # 使用價值函數預測
        with torch.no_grad():
            _, value = self.policy_model(state)
            quality_score = torch.sigmoid(value).item()  # 轉換到 0-1 範圍
        
        return quality_score

# 全局實例
_policy_network = None

def get_policy_network(model_path: Optional[str] = None) -> PolicyNetwork:
    """獲取策略網路實例"""
    global _policy_network
    
    if _policy_network is None:
        # 嘗試載入預訓練模型
        if model_path is None:
            model_path = "data/models/ppo/ppo_debate_policy.pt"
        
        _policy_network = PolicyNetwork(model_path)
    
    return _policy_network

def select_strategy(query: str, 
                   context: str = "", 
                   social_context: Optional[List[float]] = None,
                   debate_history: Optional[List[Dict]] = None) -> str:
    """選擇辯論策略（便捷函數）"""
    network = get_policy_network()
    return network.select_strategy(query, context, social_context, debate_history)

def choose_snippet(query: str, 
                  snippets: List[str], 
                  context: str = "",
                  strategy: Optional[str] = None) -> str:
    """選擇最佳片段"""
    if not snippets:
        return ""
    
    network = get_policy_network()
    
    # 評估每個片段的品質
    scores = []
    for snippet in snippets:
        combined_text = f"{query} {context} {snippet}"
        score = network.predict_quality(combined_text)
        
        # 根據策略調整分數
        if strategy:
            if strategy == 'aggressive' and any(word in snippet.lower() for word in ['wrong', 'incorrect', 'flawed']):
                score *= 1.2
            elif strategy == 'empathetic' and any(word in snippet.lower() for word in ['understand', 'appreciate', 'feel']):
                score *= 1.2
            elif strategy == 'analytical' and any(word in snippet.lower() for word in ['analyze', 'consider', 'examine']):
                score *= 1.2
        
        scores.append(score)
    
    # 選擇最高分的片段
    best_idx = np.argmax(scores)
    print(f"📄 選擇片段 {best_idx + 1}/{len(snippets)}，分數: {scores[best_idx]:.3f}")
    
    return snippets[best_idx]
