"""
RL Policy Network for Debate Strategy
包含 snippet 選擇、策略決策和品質評估功能
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DebatePolicy(nn.Module):
    """辯論策略網路"""
    
    def __init__(self, hidden_size=256, social_dim=128):
        super().__init__()
        
        # Text encoder (使用預訓練模型的特徵)
        self.text_encoder = nn.Linear(768, hidden_size)  # DistilBERT 輸出維度
        
        # Social context encoder
        self.social_encoder = nn.Linear(social_dim, hidden_size // 2)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        
        # Strategy selection head
        self.strategy_head = nn.Linear(hidden_size, 4)  # 4種策略
        
        # Snippet ranking head
        self.ranking_head = nn.Linear(hidden_size * 2, 1)  # query + snippet
        
        # Quality prediction head
        self.quality_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, text_features, social_features=None, snippet_features=None):
        """前向傳播"""
        # Encode text
        text_enc = self.activation(self.text_encoder(text_features))
        
        # Encode social context if available
        if social_features is not None:
            social_enc = self.activation(self.social_encoder(social_features))
            combined = torch.cat([text_enc, social_enc], dim=-1)
        else:
            # Use zero padding for social features
            social_enc = torch.zeros(text_enc.size(0), text_enc.size(1) // 2, device=text_enc.device)
            combined = torch.cat([text_enc, social_enc], dim=-1)
        
        # Fusion
        fused = self.activation(self.fusion(combined))
        fused = self.dropout(fused)
        
        outputs = {}
        
        # Strategy selection
        outputs['strategy_logits'] = self.strategy_head(fused)
        
        # Quality prediction
        outputs['quality_score'] = self.quality_head(fused)
        
        # Snippet ranking if snippet features provided
        if snippet_features is not None:
            snippet_enc = self.activation(self.text_encoder(snippet_features))
            ranking_input = torch.cat([fused, snippet_enc], dim=-1)
            outputs['ranking_score'] = self.ranking_head(ranking_input)
        
        return outputs

class PolicyNetwork:
    """策略網路管理器"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 載入預訓練的品質評估模型
        self.quality_model_path = Path("data/models/policy")
        self.load_quality_model()
        
        # 載入策略網路
        self.policy_model = DebatePolicy()
        if model_path and Path(model_path).exists():
            self.load_policy_model(model_path)
        
        self.policy_model.to(self.device)
        self.policy_model.eval()
        
        # 策略映射
        self.strategies = {
            0: 'aggressive',     # 積極攻擊型
            1: 'defensive',      # 防禦反駁型  
            2: 'analytical',     # 分析論證型
            3: 'empathetic'      # 同理說服型
        }
        
    def load_quality_model(self):
        """載入品質評估模型"""
        try:
            if self.quality_model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.quality_model_path))
                # 修復 meta tensor 錯誤：先載入到 CPU，再移動到目標設備
                self.quality_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.quality_model_path),
                    torch_dtype=torch.float32,  # 指定數據類型
                    low_cpu_mem_usage=False     # 關閉低內存模式
                )
                # 確保模型在正確的設備上
                self.quality_model = self.quality_model.to(self.device)
                self.quality_model.eval()
                print(f"✅ 載入品質評估模型: {self.quality_model_path}")
            else:
                print("⚠️ 品質評估模型未找到，使用預設評分")
                self.tokenizer = None
                self.quality_model = None
        except Exception as e:
            print(f"❌ 載入品質評估模型失敗: {e}")
            self.tokenizer = None
            self.quality_model = None
    
    def load_policy_model(self, model_path: str):
        """載入策略模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 載入策略模型: {model_path}")
        except Exception as e:
            print(f"⚠️ 載入策略模型失敗，使用隨機初始化: {e}")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """將文本編碼為特徵向量"""
        if self.tokenizer and self.quality_model:
            try:
                inputs = self.tokenizer(text, 
                                      truncation=True, 
                                      padding=True, 
                                      max_length=512,
                                      return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # 使用品質模型的隱藏層作為特徵
                    outputs = self.quality_model(**inputs, output_hidden_states=True)
                    # 取最後一層的 [CLS] token
                    features = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
                
                return features
            except Exception as e:
                print(f"❌ 文本編碼失敗: {e}")
                # 返回隨機特徵
                return torch.randn(1, 768, device=self.device)
        else:
            # 使用簡單的特徵提取
            return torch.randn(1, 768, device=self.device)
    
    def predict_quality(self, text: str) -> float:
        """預測文本品質分數"""
        if self.quality_model and self.tokenizer:
            try:
                inputs = self.tokenizer(text,
                                      truncation=True,
                                      padding=True,
                                      max_length=512,
                                      return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.quality_model(**inputs)
                    score = outputs.logits.item()
                
                return max(0.0, min(2.0, score))  # 限制在 0-2 範圍
            except Exception as e:
                print(f"❌ 品質預測失敗: {e}")
                return 0.5
        else:
            # 簡單的啟發式評分
            return self._heuristic_quality(text)
    
    def _heuristic_quality(self, text: str) -> float:
        """啟發式品質評分"""
        score = 0.5  # 基礎分
        
        # 長度評分
        word_count = len(text.split())
        if 50 <= word_count <= 150:
            score += 0.2
        elif word_count < 20:
            score -= 0.2
        
        # 結構評分
        if any(marker in text for marker in ['because', 'therefore', 'however', 'furthermore']):
            score += 0.15
        
        # 證據評分
        if any(marker in text for marker in ['study', 'research', 'data', 'evidence']):
            score += 0.1
        
        # 引用評分
        if '[CITE]' in text or re.search(r'\[\d+\]', text):
            score += 0.1
        
        return max(0.0, min(2.0, score))
    
    def select_strategy(self, query: str, context: str = "", social_context: List[float] = None) -> str:
        """選擇辯論策略"""
        try:
            # 編碼輸入
            query_features = self.encode_text(query + " " + context)
            
            # 處理社會背景
            if social_context:
                social_tensor = torch.tensor(social_context, device=self.device).unsqueeze(0)
            else:
                social_tensor = None
            
            # 策略預測
            with torch.no_grad():
                outputs = self.policy_model(query_features, social_tensor)
                strategy_probs = torch.softmax(outputs['strategy_logits'], dim=-1)
                strategy_idx = torch.argmax(strategy_probs, dim=-1).item()
            
            selected_strategy = self.strategies[strategy_idx]
            confidence = strategy_probs[0][strategy_idx].item()
            
            print(f"🎯 選擇策略: {selected_strategy} (信心度: {confidence:.3f})")
            return selected_strategy
            
        except Exception as e:
            print(f"❌ 策略選擇失敗: {e}")
            return 'analytical'  # 預設策略

def choose_snippet(state_text: str, pool: List[Dict], policy_net: PolicyNetwork = None) -> str:
    """選擇最佳的證據片段"""
    
    if not pool:
        return "No evidence available."
    
    # 初始化策略網路
    if policy_net is None:
        policy_net = PolicyNetwork()
    
    try:
        # 編碼查詢文本
        query_features = policy_net.encode_text(state_text)
        
        # 評估每個片段
        snippet_scores = []
        for snippet in pool:
            content = snippet.get('content', '')
            
            # 編碼片段
            snippet_features = policy_net.encode_text(content)
            
            # 計算相關性分數
            with torch.no_grad():
                outputs = policy_net.policy_model(
                    query_features, 
                    snippet_features=snippet_features
                )
                relevance_score = torch.sigmoid(outputs['ranking_score']).item()
            
            # 預測品質分數
            quality_score = policy_net.predict_quality(content)
            
            # 綜合分數
            original_score = snippet.get('similarity_score', 0.0)
            combined_score = (
                0.4 * relevance_score + 
                0.3 * quality_score + 
                0.3 * original_score
            )
            
            snippet_scores.append({
                'content': content,
                'score': combined_score,
                'relevance': relevance_score,
                'quality': quality_score,
                'original': original_score,
                'metadata': snippet
            })
        
        # 排序並選擇最佳片段
        snippet_scores.sort(key=lambda x: x['score'], reverse=True)
        best_snippet = snippet_scores[0]
        
        print(f"📝 選擇最佳片段 (分數: {best_snippet['score']:.3f})")
        print(f"   - 相關性: {best_snippet['relevance']:.3f}")
        print(f"   - 品質: {best_snippet['quality']:.3f}")
        print(f"   - 原始: {best_snippet['original']:.3f}")
        
        return best_snippet['content']
        
    except Exception as e:
        print(f"❌ 片段選擇失敗: {e}")
        # 回退到簡單選擇
        return max(pool, key=lambda x: x.get('similarity_score', 0.0))['content']

# 全域策略網路實例
_global_policy_net = None

def get_policy_network() -> PolicyNetwork:
    """獲取全域策略網路實例"""
    global _global_policy_net
    if _global_policy_net is None:
        _global_policy_net = PolicyNetwork()
    return _global_policy_net

def select_strategy(query: str, context: str = "", social_context: List[float] = None) -> str:
    """選擇辯論策略（外部接口）"""
    policy_net = get_policy_network()
    return policy_net.select_strategy(query, context, social_context)

# 主要外部接口
__all__ = ['choose_snippet', 'select_strategy', 'PolicyNetwork', 'DebatePolicy']
