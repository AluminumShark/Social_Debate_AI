# RL 增強辯論系統使用指南

## 功能概述

本系統整合了強化學習 (RL) 策略選擇和證據片段選擇功能，提供智能化的辯論回覆生成。

## 核心組件

### 1. RL Policy Network (`src/rl/policy_network.py`)

**主要功能：**
- 策略選擇：根據辯論上下文選擇最適合的辯論策略
- 片段選擇：從證據池中選擇最相關和高品質的證據片段
- 品質評估：預測文本的辯論品質分數

**策略類型：**
- `aggressive`: 積極攻擊型策略
- `defensive`: 防禦反駁型策略
- `analytical`: 分析論證型策略
- `empathetic`: 同理說服型策略

### 2. Enhanced Orchestrator (`src/orchestrator/orchestrator.py`)

**新增功能：**
- `get_rl_enhanced_reply()`: 使用 RL 策略的增強回覆生成
- 策略配置管理：不同策略的檢索參數優化
- 社會背景整合：支援社會向量輸入

## 使用方法

### 基本使用

```python
from rl.policy_network import choose_snippet, select_strategy
from orchestrator.orchestrator import create_enhanced_orchestrator

# 1. 策略選擇
query = "Should we implement universal healthcare?"
strategy = select_strategy(query)
print(f"選擇策略: {strategy}")

# 2. 證據片段選擇
pool = [
    {
        'content': '證據內容...',
        'similarity_score': 0.8,
        'score': 1.2
    }
]
chosen = choose_snippet(query, pool)
print(f"選擇片段: {chosen}")

# 3. RL 增強回覆生成
orchestrator = create_enhanced_orchestrator()
topic = "Climate change"
history = ["對話歷史..."]
social_context = [0.1, 0.2, -0.1, 0.5]  # 社會背景向量

reply = orchestrator.get_rl_enhanced_reply(
    topic, history, "A", social_context
)
```

### 完整辯論流程

```python
def generate_debate_reply(topic, history, agent, social_context=None):
    """生成 RL 增強的辯論回覆"""
    
    # 1. 構建狀態文本
    recent_turns = history[-3:] if history else []
    recent = '\n'.join(recent_turns)
    state_text = f"Topic: {topic}\nRecent turns: {recent}"
    
    # 2. 使用 RL 選擇策略
    from rl.policy_network import select_strategy
    selected_strategy = select_strategy(state_text, recent, social_context)
    
    # 3. 檢索證據池 (需要 RAG 系統)
    from rag.retriever import create_enhanced_retriever
    retriever = create_enhanced_retriever()
    pool = retriever.retrieve(query=state_text, k=5)
    
    # 4. 使用 RL 選擇最佳片段
    from rl.policy_network import choose_snippet
    chosen = choose_snippet(state_text, pool)
    
    # 5. 構建提示並生成回覆
    prompt = f"""
Topic: {topic}
Recent turns: {recent}
Social: {social_context}
Evidence Snippet: "{chosen}"
Strategy: {selected_strategy}

Write ≤120 words persuading the opponent using {selected_strategy} approach. Cite as [CITE].
"""
    
    from gpt_interface.gpt_client import chat
    reply = chat(prompt)
    return reply
```

## 策略配置

每種策略都有特定的檢索配置：

| 策略 | 檢索數量 | 索引類型 | 只選說服性內容 |
|------|----------|----------|----------------|
| aggressive | 3 | high_quality | ✓ |
| defensive | 5 | comprehensive | ✗ |
| analytical | 4 | high_quality | ✗ |
| empathetic | 3 | comprehensive | ✓ |

## 模型架構

### DebatePolicy 神經網路

```
輸入: 文本特徵 (768維) + 社會特徵 (128維)
      ↓
文本編碼器 (768 → 256)
      ↓
社會編碼器 (128 → 128)
      ↓
融合層 (384 → 256)
      ↓
策略選擇頭 (256 → 4)    品質預測頭 (256 → 1)    排序頭 (512 → 1)
```

## 訓練數據

RL 系統使用 `data/rl_pairs.csv` 中的辯論品質數據進行訓練：
- 36,277 條訓練記錄
- 品質分數範圍：-0.338 到 1.738
- 平均品質分數：0.633

## 依賴項

```
torch>=1.9.0
transformers>=4.20.0
numpy>=1.21.0
pathlib
```

## 注意事項

1. **模型路徑**: 品質評估模型預設路徑為 `data/models/policy`
2. **GPU 支援**: 自動檢測並使用 CUDA 加速
3. **回退機制**: 如果 RL 模型不可用，會回退到啟發式方法
4. **社會背景**: 可選參數，如果不提供會使用零向量填充

## 測試

運行示範腳本：
```bash
python demo_rl_orchestrator.py
```

這將展示所有 RL 功能的使用方法和效果。 