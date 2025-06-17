# 平行辯論系統架構說明

## 🏗️ 系統概述

本系統實現了 **RL + GNN + RAG 平行處理** 的智能辯論系統，支援動態說服/反駁機制，讓 AI Agent 能夠進行真實的辯論互動。

## 🔄 平行處理流程

### 1. 平行分析階段 (同時執行)

每個回合開始時，系統為每個 Agent 同時執行三個分析任務：

#### 🎯 RL 策略選擇
- **輸入**: 辯論上下文、歷史對話、社會背景
- **處理**: 神經網路策略選擇、品質預測
- **輸出**: 最適合的辯論策略 (aggressive/defensive/analytical/empathetic)

#### 🌐 GNN 社會分析  
- **輸入**: Agent ID、當前狀態、社會關係
- **處理**: 社會向量編碼、影響力計算、立場趨勢分析
- **輸出**: 社會影響力分數、立場變化趨勢

#### 📚 RAG 證據檢索
- **輸入**: 查詢上下文、辯論主題
- **處理**: 向量檢索、證據排序、片段選擇
- **輸出**: 最相關的證據片段、證據類型分布

### 2. 結果融合階段

```python
def fuse_analysis_results(rl_result, gnn_result, rag_result):
    # 策略調整：根據社會影響力調整基礎策略
    if influence_score > 0.6 and abs(stance) > 0.5:
        strategy = 'aggressive'  # 高影響力 + 強立場 = 積極
    elif influence_score < 0.4 and abs(stance) < 0.3:
        strategy = 'defensive'   # 低影響力 + 弱立場 = 謹慎
    
    # 證據選擇：結合策略和品質選擇最佳證據
    best_evidence = choose_snippet(context, evidence_pool)
    
    return fused_strategy, best_evidence
```

### 3. GPT-4o 生成階段

整合所有分析結果，生成個性化的辯論回覆：

```python
prompt = f"""
你是 Agent {agent_id}，當前立場: {stance:.2f}，信念: {conviction:.2f}
策略: {strategy}
證據: {evidence}
目標分析: {target_weaknesses}

請根據 {strategy} 策略生成回覆...
"""
```

## 🎯 動態說服/反駁機制

### 說服機制
- **觸發條件**: 說服力分數 > 0.6
- **效果**: 
  - 目標 Agent 立場向中性移動
  - 信念度降低 (×0.9)
  - 更容易在後續回合被影響

```python
if persuasion_score > 0.6:
    target.stance *= (1.0 - persuasion_effect * 0.3)  # 立場中性化
    target.conviction *= 0.9                           # 信念減弱
```

### 反駁機制
- **觸發條件**: 攻擊性分數 > 0.3 且超過抵抗閾值
- **效果**:
  - 目標 Agent 立場更加極端
  - 信念度增強 (×1.1)
  - 對後續攻擊更有抵抗力

```python
if attack_effect > 0.3:
    target.stance *= (1.0 + attack_effect * 0.2)  # 立場極化
    target.conviction = min(1.0, target.conviction * 1.1)  # 信念增強
```

## 📊 Agent 狀態模型

每個 Agent 維護以下狀態：

```python
@dataclass
class AgentState:
    agent_id: str
    current_stance: float      # -1.0 到 1.0，立場強度
    conviction: float          # 0.0 到 1.0，信念堅定度  
    social_context: List[float] # 128維社會背景向量
    persuasion_history: List[float]  # 被說服歷史
    attack_history: List[float]      # 攻擊歷史
```

### 狀態更新規則

1. **立場更新**:
   - 被說服 → 立場趨向中性
   - 被攻擊 → 立場更極端
   - 受社會影響力調節

2. **信念更新**:
   - 被說服 → 信念減弱
   - 被攻擊 → 信念增強
   - 影響後續抵抗力

3. **歷史記錄**:
   - 保存最近10次互動記錄
   - 用於趨勢分析和策略調整

## 🚀 性能優化

### 平行處理優勢
- **時間效率**: 3個分析任務同時執行，總時間 ≈ max(RL, GNN, RAG)
- **資源利用**: 充分利用多核 CPU 和 GPU 資源
- **擴展性**: 可輕鬆添加新的分析模組

### 異步執行模式
```python
async def parallel_analysis(agent_id, topic, history):
    # 創建異步任務
    rl_task = loop.run_in_executor(executor, rl_analysis, ...)
    gnn_task = loop.run_in_executor(executor, gnn_analysis, ...)
    rag_task = loop.run_in_executor(executor, rag_analysis, ...)
    
    # 等待所有任務完成
    rl_result, gnn_result, rag_result = await asyncio.gather(
        rl_task, gnn_task, rag_task
    )
```

## 🎭 使用示例

### 基本使用
```python
import asyncio
from orchestrator.parallel_orchestrator import create_parallel_orchestrator

async def run_debate():
    # 1. 創建協調器
    orchestrator = create_parallel_orchestrator()
    
    # 2. 初始化 Agent
    agent_configs = [
        {'id': 'A', 'initial_stance': 0.7, 'initial_conviction': 0.8},
        {'id': 'B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'C', 'initial_stance': 0.1, 'initial_conviction': 0.5}
    ]
    orchestrator.initialize_agents(agent_configs)
    
    # 3. 執行辯論
    for round_num in range(1, 4):
        await orchestrator.run_debate_round(
            round_number=round_num,
            topic="AI regulation",
            agent_order=['A', 'B', 'C']
        )
    
    # 4. 獲取結果
    summary = orchestrator.get_debate_summary()
    print(f"最有說服力: {summary['most_persuasive_agent']}")

# 運行
asyncio.run(run_debate())
```

## 📈 效果評估

系統提供多維度的效果評估：

### 回覆品質評估
- **說服力分數**: 基於溫和詞彙和同理表達
- **攻擊性分數**: 基於批判詞彙和對抗表達  
- **證據使用**: 引用和數據支持程度
- **長度適中**: 回覆長度合理性

### 辯論效果分析
- **立場變化**: 各 Agent 立場的動態變化
- **說服成功率**: 成功改變對手立場的比例
- **抵抗能力**: 面對攻擊時的穩定性
- **策略適應**: 策略選擇的合理性

## 🔧 配置參數

### 策略配置
```python
strategy_configs = {
    'aggressive': {'k': 3, 'index_type': 'high_quality', 'persuasion_only': True},
    'defensive': {'k': 5, 'index_type': 'comprehensive', 'persuasion_only': False},
    'analytical': {'k': 4, 'index_type': 'high_quality', 'persuasion_only': False},
    'empathetic': {'k': 3, 'index_type': 'comprehensive', 'persuasion_only': True}
}
```

### 更新參數
```python
persuasion_effect_rate = 0.3    # 說服效果強度
attack_resistance_rate = 0.8    # 攻擊抵抗比例
conviction_decay_rate = 0.9     # 信念衰減率
conviction_boost_rate = 1.1     # 信念增強率
```

## 🎯 應用場景

1. **教育辯論**: 訓練學生辯論技巧
2. **政策分析**: 模擬政策討論和決策過程
3. **產品設計**: 收集多角度用戶反饋
4. **研究工具**: 探索說服和影響機制
5. **遊戲 AI**: 創建智能 NPC 對話系統

## 🚀 未來擴展

- **情感分析**: 加入情感狀態追蹤
- **記憶系統**: 長期記憶和學習能力
- **多模態**: 支援圖像、音頻等多媒體證據
- **群體動力**: 支援更大規模的群體辯論
- **實時學習**: 在線學習和策略優化 