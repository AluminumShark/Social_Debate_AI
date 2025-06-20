# 辯論評分與勝負判定系統

## 目錄

1. [系統概述](#系統概述)
2. [評分機制](#評分機制)
3. [狀態更新規則](#狀態更新規則)
4. [投降機制](#投降機制)
5. [勝負判定](#勝負判定)
6. [策略建議](#策略建議)
7. [實戰案例](#實戰案例)
8. [技術實現](#技術實現)

## 系統概述

Social Debate AI 的辯論評分系統是一個多維度的綜合評估機制，旨在模擬真實辯論中的複雜互動。系統不僅考慮論點的說服力，還評估參與者的立場堅定度、影響力和抗壓能力。

### 核心理念

- **動態平衡**：立場和信念會根據辯論過程動態調整
- **多元評價**：綜合考慮攻擊、防守、說服等多種能力
- **策略深度**：鼓勵深思熟慮的辯論策略，而非簡單對抗

## 評分機制

### 1. 立場堅定度得分（Stance Conviction Score）

**計算公式**：
```python
stance_score = abs(current_stance) × conviction × 30
```

**詳細說明**：
- `current_stance`：當前立場（-1.0 到 1.0）
  - 正值表示支持，負值表示反對
  - 絕對值越大，立場越極端
- `conviction`：信念強度（0.0 到 1.0）
  - 表示對自己立場的堅定程度
  - 越高越不容易被說服
- 基礎權重：30 分

**評分邏輯**：
- 獎勵有明確立場且堅定的參與者
- 中立或搖擺不定的立場得分較低
- 體現了辯論中"有立場"的重要性

**範例計算**：
| Agent | 立場 | 信念 | 得分 |
|-------|------|------|------|
| A | 0.8 | 0.7 | 16.8 |
| B | -0.6 | 0.6 | 10.8 |
| C | 0.2 | 0.5 | 3.0 |

### 2. 說服他人得分（Persuasion Score）

**計算方式**：
```python
persuasion_score = 0
for other_agent in all_agents:
    if other_agent.has_surrendered:
        persuasion_score += 20  # 投降獎勵
    
    # 影響力獎勵
    avg_persuasion = mean(other_agent.persuasion_history)
    persuasion_score += avg_persuasion × 10
```

**評分項目**：

#### a) 投降獎勵（20分/人）
- 成功說服對手投降是最高成就
- 每個投降的對手提供 20 分獎勵
- 體現了"不戰而屈人之兵"的辯論藝術

#### b) 影響力獎勵（0-10分/人）
- 基於對每個對手的平均說服度
- 即使未能讓對手投降，持續的影響也有價值
- 計算所有回合的累積效果

**說服度評估標準**：
- 0.0-0.2：幾乎無影響
- 0.2-0.4：輕微影響
- 0.4-0.6：中等影響
- 0.6-0.8：顯著影響
- 0.8-1.0：強烈影響

### 3. 抗壓能力得分（Resistance Score）

**計算公式**：
```python
avg_attack = mean(attack_history)
resistance_score = (1 - avg_attack) × conviction × 20
```

**評分邏輯**：
- 衡量在面對攻擊時保持立場的能力
- 受攻擊越少，或抵抗攻擊越成功，得分越高
- 信念強度是抗壓能力的重要因素

**抗壓表現分級**：
| 平均被攻擊度 | 信念強度 | 抗壓評價 |
|-------------|---------|----------|
| < 0.3 | > 0.7 | 優秀 |
| 0.3-0.5 | 0.5-0.7 | 良好 |
| 0.5-0.7 | 0.3-0.5 | 一般 |
| > 0.7 | < 0.3 | 較差 |

### 4. 投降懲罰（Surrender Penalty）

**固定懲罰**：-50 分

**懲罰理由**：
- 投降代表完全放棄自己的立場
- 在辯論中失去了繼續爭論的能力
- 嚴厲的懲罰確保 Agent 不會輕易放棄

## 狀態更新規則

### 1. 說服效果（Persuasion Effect）

當 Agent 被說服時（說服度 > 0.6）：

```python
# 立場向中立移動
persuasion_effect = persuasion_score × (1.0 - conviction)
new_stance = current_stance × (1.0 - persuasion_effect × 0.3)

# 信念減弱
new_conviction = conviction × 0.85
```

**效果解析**：
- 立場逐漸趨向中立（0）
- 信念強度下降，更容易被進一步說服
- 高信念者有更強的"免疫力"

### 2. 攻擊效果（Attack Effect）

當 Agent 被攻擊時（攻擊效果 > 0.3）：

```python
# 計算攻擊抵抗
attack_resistance = conviction × 0.8
attack_effect = max(0, attack_score - attack_resistance)

# 立場極化
new_stance = current_stance × (1.0 + attack_effect × 0.2)

# 信念增強
new_conviction = min(1.0, conviction × 1.1)
```

**效果解析**：
- 攻擊可能導致"反彈效應"
- 立場變得更加極端
- 信念因為防禦心理而增強
- 體現了心理學中的"確認偏誤"

### 3. 歷史記錄維護

```python
# 保持最近 10 次記錄
if len(history) > 10:
    history.pop(0)
```

- 只考慮最近的互動
- 避免早期影響過度累積
- 保持動態評估的時效性

## 投降機制

### 投降條件詳解

#### 條件 1：高說服度 + 低信念
```python
if recent_persuasion > 0.6 and conviction < 0.4:
    surrender = True
```

**觸發場景**：
- 連續受到有力論證的衝擊
- 自身論點基礎不夠堅實
- 開始懷疑自己的立場

#### 條件 2：立場動搖
```python
if abs(current_stance) < 0.2 and conviction < 0.5:
    surrender = True
```

**觸發場景**：
- 已經基本被說服到中立立場
- 失去了繼續辯論的動力
- 認為雙方都有道理

#### 條件 3：連續被說服
```python
consecutive_high = all(score > 0.5 for score in persuasion_history[-3:])
if consecutive_high:
    surrender = True
```

**觸發場景**：
- 連續 3 回合都被有效說服
- 無法提出有力的反駁
- 心理防線逐漸崩潰

### 投降後果

1. **立即效果**：
   - 得分扣除 50 分
   - 停止參與後續辯論
   - 被記錄為"被說服"

2. **對其他人的影響**：
   - 說服者獲得 20 分獎勵
   - 可能影響旁觀者的立場
   - 改變辯論的力量平衡

## 勝負判定

### 1. 總分計算

```python
total_score = (
    stance_score +           # 立場堅定度
    persuasion_score +       # 說服他人
    resistance_score -       # 抗壓能力
    surrender_penalty        # 投降懲罰
)
```

### 2. 勝利類型判定

#### 壓倒性勝利（Overwhelming Victory）
```python
if len(surrendered_agents) > 0:
    verdict = f"🏆 {winner} 獲得壓倒性勝利！成功說服 {surrendered} 投降。"
```

**特徵**：
- 至少說服一名對手投降
- 展現了卓越的說服能力
- 最高級別的勝利

#### 明顯優勢（Clear Advantage）
```python
if score_difference > 30:
    verdict = f"🏆 {winner} 以明顯優勢獲勝！展現了卓越的辯論技巧。"
```

**特徵**：
- 領先第二名超過 30 分
- 在多個維度上表現優秀
- 令人信服的勝利

#### 險勝（Narrow Victory）
```python
else:
    verdict = f"🏆 {winner} 險勝！這是一場勢均力敵的精彩辯論。"
```

**特徵**：
- 領先優勢不到 30 分
- 雙方實力接近
- 可能因為細節決定勝負

### 3. 綜合評價維度

系統會生成詳細的評價報告：

```json
{
    "total_rounds": 5,
    "winner": "Agent_A",
    "scores": {
        "Agent_A": 67.5,
        "Agent_B": 35.2,
        "Agent_C": -12.3
    },
    "surrendered_agents": ["Agent_C"],
    "final_states": {
        "Agent_A": {
            "stance": 0.75,
            "conviction": 0.65,
            "has_surrendered": false,
            "final_position": "支持"
        }
    },
    "verdict": "壓倒性勝利"
}
```

## 策略建議

### 1. 進攻型策略

**適用情況**：
- 對手信念較弱（< 0.5）
- 自己有強力證據支持
- 需要快速建立優勢

**執行要點**：
- 集中火力攻擊對手論點弱點
- 使用 aggressive 策略
- 配合高質量的 RAG 證據

**風險**：
- 可能觸發對手的防禦機制
- 過度攻擊可能被視為不理性

### 2. 防守型策略

**適用情況**：
- 自己信念堅定（> 0.7）
- 面對強勢對手
- 需要保持得分優勢

**執行要點**：
- 鞏固自己的核心論點
- 使用 defensive 策略
- 重點提升抗壓能力得分

**優勢**：
- 不易被說服投降
- 穩定累積立場得分

### 3. 平衡型策略

**適用情況**：
- 多方混戰局面
- 需要靈活應對
- 追求綜合得分

**執行要點**：
- 根據 GNN 預測動態調整策略
- 平衡攻擊和防守
- 注意觀察對手狀態變化

### 4. 說服型策略

**適用情況**：
- 對手立場不夠堅定
- 自己有說服力優勢
- 追求投降獎勵

**執行要點**：
- 使用 empathetic 策略建立共鳴
- 循序漸進削弱對手信念
- 在關鍵時刻施加說服壓力

## 實戰案例

### 案例 1：三方辯論的動態平衡

**初始狀態**：
- Agent_A：立場 +0.8（強烈支持），信念 0.7
- Agent_B：立場 -0.6（反對），信念 0.6  
- Agent_C：立場 0.0（中立），信念 0.5

**第一回合**：
- A 使用 analytical 策略，提出數據論證
- B 使用 aggressive 策略，攻擊 A 的假設
- C 使用 empathetic 策略，理解雙方觀點

**效果**：
- A 受到攻擊，立場更堅定：+0.85
- B 的攻擊被抵抗，信念微增：0.62
- C 被雙方影響，略偏向 A：+0.15

**第二回合**：
- A 調整為 defensive，鞏固論點
- B 繼續 aggressive，加大攻擊
- C 改用 analytical，提出折衷方案

**關鍵轉折**：
- B 的過度攻擊引起反感
- C 的理性分析獲得認可
- A 開始考慮 C 的觀點

**最終結果**：
- Agent_A：45.6 分（立場堅定但未能說服他人）
- Agent_B：28.3 分（攻擊策略效果有限）
- Agent_C：52.1 分（成功影響雙方，險勝）

### 案例 2：說服致勝

**場景**：關於"全民基本收入"的辯論

**關鍵策略**：
1. Agent_A 先用 analytical 建立理論基礎
2. 識別 Agent_B 的信念薄弱點
3. 轉換為 empathetic 策略，理解對方擔憂
4. 提供具體解決方案消除顧慮
5. 連續三回合保持高說服度

**結果**：
- Agent_B 在第 4 回合投降
- Agent_A 獲得壓倒性勝利
- 最終得分：A (72.5) vs B (-25.0)

## 技術實現

### 1. 核心數據結構

```python
@dataclass
class AgentState:
    agent_id: str
    current_stance: float          # -1.0 到 1.0
    conviction: float              # 0.0 到 1.0
    social_context: List[float]    # 128 維社會向量
    persuasion_history: List[float]
    attack_history: List[float]
    has_surrendered: bool = False
```

### 2. 評分計算函數

```python
def calculate_agent_score(agent_id: str, state: AgentState, 
                         all_states: Dict[str, AgentState]) -> float:
    score = 0
    
    # 立場堅定度
    stance_score = abs(state.current_stance) * state.conviction * 30
    score += stance_score
    
    # 說服他人
    persuasion_score = 0
    for other_id, other_state in all_states.items():
        if other_id != agent_id:
            if other_state.has_surrendered:
                persuasion_score += 20
            avg_persuasion = sum(other_state.persuasion_history) / len(other_state.persuasion_history)
            persuasion_score += avg_persuasion * 10
    score += persuasion_score
    
    # 抗壓能力
    if len(state.attack_history) > 0:
        avg_attack = sum(state.attack_history) / len(state.attack_history)
        resistance_score = (1 - avg_attack) * state.conviction * 20
        score += resistance_score
    
    # 投降懲罰
    if state.has_surrendered:
        score -= 50
    
    return score
```

### 3. 效果評估函數

```python
def evaluate_response_effects(response: str, target_agents: List[str]) -> Dict:
    # 關鍵詞分析
    persuasion_indicators = ['however', 'consider', '但是', '考慮', '理解']
    attack_indicators = ['wrong', 'flawed', '錯誤', '缺陷', '謬誤']
    evidence_indicators = ['research', 'data', '研究', '數據', '證據']
    
    # 計算各項得分
    response_lower = response.lower()
    persuasion_score = min(1.0, sum(ind in response_lower for ind in persuasion_indicators) * 0.3)
    attack_score = min(1.0, sum(ind in response_lower for ind in attack_indicators) * 0.4)
    evidence_score = min(1.0, sum(ind in response_lower for ind in evidence_indicators) * 0.35)
    
    return {
        'persuasion_score': persuasion_score,
        'attack_score': attack_score,
        'evidence_score': evidence_score
    }
```

### 4. 配置參數

系統的行為可以通過 `configs/debate.yaml` 調整：

```yaml
victory_conditions:
  surrender_threshold: 0.4      # 投降的信念閾值
  stance_neutral_threshold: 0.2 # 中立立場閾值
  consecutive_persuasion: 3     # 連續被說服回合數

persuasion_factors:
  base_persuasion: 0.3         # 基礎說服力
  strategy_bonus: 0.2          # 策略加成
  evidence_bonus: 0.3          # 證據加成
  social_influence: 0.2        # 社會影響力
```

## 總結

Social Debate AI 的評分系統通過多維度評估，真實地模擬了辯論的複雜性：

1. **不只看論點**：同時評估立場、說服力、抗壓力
2. **動態博弈**：狀態隨辯論進程不斷變化
3. **策略深度**：不同策略適用於不同場景
4. **心理真實**：模擬了確認偏誤、防禦機制等心理現象

這個系統鼓勵參與者：
- 保持理性但堅定的立場
- 用證據和邏輯說服對手
- 在攻擊和防守間找到平衡
- 展現真正的辯論藝術

通過這樣的設計，AI 辯論不再是簡單的觀點輸出，而是一場需要智慧、策略和心理素質的綜合較量。 