# 📚 Social Debate AI 訓練指南

本指南詳細說明如何訓練 Social Debate AI 系統的三大核心模型：RAG、GNN 和 RL。

## 📋 目錄

- [環境準備](#環境準備)
- [快速訓練](#快速訓練)
- [RAG 系統訓練](#rag-系統訓練)
- [GNN 模型訓練](#gnn-模型訓練)
- [RL 模型訓練](#rl-模型訓練)
- [訓練監控](#訓練監控)
- [常見問題](#常見問題)

## 🔧 環境準備

### 1. 硬體要求
- **最低配置**：8GB RAM, 4核 CPU
- **推薦配置**：16GB RAM, RTX 3060+ GPU, 8核 CPU
- **磁碟空間**：至少 20GB（用於數據和模型）

### 2. 環境設置
```bash
# 創建虛擬環境
conda create -n social_debate python=3.8
conda activate social_debate

# 安裝 PyTorch (根據您的 CUDA 版本)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch torchvision torchaudio

# 安裝其他依賴
pip install -r requirements.txt
```

### 3. API Key 設置
```bash
# 創建 .env 文件
cp env.example .env

# 編輯 .env，添加您的 OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here
```

## 🚀 快速訓練

### 訓練所有模型（推薦）
```bash
python train_all.py --all
```

這將按順序訓練：
1. GNN 社會網絡模型
2. RL 策略選擇模型
3. RAG 檢索索引

預計總時間：30-60 分鐘（取決於硬體）

### 單獨訓練模型
```bash
# 只訓練 GNN
python train_all.py --gnn

# 只訓練 RL
python train_all.py --rl

# 只構建 RAG 索引
python train_all.py --rag        # 簡單索引（快速）
python train_all.py --rag-chroma  # Chroma 向量索引（完整）
python train_all.py --rag-both    # 兩種索引都構建
```

## 📚 RAG 系統訓練

RAG (Retrieval-Augmented Generation) 系統負責檢索相關證據支持辯論。

### 1. 簡單索引（快速測試）
```bash
python train_all.py --rag
```

- **處理時間**：約 5 分鐘
- **文檔數量**：45,974 個
- **索引大小**：約 50MB
- **適用場景**：快速測試和開發

### 2. Chroma 向量索引（生產環境）
```bash
python train_all.py --rag-chroma
```

- **處理時間**：約 20-30 分鐘
- **文檔片段**：94,525 個
- **索引大小**：約 500MB
- **嵌入成本**：約 $0.02（使用 OpenAI API）
- **適用場景**：生產環境，高質量檢索

### 3. 配置說明
編輯 `configs/rag.yaml`：

```yaml
chroma:
  embedding:
    batch_size: 500  # 增加批次大小加快處理
    model: "text-embedding-3-small"  # 可選 text-embedding-3-large

indexing:
  quality_filter:
    min_score: 10    # 最低評分過濾
    min_length: 50   # 最短文本長度
```

## 🔗 GNN 模型訓練

GNN (Graph Neural Network) 模型使用監督式學習預測說服成功率和最佳策略。

### 1. 訓練命令
```bash
python train_all.py --gnn
```

### 2. 訓練架構
- **模型類型**：GraphSAGE + GAT 注意力機制
- **任務類型**：多任務學習
  - Delta 預測（二分類）
  - 品質評分（回歸）
  - 策略分類（多分類）
- **訓練數據**：CMV 數據集的 delta/non-delta comments
- **訓練時間**：約 10-15 分鐘（GPU）

### 3. 監控訓練
訓練過程會顯示：
```
Epoch 10/50, Loss: 1.2345, Delta Acc: 0.5678, Quality MAE: 2.3456, Strategy Acc: 0.4567
Epoch 20/50, Loss: 0.8901, Delta Acc: 0.6234, Quality MAE: 1.8901, Strategy Acc: 0.5678
Epoch 30/50, Loss: 0.5678, Delta Acc: 0.6789, Quality MAE: 1.4567, Strategy Acc: 0.6234
```

### 4. 訓練效果
- **Delta 準確率**：約 67-70%
- **策略準確率**：約 64-67%
- **品質預測 MAE**：約 1.2-1.5

### 5. 配置調整
編輯 `configs/gnn.yaml`：

```yaml
training:
  epochs: 50           # 可增加到 100 以獲得更好效果
  batch_size: 32       # 根據 GPU 記憶體調整
  learning_rate: 0.001 # 可調整學習率
  
model:
  hidden_dim: 768      # BERT 嵌入維度
  num_layers: 3        # GNN 層數
  dropout: 0.1         # Dropout 率
```

## 🎮 RL 模型訓練

RL (Reinforcement Learning) 模型使用 PPO 算法學習最佳辯論策略。

### 1. 訓練命令
```bash
python train_all.py --rl
```

### 2. PPO 訓練架構
- **算法**：Proximal Policy Optimization (PPO)
- **網路架構**：Actor-Critic 雙網路
- **狀態空間**：文本嵌入 + 立場 + 回合 + 歷史
- **動作空間**：4種策略（aggressive、defensive、analytical、empathetic）
- **訓練時間**：約 20-30 分鐘（1000 episodes）

### 3. 環境設計
辯論環境模擬真實互動：
- **初始狀態**：隨機立場和信念
- **狀態轉移**：基於策略效果
- **獎勵機制**：
  - 策略效果獎勵（-1 到 +1）
  - 說服成功獎勵（+5）
  - 策略多樣性獎勵（+0.1）
- **終止條件**：投降或達到最大回合

### 4. 訓練監控
```
Episode 100/1000, Avg Reward: -2.34, Policy Loss: 0.456, Value Loss: 1.234
Episode 200/1000, Avg Reward: -0.89, Policy Loss: 0.234, Value Loss: 0.789
Episode 500/1000, Avg Reward: 1.23, Policy Loss: 0.123, Value Loss: 0.456
Episode 1000/1000, Avg Reward: 2.56, Policy Loss: 0.089, Value Loss: 0.234
```

### 5. 配置優化
編輯 `configs/rl.yaml`：

```yaml
ppo:
  episodes: 1000       # 訓練回合數
  max_steps: 50        # 每回合最大步數
  batch_size: 64       # 批次大小
  learning_rate: 3e-4  # 學習率
  gamma: 0.99          # 折扣因子
  clip_epsilon: 0.2    # PPO 裁剪參數
  
environment:
  reward_scale: 1.0    # 獎勵縮放
  persuasion_bonus: 5  # 說服成功獎勵
  diversity_bonus: 0.1 # 策略多樣性獎勵
```

## 📊 訓練監控

### 1. 檢查訓練狀態
```bash
# 查看模型文件
ls -la data/models/

# 預期輸出：
# gnn_social.pt      (GNN 模型)
# policy/            (RL 模型目錄)
# rag/               (RAG 索引)
```

### 2. 驗證模型
```bash
# 運行系統整合測試
python test_system_integrity.py
```

### 3. GPU 使用監控
```bash
# NVIDIA GPU
nvidia-smi -l 1

# 查看 GPU 記憶體使用
watch -n 1 nvidia-smi
```

## ❓ 常見問題

### Q1: CUDA out of memory
**解決方案**：
- 減小批次大小：在配置文件中調整 `batch_size`
- 使用 CPU 訓練：安裝 CPU 版本的 PyTorch
- 使用混合精度訓練（已自動啟用）

### Q2: OpenAI API 錯誤
**解決方案**：
- 檢查 API Key 是否正確設置
- 確認 API 額度充足
- 使用簡單索引代替 Chroma 索引

### Q3: 訓練時間過長
**解決方案**：
- 使用 GPU 加速
- 減少訓練輪數
- 使用預訓練模型

### Q4: 模型效果不佳
**解決方案**：
- 增加訓練輪數
- 調整學習率
- 確保數據質量

## 🎯 訓練建議

1. **首次使用**：先用默認參數訓練，熟悉流程
2. **優化效果**：逐步調整參數，觀察效果變化
3. **生產部署**：使用完整數據集和更多訓練輪數
4. **定期更新**：隨著新數據積累，定期重新訓練

## 📈 進階訓練

### 1. 自定義數據集
```python
# 準備您的數據為 JSONL 格式
# 放置在 data/raw/ 目錄
# 修改配置文件中的數據路徑
```

### 2. 模型微調
```python
# 修改模型架構
# 編輯 src/gnn/social_encoder.py
# 編輯 src/rl/policy_network.py
```

### 3. 分散式訓練
```bash
# 使用多 GPU
CUDA_VISIBLE_DEVICES=0,1 python train_all.py --all
```

---

💡 **提示**：訓練過程中如遇到問題，請查看日誌文件或提交 Issue。

📧 **支援**：如需協助，請聯繫 your-email@example.com 