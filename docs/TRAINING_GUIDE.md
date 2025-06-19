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

GNN (Graph Neural Network) 模型使用 Deep Graph Infomax (DGI) 算法學習社會網絡表示。

### 1. 訓練命令
```bash
python train_all.py --gnn
```

### 2. 訓練參數
- **圖規模**：14,307 個節點，38,606 條邊
- **嵌入維度**：128
- **訓練輪數**：200 epochs
- **訓練時間**：約 5-10 分鐘（GPU）

### 3. 監控訓練
訓練過程會顯示：
```
Epoch 20/200, 損失: 0.9506
Epoch 40/200, 損失: 0.3868
Epoch 60/200, 損失: 0.1445
```

損失應該持續下降，最終收斂到 0.02-0.05 之間。

### 4. 配置調整
編輯 `configs/gnn.yaml`：

```yaml
training:
  epochs: 200          # 可增加到 500 以獲得更好效果
  hidden_dim: 128      # 可嘗試 256 或 512
  learning_rate: 0.01  # 可調整學習率
```

## 🎮 RL 模型訓練

RL (Reinforcement Learning) 模型基於 DistilBERT，用於選擇最佳辯論策略。

### 1. 訓練命令
```bash
python train_all.py --rl
```

### 2. 數據處理
首次運行會自動處理原始數據：
- **原始數據**：10,303 個辯論對
- **處理後**：36,277 個訓練樣本
- **處理時間**：約 5 分鐘

### 3. 訓練過程
- **基礎模型**：distilbert-base-uncased
- **訓練輪數**：3 epochs（可調整）
- **批次大小**：32（RTX 3090），16（較小 GPU）
- **訓練時間**：約 15-20 分鐘（GPU）

### 4. 訓練監控
```
訓練進度: 100%|████████| 908/908 [05:23<00:00, 2.81it/s]
評估結果:
  mse: 0.1234
  mae: 0.2567
  rmse: 0.3512
  r2: 0.7890
```

### 5. 配置優化
編輯 `configs/rl.yaml`：

```yaml
training:
  epochs: 5            # 增加訓練輪數
  batch_size: 16       # 根據 GPU 記憶體調整
  learning_rate: 5e-5  # 微調學習率
  
policy_network:
  strategies:
    - aggressive       # 激進策略
    - defensive        # 防守策略
    - analytical       # 分析策略
    - empathetic      # 共情策略
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
 