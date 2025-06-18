# RL 訓練指南

## 概述

本指南介紹如何使用 Social Debate AI 系統的 RL (強化學習) 訓練 pipeline。RL 模組負責訓練策略網路，用於智能選擇論證片段和決定辯論策略。

## 系統架構

```
src/rl/
├── data_processor.py      # 數據處理模組
├── trainer.py            # 訓練器模組
├── pipeline.py           # 完整訓練 pipeline
├── evaluator.py          # 模型評估器
├── policy_network.py     # 策略網路核心
└── build_offline_dataset.py  # 離線數據集構建 (舊版)
```

## 快速開始

### 1. 環境準備

確保已安裝必要的依賴：

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn
```

### 2. 數據準備

將原始 CMV 數據放置在 `data/raw/pairs.jsonl`：

```
data/
├── raw/
│   └── pairs.jsonl       # 原始 CMV 數據
├── rl/
│   └── rl_pairs.csv      # 處理後的訓練數據 (自動生成)
└── models/
    └── policy/           # 訓練後的模型 (自動生成)
```

### 3. 一鍵訓練

使用完整的訓練 pipeline：

```bash
# 基本訓練
python src/rl/pipeline.py

# 自定義參數
python src/rl/pipeline.py --input data/raw/pairs.jsonl --output data/models/my_policy --force-reprocess
```

### 4. 分步訓練

如果需要更細粒度的控制：

```bash
# 步驟 1: 數據處理
python src/rl/data_processor.py

# 步驟 2: 模型訓練
python src/rl/trainer.py

# 步驟 3: 模型評估
python src/rl/evaluator.py --model data/models/policy
```

## 詳細說明

### 數據處理 (data_processor.py)

**功能**：
- 處理原始 CMV 數據
- 提取特徵和計算品質分數
- 生成訓練樣本

**品質分數計算**：
- **Delta 評論** (成功說服)：基礎分數 1.0 + 評論分數 + 相似度 + 長度分數 + 結構分數
- **Non-delta 評論** (未說服)：較低的基礎分數 + 其他指標
- 最終分數範圍：0-2

**使用方法**：
```python
from src.rl.data_processor import RLDataProcessor

processor = RLDataProcessor(
    input_path="data/raw/pairs.jsonl",
    output_path="data/rl/rl_pairs.csv"
)
df = processor.run()
```

### 模型訓練 (trainer.py)

**功能**：
- 基於 DistilBERT 的回歸模型
- 支援 CUDA 加速和混合精度訓練
- 自動調整批次大小
- 早停機制

**訓練參數**：
- 學習率：5e-5
- 訓練輪數：3
- 批次大小：根據 GPU 記憶體自動調整
- 評估指標：MSE, MAE, RMSE, R²

**使用方法**：
```python
from src.rl.trainer import RLTrainer

trainer = RLTrainer(
    data_path="data/rl/rl_pairs.csv",
    model_name="distilbert-base-uncased",
    output_dir="data/models/policy"
)
results = trainer.train()
```

### 完整 Pipeline (pipeline.py)

**功能**：
- 整合數據處理、訓練、驗證
- 自動檢查依賴和中間結果
- 詳細的進度報告

**使用方法**：
```bash
# 命令行使用
python src/rl/pipeline.py [選項]

# 程式化使用
from src.rl.pipeline import RLPipeline

pipeline = RLPipeline(
    input_data="data/raw/pairs.jsonl",
    model_output="data/models/policy",
    force_reprocess=False
)
success = pipeline.run()
```

### 模型評估 (evaluator.py)

**功能**：
- 數值評估：MSE, MAE, R², 相關係數
- 按類別分析：Delta vs Non-delta 性能
- 功能測試：策略選擇、品質預測、片段選擇
- 視覺化報告

**使用方法**：
```python
from src.rl.evaluator import RLEvaluator

evaluator = RLEvaluator("data/models/policy")
report_path = evaluator.generate_report("data/models/policy/evaluation")
```

## 策略網路使用

訓練完成後，可以在辦論系統中使用策略網路：

```python
from src.rl.policy_network import PolicyNetwork, choose_snippet

# 載入策略網路
policy_net = PolicyNetwork("data/models/policy")

# 策略選擇
strategy = policy_net.select_strategy("Should we implement universal healthcare?")
# 返回: "analytical", "aggressive", "defensive", "empathetic"

# 品質預測
quality = policy_net.predict_quality("This is a strong argument...")
# 返回: 0.0-2.0 的品質分數

# 片段選擇
evidence_pool = [
    {'content': 'Evidence 1...', 'similarity_score': 0.8},
    {'content': 'Evidence 2...', 'similarity_score': 0.6}
]
chosen = choose_snippet("Query", evidence_pool, policy_net)
# 返回: 選擇的最佳片段內容
```

## 訓練監控

### GPU 使用情況

訓練器會自動檢測 GPU 並調整參數：
- **RTX 3090+ (≥20GB)**：批次大小 32/64
- **RTX 3080 (≥10GB)**：批次大小 24/32  
- **其他 GPU**：批次大小 16/16
- **CPU 模式**：批次大小 8/8

### 訓練記錄

訓練完成後會生成：
- `training_log.json`：詳細訓練記錄
- `config.json`：模型配置
- `pytorch_model.bin`：模型權重
- `tokenizer.json`：分詞器

### 評估報告

評估器會生成：
- `evaluation_report.json`：JSON 格式詳細報告
- `evaluation_summary.txt`：文本摘要
- `evaluation_plots.png`：視覺化圖表

## 故障排除

### 常見問題

1. **記憶體不足**
   ```
   解決：降低批次大小或使用 CPU 模式
   ```

2. **數據文件不存在**
   ```
   解決：確保 data/raw/pairs.jsonl 存在
   ```

3. **模型載入失敗**
   ```
   解決：檢查模型路徑和文件完整性
   ```

### 調試模式

啟用詳細日誌：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 性能優化

1. **使用 SSD 儲存數據**
2. **確保充足的 RAM (建議 16GB+)**
3. **使用 CUDA 11.8+ 和 PyTorch 2.0+**

## 高級配置

### 自定義模型

```python
trainer = RLTrainer(
    model_name="bert-base-uncased",  # 使用 BERT 替代 DistilBERT
    max_length=1024,                # 增加序列長度
    output_dir="data/models/bert_policy"
)
```

### 自定義品質分數

修改 `data_processor.py` 中的 `calculate_quality_score` 方法：

```python
def calculate_quality_score(self, comment, submission, is_delta, similarity):
    # 自定義評分邏輯
    custom_score = your_scoring_function(comment, submission, is_delta, similarity)
    return custom_score
```

### 多 GPU 訓練

```python
# 在 trainer.py 中啟用 DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## 整合到辯論系統

訓練完成的模型可以整合到主要的辯論系統中：

```python
# 在 orchestrator 中使用
from src.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator()
reply = orchestrator.get_rl_enhanced_reply(
    query="Should we implement universal healthcare?",
    context="Previous discussion context...",
    agent_id="agent_a"
)
```

## 版本管理

建議為不同的訓練實驗使用版本化的模型目錄：

```
data/models/
├── policy_v1.0/          # 基礎版本
├── policy_v1.1/          # 改進品質分數
├── policy_v2.0/          # 使用 BERT
└── policy_experimental/   # 實驗版本
```

## 性能基準

### 預期性能指標

在標準 CMV 數據集上的預期性能：
- **MSE**: < 0.1
- **MAE**: < 0.25  
- **R²**: > 0.7
- **相關係數**: > 0.8

### 訓練時間

在不同硬體上的預期訓練時間（10K 樣本）：
- **RTX 3090**: ~15-20 分鐘
- **RTX 3080**: ~25-30 分鐘
- **RTX 3060**: ~45-60 分鐘
- **CPU**: ~2-3 小時

## 結論

RL 訓練 pipeline 提供了完整的端到端解決方案，從原始數據處理到模型部署。通過合理配置和監控，可以訓練出高性能的策略網路，顯著提升辯論系統的智能化水平。

如有問題，請參考代碼註釋或聯繫開發團隊。 