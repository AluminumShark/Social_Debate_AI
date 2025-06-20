# 🚀 快速開始指南

5 分鐘內啟動並運行 Social Debate AI！

## 📋 前置要求

- Python 3.8+
- 8GB+ RAM
- Git

## 🔧 安裝步驟

### 1. 克隆專案
```bash
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI
```

### 2. 創建虛擬環境
```bash
# 使用 conda（推薦）
conda create -n social_debate python=3.8
conda activate social_debate

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

### 4. 設置 API Key（可選）
如果要使用完整的 RAG 功能：
```bash
cp env.example .env
# 編輯 .env 文件，添加您的 OpenAI API Key
```

## 🎮 快速運行

### 方式一：使用 Web UI（推薦）
```bash
# Windows
scripts\start_flask.bat

# Linux/Mac
chmod +x scripts/start_flask.sh
./scripts/start_flask.sh

# 或直接運行
python run_flask.py
```

打開瀏覽器訪問 http://localhost:5000

### 方式二：快速演示
```bash
# 使用預設模型運行演示
python quick_demo.py
```

## 📚 使用 Web UI

### 1. 初始化系統
- 打開 http://localhost:5000
- 系統會自動初始化

### 2. 設置辯論主題
輸入您想討論的主題，例如：
- "人工智慧是否應該被政府監管？"
- "基本收入是否可行？"
- "社交媒體的影響是正面還是負面？"

### 3. 開始辯論
- 點擊「下一回合」按鈕
- 觀察三個 Agent 的辯論過程
- 查看實時的立場和信念變化

### 4. 分析結果
- 系統會自動判定勝負
- 可以導出完整的辯論記錄
- 查看詳細的評分細節

### 5. 理解評分系統
系統通過以下維度評估辯論表現：
- **立場堅定度**：保持明確立場的能力
- **說服力**：影響他人觀點的能力
- **抗壓能力**：面對攻擊時的防守能力
- **綜合表現**：是否能讓對手投降

詳細評分機制請參考[辯論評分系統文檔](DEBATE_SCORING_SYSTEM.md)

## 🎓 訓練模型（可選）

如果您有原始數據並想訓練自己的模型：

### 快速訓練（演示規模）
```bash
# 訓練小規模模型（約 10 分鐘）
python train_all.py --all --demo
```

### 完整訓練
```bash
# 訓練完整模型（約 30-60 分鐘）
python train_all.py --all
```

## 🔍 驗證安裝

運行系統測試：
```bash
python test_system_integrity.py
```

預期輸出：
```
✅ GPT 接口測試通過
✅ RAG 系統測試通過
✅ GNN 系統測試通過
✅ RL 系統測試通過
✅ 系統整合測試通過
```

## ❓ 常見問題

### Q: 沒有 GPU 可以運行嗎？
A: 可以！系統會自動使用 CPU。訓練會慢一些，但推理速度影響不大。

### Q: 必須要 OpenAI API Key 嗎？
A: 不是必須的。沒有 API Key 時會使用簡單索引，功能略有限制。

### Q: 如何更改辯論參數？
A: 編輯 `configs/debate.yaml` 文件，可以調整回合數、Agent 數量等。

### Q: 系統佔用太多記憶體？
A: 可以在配置文件中減小批次大小，或使用 `--demo` 模式。

## 📖 下一步

- 查看[訓練指南](TRAINING_GUIDE.md)了解如何訓練模型
- 查看[API 文檔](API_REFERENCE.md)了解如何集成到您的應用
- 查看[部署指南](DEPLOYMENT.md)了解生產環境部署

## 🆘 需要幫助？

- 提交 [GitHub Issue](https://github.com/your-username/Social_Debate_AI/issues)
- 發送郵件至 your-email@example.com
- 查看[完整文檔](../README.md)

---

🎉 恭喜！您已經成功運行 Social Debate AI。開始探索智能辯論的世界吧！ 