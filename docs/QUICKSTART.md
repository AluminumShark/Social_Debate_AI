# 🚀 Social Debate AI 快速開始指南

本指南將幫助您在 5 分鐘內啟動並運行 Social Debate AI 系統。

## 📋 前置需求

- Python 3.8 或更高版本
- Git
- 至少 8GB RAM
- OpenAI API Key（可選，用於真實 GPT 回應）

## 🔧 安裝步驟

### 1. 克隆專案

```bash
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI
```

### 2. 創建虛擬環境（推薦）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 設置環境變數（可選）

創建 `.env` 文件：

```bash
# 複製範例文件
cp .env.example .env

# 編輯 .env 文件，添加您的 OpenAI API Key
OPENAI_API_KEY=your-api-key-here
```

## 🎮 快速運行

### 方式一：使用啟動腳本（推薦）

```bash
# Windows
scripts\start_flask.bat

# Linux/Mac
chmod +x scripts/start_flask.sh
./scripts/start_flask.sh
```

### 方式二：直接運行

```bash
python run_flask.py
```

### 方式三：開發模式

```bash
cd ui
python app.py
```

## 🌐 訪問系統

啟動後，在瀏覽器中訪問：

- 本地訪問：http://localhost:5000
- 網路訪問：http://[您的IP]:5000

## 📱 使用界面

### 主界面說明

```
┌─────────────────────────────────────────────────┐
│                 Social Debate AI                 │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ 辯論設置     │  │ 辯論內容               │  │
│  │              │  │                         │  │
│  │ 主題: _____ │  │ [辯論對話顯示區域]     │  │
│  │ [設置主題]   │  │                         │  │
│  │ [下一回合]   │  │                         │  │
│  │              │  │                         │  │
│  │ Agent 狀態   │  │                         │  │
│  │ A: ████░░░░ │  │                         │  │
│  │ B: ██░░░░░░ │  │                         │  │
│  │ C: █████░░░ │  │                         │  │
│  └──────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 操作步驟

1. **設置辯論主題**
   - 在左側輸入框中輸入辯論主題
   - 點擊「設置主題」按鈕

2. **開始辯論**
   - 點擊「下一回合」按鈕
   - 系統將自動進行一輪辯論

3. **觀察進度**
   - 查看右側的辯論內容
   - 觀察左側 Agent 的狀態變化

4. **辯論結束**
   - 當有 Agent 投降或達到最大回合數時結束
   - 系統會顯示勝負結果

5. **導出或重置**
   - 點擊頂部「導出」保存辯論記錄
   - 點擊「重置」開始新的辯論

## 🎯 範例主題

試試這些有趣的辯論主題：

- 人工智慧是否應該由政府監管？
- 遠程工作是否比辦公室工作更有效率？
- 社交媒體對社會的影響是正面還是負面？
- 加密貨幣是否會取代傳統貨幣？
- 基因編輯技術是否應該用於人類增強？

## 🛠️ 命令行模式

如果您想使用命令行界面：

```bash
python run_social_debate_ai.py
```

## 🔧 常見問題

### Q1: 出現 "Module not found" 錯誤
**A:** 確保已安裝所有依賴：
```bash
pip install -r requirements.txt
```

### Q2: OpenAI API 錯誤
**A:** 檢查您的 API Key 是否正確設置在 `.env` 文件中

### Q3: 端口被佔用
**A:** 修改 `run_flask.py` 中的端口號：
```python
app.run(host='0.0.0.0', port=5001)  # 改為其他端口
```

### Q4: 無法訪問網頁
**A:** 檢查防火牆設置，確保允許 5000 端口

## 📊 系統需求

### 最低需求
- CPU: 雙核處理器
- RAM: 8GB
- 硬碟: 5GB 可用空間

### 推薦配置
- CPU: 四核處理器或更高
- RAM: 16GB
- GPU: NVIDIA GPU（用於加速推理）
- 硬碟: 10GB 可用空間

## 🎓 下一步

恭喜！您已成功運行 Social Debate AI。接下來您可以：

1. **深入了解系統**
   - 閱讀 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解專案結構
   - 查看 [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) 了解技術原理

2. **訓練自己的模型**
   - 參考 [RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md) 訓練 RL 模型
   - 查看 [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) 了解其他模組訓練

3. **參與開發**
   - Fork 專案並提交 Pull Request
   - 在 Issues 中報告問題或建議

## 💡 提示

- 第一次運行可能需要下載模型，請耐心等待
- 使用真實的 OpenAI API 可獲得更好的辯論效果
- 嘗試不同的辯論主題以探索系統能力

---

如有任何問題，歡迎查看其他文檔或提出 Issue！ 