# 📚 Social Debate AI 文檔中心

歡迎來到 Social Debate AI 的文檔中心！本目錄包含系統的所有技術文檔和使用指南。

## 🗂️ 文檔索引

### 📋 基礎文檔

| 文檔 | 說明 | 適合對象 |
|------|------|----------|
| [QUICKSTART.md](QUICKSTART.md) | 快速開始指南 | 初次使用者 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 專案結構說明 | 開發者 |

### 🔬 技術文檔

| 文檔 | 說明 | 重要性 |
|------|------|--------|
| [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) | 技術實現細節 | ⭐⭐⭐⭐⭐ |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 模型訓練指南 | ⭐⭐⭐⭐ |
| [RL_USAGE.md](RL_USAGE.md) | RL 模組使用指南 | ⭐⭐⭐ |
| [API_REFERENCE.md](API_REFERENCE.md) | Web API 參考 | ⭐⭐⭐ |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 部署指南 | ⭐⭐⭐⭐ |

## 🎯 閱讀路線圖

### 👶 初學者路線
```mermaid
graph LR
    A[QUICKSTART.md] --> B[主目錄 README.md]
    B --> C[PROJECT_STRUCTURE.md]
    C --> D[運行 Web UI]
```

### 👨‍💻 開發者路線
```mermaid
graph LR
    A[PROJECT_STRUCTURE.md] --> B[TECHNICAL_DETAILS.md]
    B --> C[TRAINING_GUIDE.md]
    C --> D[RL_USAGE.md]
    D --> E[開始開發]
```

### 🔬 研究者路線
```mermaid
graph LR
    A[TECHNICAL_DETAILS.md] --> B[深入算法原理]
    B --> C[TRAINING_GUIDE.md]
    C --> D[實驗複現]
```

## 📖 文檔內容概覽

### QUICKSTART.md
- 環境配置步驟
- 快速運行指令
- 常見問題解答
- 基本使用範例

### PROJECT_STRUCTURE.md
- 完整目錄結構圖
- 各模組功能說明
- 檔案命名規範
- 代碼組織邏輯

### TECHNICAL_DETAILS.md ⭐
- **系統架構**：完整的系統設計和數據流
- **RAG 實現**：Chroma 向量資料庫和檢索策略
- **GNN 原理**：Deep Graph Infomax 算法詳解
- **RL 機制**：策略網路和 Thompson Sampling
- **並行協調**：異步執行和結果融合
- **勝負判定**：投降條件和評分系統

### TRAINING_GUIDE.md
- **數據準備**：訓練數據格式和預處理
- **RAG 訓練**：建立向量索引的完整流程
- **GNN 訓練**：圖神經網路的訓練步驟
- **RL 訓練**：強化學習的訓練策略
- **訓練技巧**：超參數調整和最佳實踐

### RL_USAGE.md
- **模組架構**：PolicyNetwork 類別設計
- **策略詳解**：四種辯論策略的使用場景
- **API 範例**：程式碼使用範例
- **整合指南**：與其他模組的整合方法

### API_REFERENCE.md
- **端點列表**：所有 Web API 端點
- **請求格式**：詳細的請求和響應格式
- **使用範例**：Python 和 JavaScript 範例
- **錯誤處理**：統一的錯誤格式說明

### DEPLOYMENT.md
- **部署選項**：本地、雲端、Docker
- **生產配置**：Nginx、Gunicorn 設置
- **安全措施**：HTTPS、防火牆配置
- **性能優化**：緩存、負載均衡策略

## 🚀 快速導航

### 想要快速運行系統？
👉 查看 [QUICKSTART.md](QUICKSTART.md)

### 想要了解技術原理？
👉 查看 [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)

### 想要訓練自己的模型？
👉 查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### 想要使用 RL 模組？
👉 查看 [RL_USAGE.md](RL_USAGE.md)

## 💡 實用提示

1. **文檔更新**：所有文檔會隨著系統更新而維護
2. **範例代碼**：每個技術文檔都包含可運行的範例
3. **問題反饋**：如有疑問，歡迎在 GitHub 提出 Issue

## 📊 系統特色一覽

| 特色 | 技術 | 文檔 |
|------|------|------|
| 智能檢索 | RAG + Chroma | TECHNICAL_DETAILS.md |
| 社會建模 | GNN + DGI | TECHNICAL_DETAILS.md |
| 策略學習 | RL + Thompson | RL_USAGE.md |
| 並行處理 | AsyncIO | TECHNICAL_DETAILS.md |
| Web 介面 | Flask + Bootstrap | QUICKSTART.md |

## 🔄 最近更新

- **2024-01** - 新增 TECHNICAL_DETAILS.md 完整技術文檔
- **2024-01** - 更新 Flask Web UI 使用說明
- **2024-01** - 優化文檔結構和導航

---

📧 如有任何問題或建議，歡迎聯繫我們！ 