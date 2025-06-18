# 📁 專案結構

```
Social_Debate_AI/
│
├── 📂 web/                     # Flask Web 應用
│   ├── 📄 app.py              # 後端 API 服務
│   ├── 📂 templates/          # HTML 模板
│   │   └── index.html         # 主頁面
│   └── 📂 static/             # 靜態資源
│       ├── 📂 css/            # 樣式表
│       └── 📂 js/             # JavaScript
│
├── 📂 src/                     # 核心源代碼
│   ├── 📂 agents/             # AI Agent 實現
│   ├── 📂 rag/                # RAG 檢索系統
│   ├── 📂 gnn/                # GNN 社會網絡
│   ├── 📂 rl/                 # RL 策略學習
│   ├── 📂 orchestrator/       # 辯論協調器
│   ├── 📂 dialogue/           # 對話管理
│   ├── 📂 gpt_interface/      # GPT 接口
│   └── 📂 utils/              # 工具函數
│
├── 📂 configs/                 # 配置文件
│   ├── debate.yaml            # 辯論配置
│   ├── gnn.yaml              # GNN 配置
│   ├── rag.yaml              # RAG 配置
│   └── rl.yaml               # RL 配置
│
├── 📂 scripts/                 # 啟動腳本
│   ├── start_flask.bat       # Windows 啟動
│   └── start_flask.sh        # Linux/Mac 啟動
│
├── 📂 docs/                    # 文檔資料
│   ├── QUICKSTART.md         # 快速開始
│   ├── TRAINING_GUIDE.md     # 訓練指南
│   └── ...                   # 其他文檔
│
├── 📂 tests/                   # 測試套件
│
├── 📂 data/                    # 數據目錄
│   ├── 📂 models/            # 訓練模型
│   └── 📂 rag/               # RAG 索引
│
├── 📄 run_flask.py            # Flask 啟動入口
├── 📄 run_social_debate_ai.py # 命令行入口
├── 📄 train_all_models.py     # 統一訓練腳本
├── 📄 requirements.txt        # Python 依賴
├── 📄 .env.example           # 環境變數範例
├── 📄 README.md              # 專案說明
└── 📄 LICENSE                # 授權文件
```

## 🔑 關鍵文件說明

### 入口文件
- **run_flask.py** - Web UI 啟動點
- **run_social_debate_ai.py** - 命令行介面
- **train_all_models.py** - 模型訓練

### 核心模組
- **src/agents/** - 定義辯論 Agent 的行為
- **src/rl/** - 強化學習策略選擇
- **src/gnn/** - 社會網絡建模
- **src/rag/** - 知識檢索系統
- **src/orchestrator/** - 協調辯論流程

### Web 應用
- **web/app.py** - Flask 後端 API
- **web/templates/** - 前端 HTML
- **web/static/** - CSS/JS 資源

### 配置與文檔
- **configs/** - 系統配置文件
- **docs/** - 詳細技術文檔 