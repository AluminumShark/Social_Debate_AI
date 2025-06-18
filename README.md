# 🤖 Social Debate AI

基於深度學習的多智能體社會辯論系統，整合 RAG、GNN、RL 技術實現智能辯論模擬。

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ 特色功能

- 🎯 **多智能體辯論** - 3個具有不同立場的AI Agent進行動態辯論
- 📚 **RAG檢索增強** - 基於向量資料庫的證據檢索系統
- 🔗 **GNN社會網絡** - 模擬Agent間的社會關係和影響力
- 🎮 **RL策略學習** - 強化學習優化辯論策略選擇
- 🌐 **Web介面** - 現代化的Flask Web UI

## 🏗️ 系統架構總覽

```mermaid
flowchart TB
    subgraph User["👤 用戶介面"]
        UI[["🌐 Flask Web UI<br/>現代化響應式設計"]]
    end
    
    subgraph Core["🧠 核心系統"]
        direction TB
        PO[["⚡ Parallel Orchestrator<br/>並行協調器"]]
        DM[["🎭 Dialogue Manager<br/>對話管理器"]]
    end
    
    subgraph AI["🤖 AI 模組"]
        direction LR
        RAG[["📚 RAG<br/>檢索增強生成<br/>37,898 文檔"]]
        GNN[["🔗 GNN<br/>圖神經網路<br/>DGI 算法"]]
        RL[["🎮 RL<br/>強化學習<br/>4 種策略"]]
    end
    
    subgraph Agents["💭 智能體"]
        direction LR
        A[["🔴 Agent A<br/>立場: +0.8"]]
        B[["🟢 Agent B<br/>立場: -0.6"]]
        C[["🟡 Agent C<br/>立場: 0.0"]]
    end
    
    UI <==> PO
    PO <==> DM
    PO ==> RAG & GNN & RL
    DM <==> A & B & C
    
    RAG -.->|證據| DM
    GNN -.->|社會影響| DM
    RL -.->|策略選擇| DM
    
    style User fill:#FFE4E1,stroke:#FF1493,stroke-width:3px
    style Core fill:#F0E68C,stroke:#DAA520,stroke-width:3px
    style AI fill:#E0FFFF,stroke:#00CED1,stroke-width:3px
    style Agents fill:#F5F5DC,stroke:#8B4513,stroke-width:3px
```

## 🚀 快速開始

```bash
# 1. 克隆專案
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 運行系統
python run_flask.py

# 4. 訪問 http://localhost:5000
```

詳細安裝指南請見 [docs/QUICKSTART.md](docs/QUICKSTART.md)

## 📁 專案結構

```
Social_Debate_AI/
├── ui/                    # Flask Web 應用
│   ├── app.py             # 後端 API
│   ├── templates/         # HTML 模板
│   └── static/           # CSS/JS 資源
├── src/                   # 核心模組
│   ├── agents/           # Agent 實現
│   ├── rag/              # RAG 檢索系統
│   ├── gnn/              # GNN 社會網絡
│   ├── rl/               # RL 策略學習
│   └── orchestrator/     # 辯論協調器
├── configs/              # 配置檔案
├── scripts/              # 啟動腳本
├── docs/                 # 詳細文檔
└── tests/                # 測試套件
```

## 🎮 使用方式

### Web UI (推薦)
```bash
# Windows
scripts\start_flask.bat

# Linux/Mac
./scripts/start_flask.sh
```

### 命令行
```bash
python run_social_debate_ai.py
```

### 訓練模型
```bash
python train_models.py
```

## 📚 文檔

- [快速開始指南](docs/QUICKSTART.md)
- [專案結構說明](docs/PROJECT_STRUCTURE.md)
- [技術實現細節](docs/TECHNICAL_DETAILS.md)
- [訓練指南](docs/TRAINING_GUIDE.md)
- [RL使用指南](docs/RL_USAGE.md)

## 🛠️ 技術架構

```mermaid
graph TB
    subgraph "輸入層"
        A[("🎯 辯論主題<br/>Topic Input")]
    end
    
    subgraph "智能分析層"
        B[["📚 RAG 檢索<br/>Evidence Retrieval"]]
        C[["🔗 GNN 分析<br/>Social Network Analysis"]]
        D[["🎮 RL 策略<br/>Strategy Selection"]]
    end
    
    subgraph "生成層"
        E[["💬 生成回應<br/>Response Generation"]]
    end
    
    subgraph "反饋層"
        F[["🔄 更新狀態<br/>State Update"]]
    end
    
    A ==> B
    B ==> C
    C ==> D
    D ==> E
    E ==> F
    F -.->|循環迭代| B
    
    style A fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px
    style B fill:#E6F3FF,stroke:#4169E1,stroke-width:2px
    style C fill:#F0FFF0,stroke:#228B22,stroke-width:2px
    style D fill:#FFF0F5,stroke:#C71585,stroke-width:2px
    style E fill:#F0F8FF,stroke:#4682B4,stroke-width:2px
    style F fill:#FFFACD,stroke:#DAA520,stroke-width:2px
    
    classDef inputClass fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px,color:#000
    classDef processClass fill:#E6F3FF,stroke:#4169E1,stroke-width:2px,color:#000
    classDef outputClass fill:#F0F8FF,stroke:#4682B4,stroke-width:2px,color:#000
```

## 📊 數據集

本專案使用以下數據集進行訓練：

### Reddit ChangeMyView Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3778297.svg)](https://doi.org/10.5281/zenodo.3778297)

本專案的 RAG 檢索系統使用了 Reddit ChangeMyView 數據集，該數據集包含了豐富的辯論和說服性對話內容。

**引用方式**：
```bibtex
@dataset{reddit_changemyview,
  author       = {Reddit ChangeMyView Community},
  title        = {Reddit ChangeMyView Dataset},
  year         = {2020},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3778297},
  url          = {https://doi.org/10.5281/zenodo.3778297}
}
```

該數據集提供了：
- 37,898 個高質量的辯論文檔
- 多樣化的辯論主題和觀點
- 真實的說服策略和論證模式
- 豐富的元數據標註

## 🎯 辯論流程

```mermaid
graph TD
    Start([🚀 開始辯論]) --> Topic[/📝 設定主題/]
    Topic --> Init[["🔧 初始化<br/>Agent 狀態"]]
    
    Init --> Round{{"🔄 辯論回合<br/>Round N"}}
    
    Round --> Analysis[["🧪 平行分析<br/>RAG + GNN + RL"]]
    Analysis --> AgentA[["🔴 Agent A 發言<br/>策略: Aggressive"]]
    AgentA --> AgentB[["🟢 Agent B 發言<br/>策略: Analytical"]]
    AgentB --> AgentC[["🟡 Agent C 發言<br/>策略: Empathetic"]]
    
    AgentC --> Update[["📊 更新狀態<br/>立場 & 信念"]]
    Update --> Check{{⚖️ 檢查條件}}
    
    Check -->|繼續| Round
    Check -->|投降| End1[["🏳️ 有人投降<br/>辯論結束"]]
    Check -->|回合上限| End2[["⏰ 達到最大回合<br/>辯論結束"]]
    
    End1 --> Result[["🏆 計算結果<br/>判定勝負"]]
    End2 --> Result
    Result --> Export([💾 導出記錄])
    
    style Start fill:#90EE90,stroke:#006400,stroke-width:3px
    style Topic fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    style Round fill:#E6E6FA,stroke:#9370DB,stroke-width:3px
    style Analysis fill:#F0E68C,stroke:#DAA520,stroke-width:2px
    style Result fill:#FFD700,stroke:#B8860B,stroke-width:3px
    style Export fill:#87CEEB,stroke:#4682B4,stroke-width:3px
```

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

⭐ 如果這個專案對您有幫助，請給我們一個 Star！ 