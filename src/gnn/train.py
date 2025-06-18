"""
GNN 訓練主程式
使用 Deep Graph Infomax (DGI) 訓練社會網絡嵌入
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.gnn.train_dgi import train_dgi
from src.utils.config_loader import ConfigLoader
import argparse
import torch

def main():
    """主訓練函數"""
    parser = argparse.ArgumentParser(description="訓練 GNN 社會網絡模型")
    parser.add_argument("--epochs", type=int, default=200, help="訓練輪數")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隱藏層維度")
    parser.add_argument("--batch_size", type=int, default=4096, help="批次大小")
    parser.add_argument("--output", type=str, default="data/models/gnn_social.pt", help="輸出路徑")
    parser.add_argument("--seed", type=int, default=517466, help="隨機種子")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 GNN 訓練程式")
    print("=" * 50)
    
    # 載入配置
    config = ConfigLoader.load("gnn")
    
    # 合併配置和命令行參數
    epochs = args.epochs or config.get("training", {}).get("epochs", 200)
    hidden_dim = args.hidden_dim or config.get("model", {}).get("hidden_dim", 128)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 4096)
    output_path = args.output or "data/models/gnn_social.pt"
    
    # 設置隨機種子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"配置參數:")
    print(f"  - 訓練輪數: {epochs}")
    print(f"  - 隱藏維度: {hidden_dim}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 輸出路徑: {output_path}")
    print(f"  - 隨機種子: {args.seed}")
    print("-" * 50)
    
    try:
        # 執行訓練
        train_dgi(
            epochs=epochs,
            hid=hidden_dim,
            batch_size=batch_size,
            out=output_path
        )
        print("\n✅ GNN 訓練完成！")
        
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}")
        raise

if __name__ == "__main__":
    main() 