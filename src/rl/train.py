"""
RL 訓練主程式
訓練策略網路以選擇最佳辯論策略
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rl.trainer import RLTrainer
from src.rl.data_processor import RLDataProcessor
from src.utils.config_loader import ConfigLoader
import argparse
import torch
import numpy as np
import random

def set_seed(seed: int = 517466):
    """設置所有隨機種子以確保可重複性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """主訓練函數"""
    parser = argparse.ArgumentParser(description="訓練 RL 策略網路")
    parser.add_argument("--data_path", type=str, default="data/rl/rl_pairs.csv", help="訓練數據路徑")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="基礎模型名稱")
    parser.add_argument("--output_dir", type=str, default="data/models/policy", help="輸出目錄")
    parser.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="學習率")
    parser.add_argument("--seed", type=int, default=517466, help="隨機種子")
    parser.add_argument("--process_data", action="store_true", help="是否先處理原始數據")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 RL 訓練程式")
    print("=" * 50)
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 載入配置
    config = ConfigLoader.load("rl")
    
    # 如果需要，先處理數據
    if args.process_data or not Path(args.data_path).exists():
        print("\n📊 處理原始數據...")
        raw_data_path = config.get("data_processing", {}).get("input_path", "data/raw/pairs.jsonl")
        
        # 使用 RLDataProcessor 類
        processor = RLDataProcessor(
            input_path=raw_data_path,
            output_path=args.data_path
        )
        processor.run()  # 使用 run() 方法，它會處理並保存數據
        print("✅ 數據處理完成")
    
    # 合併配置和命令行參數
    training_config = config.get("training", {})
    model_config = config.get("policy_network", {})
    
    # 創建訓練器
    trainer = RLTrainer(
        data_path=args.data_path,
        model_name=args.model_name or model_config.get("base_model", "distilbert-base-uncased"),
        output_dir=args.output_dir,
        max_length=training_config.get("max_length", 512)
    )
    
    print(f"\n配置參數:")
    print(f"  - 數據路徑: {args.data_path}")
    print(f"  - 模型名稱: {trainer.model_name}")
    print(f"  - 輸出目錄: {trainer.output_dir}")
    print(f"  - 訓練輪數: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 學習率: {args.learning_rate}")
    print(f"  - 隨機種子: {args.seed}")
    print("-" * 50)
    
    try:
        # 執行訓練
        results = trainer.train()
        
        print("\n✅ RL 訓練完成！")
        print("\n📊 訓練結果:")
        for key, value in results['eval_results'].items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
        print(f"  - 訓練時間: {results['training_time']:.2f} 秒")
        print(f"  - 模型保存於: {results['model_path']}")
        
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}")
        raise

if __name__ == "__main__":
    main() 