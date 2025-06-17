"""
RL 訓練 Pipeline
整合數據處理、模型訓練和評估的完整流程
"""

import sys
from pathlib import Path
import argparse
import time
from typing import Optional

# 添加 src 路徑
sys.path.append(str(Path(__file__).parent.parent))

from data_processor import RLDataProcessor
from trainer import RLTrainer

class RLPipeline:
    """RL 訓練 Pipeline"""
    
    def __init__(self, 
                 input_data: str = "data/raw/pairs.jsonl",
                 processed_data: str = "data/rl/rl_pairs.csv",
                 model_output: str = "data/models/policy",
                 force_reprocess: bool = False):
        
        self.input_data = Path(input_data)
        self.processed_data = Path(processed_data)
        self.model_output = Path(model_output)
        self.force_reprocess = force_reprocess
        
        print("🚀 RL 訓練 Pipeline 初始化")
        print(f"  輸入數據: {self.input_data}")
        print(f"  處理後數據: {self.processed_data}")
        print(f"  模型輸出: {self.model_output}")
        print(f"  強制重新處理: {self.force_reprocess}")
    
    def check_data_exists(self) -> bool:
        """檢查處理後的數據是否存在"""
        return self.processed_data.exists() and not self.force_reprocess
    
    def step1_process_data(self) -> bool:
        """步驟1: 數據處理"""
        print("\n" + "="*60)
        print("📊 步驟 1: 數據處理")
        print("="*60)
        
        if self.check_data_exists():
            print(f"✅ 發現已處理的數據: {self.processed_data}")
            print("⏭️  跳過數據處理步驟")
            return True
        
        try:
            processor = RLDataProcessor(
                input_path=str(self.input_data),
                output_path=str(self.processed_data)
            )
            
            df = processor.run()
            
            if len(df) == 0:
                print("❌ 沒有生成有效的訓練樣本")
                return False
            
            print(f"✅ 數據處理完成，生成 {len(df)} 個訓練樣本")
            return True
            
        except Exception as e:
            print(f"❌ 數據處理失敗: {e}")
            return False
    
    def step2_train_model(self) -> bool:
        """步驟2: 模型訓練"""
        print("\n" + "="*60)
        print("🎯 步驟 2: 模型訓練")
        print("="*60)
        
        try:
            trainer = RLTrainer(
                data_path=str(self.processed_data),
                output_dir=str(self.model_output)
            )
            
            results = trainer.train()
            
            print(f"✅ 模型訓練完成")
            print(f"  訓練時間: {results['training_time']:.2f} 秒")
            print(f"  模型保存至: {results['model_path']}")
            
            # 顯示評估結果
            eval_results = results['eval_results']
            print(f"  最終 MSE: {eval_results['eval_mse']:.4f}")
            print(f"  最終 MAE: {eval_results['eval_mae']:.4f}")
            print(f"  最終 R²: {eval_results['eval_r2']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型訓練失敗: {e}")
            return False
    
    def step3_validate_model(self) -> bool:
        """步驟3: 模型驗證"""
        print("\n" + "="*60)
        print("🔍 步驟 3: 模型驗證")
        print("="*60)
        
        try:
            # 檢查模型文件是否存在
            model_files = [
                self.model_output / "config.json",
                self.model_output / "pytorch_model.bin",
                self.model_output / "tokenizer.json"
            ]
            
            missing_files = [f for f in model_files if not f.exists()]
            if missing_files:
                print(f"❌ 缺少模型文件: {missing_files}")
                return False
            
            # 嘗試載入模型
            print("🔄 驗證模型載入...")
            from policy_network import PolicyNetwork
            
            policy_net = PolicyNetwork(model_path=str(self.model_output))
            
            # 測試策略選擇
            test_query = "Should we implement universal healthcare?"
            strategy = policy_net.select_strategy(test_query)
            print(f"✅ 策略選擇測試通過: {strategy}")
            
            # 測試品質預測
            quality_score = policy_net.predict_quality(test_query)
            print(f"✅ 品質預測測試通過: {quality_score:.3f}")
            
            # 測試片段選擇
            test_pool = [
                {'content': 'Universal healthcare reduces costs', 'similarity_score': 0.8},
                {'content': 'Private healthcare is more efficient', 'similarity_score': 0.6}
            ]
            
            from policy_network import choose_snippet
            chosen = choose_snippet(test_query, test_pool, policy_net)
            print(f"✅ 片段選擇測試通過: {chosen[:50]}...")
            
            print("✅ 模型驗證完成，所有功能正常")
            return True
            
        except Exception as e:
            print(f"❌ 模型驗證失敗: {e}")
            return False
    
    def run(self) -> bool:
        """執行完整的訓練 pipeline"""
        print("🚀 開始 RL 訓練 Pipeline")
        start_time = time.time()
        
        # 步驟1: 數據處理
        if not self.step1_process_data():
            print("❌ Pipeline 失敗於數據處理步驟")
            return False
        
        # 步驟2: 模型訓練
        if not self.step2_train_model():
            print("❌ Pipeline 失敗於模型訓練步驟")
            return False
        
        # 步驟3: 模型驗證
        if not self.step3_validate_model():
            print("❌ Pipeline 失敗於模型驗證步驟")
            return False
        
        # 完成
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🎉 RL 訓練 Pipeline 完成！")
        print("="*60)
        print(f"⏱️  總耗時: {total_time:.2f} 秒")
        print(f"📁 模型保存位置: {self.model_output}")
        print(f"📊 處理後數據: {self.processed_data}")
        
        return True

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="RL 訓練 Pipeline")
    parser.add_argument("--input", default="data/raw/pairs.jsonl", 
                       help="輸入數據路徑")
    parser.add_argument("--output", default="data/models/policy", 
                       help="模型輸出路徑")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="強制重新處理數據")
    
    args = parser.parse_args()
    
    # 創建 pipeline
    pipeline = RLPipeline(
        input_data=args.input,
        model_output=args.output,
        force_reprocess=args.force_reprocess
    )
    
    # 執行 pipeline
    success = pipeline.run()
    
    if success:
        print("\n✅ Pipeline 執行成功！")
        exit(0)
    else:
        print("\n❌ Pipeline 執行失敗！")
        exit(1)

if __name__ == "__main__":
    main() 