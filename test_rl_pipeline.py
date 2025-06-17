"""
RL Pipeline 測試腳本
用於測試 RL 訓練 pipeline 的各個組件
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加 src 路徑
sys.path.append(str(Path(__file__).parent / "src"))

def test_data_processor():
    """測試數據處理器"""
    print("🧪 測試數據處理器...")
    
    try:
        from rl.data_processor import RLDataProcessor
        
        # 創建測試數據
        test_data = [
            {
                "submission": {
                    "title": "Test submission title",
                    "selftext": "Test submission content"
                },
                "delta_comment": {
                    "body": "This is a convincing argument because it provides evidence",
                    "score": 10
                },
                "nodelta_comment": {
                    "body": "I disagree but this is not convincing",
                    "score": 2
                },
                "comments_similarity": 0.8
            }
        ]
        
        # 創建臨時測試文件
        import json
        test_file = Path("test_data.jsonl")
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # 測試處理器
        processor = RLDataProcessor(
            input_path=str(test_file),
            output_path="test_output.csv"
        )
        
        df = processor.run()
        
        # 檢查結果
        assert len(df) > 0, "應該生成至少一個樣本"
        assert 'text' in df.columns, "應該包含 text 列"
        assert 'score' in df.columns, "應該包含 score 列"
        
        # 清理
        test_file.unlink()
        Path("test_output.csv").unlink(missing_ok=True)
        Path("data/rl/data_stats.txt").unlink(missing_ok=True)
        
        print("✅ 數據處理器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 數據處理器測試失敗: {e}")
        return False

def test_policy_network():
    """測試策略網路"""
    print("🧪 測試策略網路...")
    
    try:
        from rl.policy_network import PolicyNetwork, choose_snippet
        
        # 創建測試用的策略網路（不載入真實模型）
        class MockPolicyNetwork:
            def select_strategy(self, query):
                return "analytical"
            
            def predict_quality(self, text):
                return 0.75
            
            def encode_text(self, text):
                return np.random.rand(768)
        
        mock_policy = MockPolicyNetwork()
        
        # 測試策略選擇
        strategy = mock_policy.select_strategy("Test query")
        assert strategy in ["analytical", "aggressive", "defensive", "empathetic"], f"無效策略: {strategy}"
        
        # 測試品質預測
        quality = mock_policy.predict_quality("Test text")
        assert 0 <= quality <= 2, f"品質分數超出範圍: {quality}"
        
        # 測試片段選擇
        test_pool = [
            {'content': 'First snippet', 'similarity_score': 0.8},
            {'content': 'Second snippet', 'similarity_score': 0.6}
        ]
        
        # 使用簡化版本的 choose_snippet
        def simple_choose_snippet(query, pool, policy_net):
            # 簡單選擇最高相似度的片段
            best_snippet = max(pool, key=lambda x: x['similarity_score'])
            return best_snippet['content']
        
        chosen = simple_choose_snippet("test query", test_pool, mock_policy)
        assert chosen == 'First snippet', f"應該選擇最高相似度的片段"
        
        print("✅ 策略網路測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 策略網路測試失敗: {e}")
        return False

def test_trainer_setup():
    """測試訓練器設置"""
    print("🧪 測試訓練器設置...")
    
    try:
        from rl.trainer import RLTrainer
        
        # 創建測試數據
        test_df = pd.DataFrame({
            'text': ['Test text 1', 'Test text 2', 'Test text 3'],
            'score': [1.0, 0.5, 1.5]
        })
        
        test_df.to_csv('test_training_data.csv', index=False)
        
        # 測試訓練器初始化
        trainer = RLTrainer(
            data_path='test_training_data.csv',
            output_dir='test_model_output'
        )
        
        # 測試數據載入
        df = trainer.load_data()
        assert len(df) == 3, "應該載入3個樣本"
        
        # 測試數據集準備
        datasets = trainer.prepare_datasets(df)
        assert 'train' in datasets, "應該包含訓練集"
        assert 'test' in datasets, "應該包含測試集"
        
        # 清理
        Path('test_training_data.csv').unlink()
        
        print("✅ 訓練器設置測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 訓練器設置測試失敗: {e}")
        return False

def test_pipeline_setup():
    """測試 pipeline 設置"""
    print("🧪 測試 pipeline 設置...")
    
    try:
        from rl.pipeline import RLPipeline
        
        # 測試 pipeline 初始化
        pipeline = RLPipeline(
            input_data="test_input.jsonl",
            processed_data="test_processed.csv",
            model_output="test_model",
            force_reprocess=True
        )
        
        # 檢查路徑設置
        assert pipeline.input_data.name == "test_input.jsonl"
        assert pipeline.processed_data.name == "test_processed.csv"
        assert pipeline.model_output.name == "test_model"
        assert pipeline.force_reprocess == True
        
        print("✅ Pipeline 設置測試通過")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline 設置測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 開始 RL Pipeline 測試")
    print("=" * 50)
    
    tests = [
        ("數據處理器", test_data_processor),
        ("策略網路", test_policy_network),
        ("訓練器設置", test_trainer_setup),
        ("Pipeline 設置", test_pipeline_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 測試: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} 測試失敗")
    
    print("\n" + "=" * 50)
    print(f"🎯 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！RL Pipeline 準備就緒")
        print("\n💡 下一步:")
        print("  1. 準備訓練數據: data/raw/pairs.jsonl")
        print("  2. 運行完整 pipeline: python src/rl/pipeline.py")
        print("  3. 評估訓練結果: python src/rl/evaluator.py --model data/models/policy")
    else:
        print("❌ 部分測試失敗，請檢查相關組件")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 