"""
RL 模型評估器
評估訓練後的策略網路性能
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class RLEvaluator:
    """RL 模型評估器"""
    
    def __init__(self, model_path: str, test_data_path: str = None):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path) if test_data_path else None
        
        # 載入模型和 tokenizer
        print(f"📦 載入模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        
        # 設備檢測
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型載入完成，使用設備: {self.device}")
    
    def predict_batch(self, texts: List[str]) -> np.ndarray:
        """批量預測"""
        # Tokenization
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移到設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 預測
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()
        
        # 確保返回 1D array
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        
        return predictions
    
    def evaluate_test_set(self, test_df: pd.DataFrame = None) -> Dict:
        """評估測試集"""
        if test_df is None:
            if self.test_data_path is None:
                raise ValueError("需要提供測試數據")
            test_df = pd.read_csv(self.test_data_path)
        
        print(f"📊 評估測試集，共 {len(test_df)} 個樣本")
        
        # 批量預測
        predictions = self.predict_batch(test_df['text'].tolist())
        true_values = test_df['score'].values
        
        # 計算指標
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)
        
        # 計算相關係數
        correlation = np.corrcoef(true_values, predictions)[0, 1]
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'predictions': predictions,
            'true_values': true_values
        }
        
        print(f"📈 評估結果:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  相關係數: {correlation:.4f}")
        
        return results
    
    def analyze_by_category(self, test_df: pd.DataFrame) -> Dict:
        """按類別分析性能"""
        print("🔍 按類別分析性能...")
        
        predictions = self.predict_batch(test_df['text'].tolist())
        
        # 按 delta/non-delta 分析
        if 'is_delta' in test_df.columns:
            delta_mask = test_df['is_delta'].astype(bool)
            
            # Delta 樣本
            delta_true = test_df[delta_mask]['score'].values
            delta_pred = predictions[delta_mask]
            
            # Non-delta 樣本
            nodelta_true = test_df[~delta_mask]['score'].values
            nodelta_pred = predictions[~delta_mask]
            
            results = {
                'delta': {
                    'count': len(delta_true),
                    'mse': mean_squared_error(delta_true, delta_pred),
                    'mae': mean_absolute_error(delta_true, delta_pred),
                    'r2': r2_score(delta_true, delta_pred),
                    'correlation': np.corrcoef(delta_true, delta_pred)[0, 1] if len(delta_true) > 1 else 0
                },
                'nodelta': {
                    'count': len(nodelta_true),
                    'mse': mean_squared_error(nodelta_true, nodelta_pred),
                    'mae': mean_absolute_error(nodelta_true, nodelta_pred),
                    'r2': r2_score(nodelta_true, nodelta_pred),
                    'correlation': np.corrcoef(nodelta_true, nodelta_pred)[0, 1] if len(nodelta_true) > 1 else 0
                }
            }
            
            print(f"  Delta 樣本 ({results['delta']['count']} 個):")
            print(f"    MSE: {results['delta']['mse']:.4f}")
            print(f"    R²: {results['delta']['r2']:.4f}")
            
            print(f"  Non-delta 樣本 ({results['nodelta']['count']} 個):")
            print(f"    MSE: {results['nodelta']['mse']:.4f}")
            print(f"    R²: {results['nodelta']['r2']:.4f}")
            
            return results
        
        return {}
    
    def plot_results(self, results: Dict, save_path: str = None):
        """繪製評估結果圖表"""
        predictions = results['predictions']
        true_values = results['true_values']
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 真實值 vs 預測值散點圖
        axes[0, 0].scatter(true_values, predictions, alpha=0.6)
        axes[0, 0].plot([true_values.min(), true_values.max()], 
                       [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真實值')
        axes[0, 0].set_ylabel('預測值')
        axes[0, 0].set_title(f'真實值 vs 預測值 (R² = {results["r2"]:.3f})')
        
        # 2. 殘差圖
        residuals = true_values - predictions
        axes[0, 1].scatter(predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('預測值')
        axes[0, 1].set_ylabel('殘差')
        axes[0, 1].set_title('殘差圖')
        
        # 3. 預測值分布
        axes[1, 0].hist(predictions, bins=30, alpha=0.7, label='預測值')
        axes[1, 0].hist(true_values, bins=30, alpha=0.7, label='真實值')
        axes[1, 0].set_xlabel('分數')
        axes[1, 0].set_ylabel('頻率')
        axes[1, 0].set_title('分數分布比較')
        axes[1, 0].legend()
        
        # 4. 誤差分布
        errors = np.abs(residuals)
        axes[1, 1].hist(errors, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('絕對誤差')
        axes[1, 1].set_ylabel('頻率')
        axes[1, 1].set_title(f'絕對誤差分布 (MAE = {results["mae"]:.3f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 圖表保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def test_policy_functions(self) -> Dict:
        """測試策略功能"""
        print("🧪 測試策略功能...")
        
        from policy_network import PolicyNetwork, choose_snippet
        
        # 創建策略網路
        policy_net = PolicyNetwork(model_path=str(self.model_path))
        
        # 測試案例
        test_cases = [
            {
                'query': 'Should we implement universal healthcare?',
                'pool': [
                    {'content': 'Universal healthcare reduces costs and improves access', 'similarity_score': 0.9},
                    {'content': 'Private healthcare offers better quality', 'similarity_score': 0.7},
                    {'content': 'Healthcare is a human right', 'similarity_score': 0.8}
                ]
            },
            {
                'query': 'Is climate change caused by human activities?',
                'pool': [
                    {'content': 'Scientific consensus supports human-caused climate change', 'similarity_score': 0.95},
                    {'content': 'Natural climate variation exists', 'similarity_score': 0.6},
                    {'content': 'Carbon emissions are the main driver', 'similarity_score': 0.85}
                ]
            }
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n測試案例 {i+1}: {case['query'][:50]}...")
            
            # 策略選擇
            strategy = policy_net.select_strategy(case['query'])
            
            # 品質預測
            quality = policy_net.predict_quality(case['query'])
            
            # 片段選擇
            chosen_snippet = choose_snippet(case['query'], case['pool'], policy_net)
            
            result = {
                'query': case['query'],
                'strategy': strategy,
                'quality': quality,
                'chosen_snippet': chosen_snippet[:100] + '...' if len(chosen_snippet) > 100 else chosen_snippet
            }
            
            results.append(result)
            
            print(f"  策略: {strategy}")
            print(f"  品質分數: {quality:.3f}")
            print(f"  選擇片段: {result['chosen_snippet']}")
        
        return results
    
    def generate_report(self, output_dir: str = None) -> str:
        """生成評估報告"""
        if output_dir is None:
            output_dir = self.model_path.parent / "evaluation"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📝 生成評估報告...")
        
        # 載入測試數據
        test_data_path = self.model_path.parent.parent / "rl" / "rl_pairs.csv"
        if test_data_path.exists():
            df = pd.read_csv(test_data_path)
            # 使用 20% 作為測試集
            test_df = df.sample(frac=0.2, random_state=42)
        else:
            print("⚠️  找不到測試數據，跳過數值評估")
            test_df = None
        
        report_data = {
            'model_path': str(self.model_path),
            'evaluation_time': pd.Timestamp.now().isoformat(),
            'device': str(self.device)
        }
        
        # 數值評估
        if test_df is not None:
            eval_results = self.evaluate_test_set(test_df)
            report_data['numerical_evaluation'] = {
                'test_samples': len(test_df),
                'mse': float(eval_results['mse']),
                'mae': float(eval_results['mae']),
                'rmse': float(eval_results['rmse']),
                'r2': float(eval_results['r2']),
                'correlation': float(eval_results['correlation'])
            }
            
            # 按類別分析
            category_results = self.analyze_by_category(test_df)
            if category_results:
                report_data['category_analysis'] = category_results
            
            # 繪製圖表
            plot_path = output_dir / "evaluation_plots.png"
            self.plot_results(eval_results, str(plot_path))
        
        # 功能測試
        policy_results = self.test_policy_functions()
        report_data['policy_function_tests'] = policy_results
        
        # 保存報告
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成文本報告
        text_report_path = output_dir / "evaluation_summary.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("RL 模型評估報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"模型路徑: {self.model_path}\n")
            f.write(f"評估時間: {report_data['evaluation_time']}\n")
            f.write(f"使用設備: {report_data['device']}\n\n")
            
            if 'numerical_evaluation' in report_data:
                eval_data = report_data['numerical_evaluation']
                f.write("數值評估結果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"測試樣本數: {eval_data['test_samples']}\n")
                f.write(f"MSE: {eval_data['mse']:.4f}\n")
                f.write(f"MAE: {eval_data['mae']:.4f}\n")
                f.write(f"RMSE: {eval_data['rmse']:.4f}\n")
                f.write(f"R²: {eval_data['r2']:.4f}\n")
                f.write(f"相關係數: {eval_data['correlation']:.4f}\n\n")
            
            f.write("策略功能測試:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(policy_results):
                f.write(f"測試 {i+1}:\n")
                f.write(f"  查詢: {result['query']}\n")
                f.write(f"  策略: {result['strategy']}\n")
                f.write(f"  品質分數: {result['quality']:.3f}\n")
                f.write(f"  選擇片段: {result['chosen_snippet']}\n\n")
        
        print(f"✅ 評估報告生成完成:")
        print(f"  JSON 報告: {report_path}")
        print(f"  文本摘要: {text_report_path}")
        if test_df is not None:
            print(f"  圖表: {plot_path}")
        
        return str(report_path)

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL 模型評估器")
    parser.add_argument("--model", required=True, help="模型路徑")
    parser.add_argument("--test-data", help="測試數據路徑")
    parser.add_argument("--output", help="輸出目錄")
    
    args = parser.parse_args()
    
    evaluator = RLEvaluator(args.model, args.test_data)
    report_path = evaluator.generate_report(args.output)
    
    print(f"\n🎉 評估完成，報告保存至: {report_path}")

if __name__ == "__main__":
    main() 