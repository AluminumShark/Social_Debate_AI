"""
RL 策略網路訓練器
支援 CUDA 加速和混合精度訓練
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import time
from typing import Dict, Optional

class RLTrainer:
    """RL 策略網路訓練器"""
    
    def __init__(self, 
                 data_path: str = "data/rl/rl_pairs.csv",
                 model_name: str = "distilbert-base-uncased",
                 output_dir: str = "data/models/policy",
                 max_length: int = 512):
        
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        
        # 設備檢測
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 RL 訓練器初始化")
        print(f"  數據路徑: {self.data_path}")
        print(f"  模型: {self.model_name}")
        print(f"  輸出目錄: {self.output_dir}")
        print(f"  設備: {self.device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def load_data(self) -> pd.DataFrame:
        """載入訓練數據"""
        print(f"📂 載入數據: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"找不到數據文件: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"  數據總數: {len(df)} 筆")
        print(f"  分數範圍: {df['score'].min():.3f} ~ {df['score'].max():.3f}")
        print(f"  平均分數: {df['score'].mean():.3f}")
        
        # 檢查數據品質
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f"⚠️  發現 {null_count} 個空值，將自動處理")
            df = df.dropna()
        
        # 檢查文本長度
        text_lengths = df['text'].str.len()
        print(f"  文本長度: 平均 {text_lengths.mean():.0f}, 最大 {text_lengths.max()}")
        
        return df
    
    def prepare_datasets(self, df: pd.DataFrame) -> Dict:
        """準備訓練和驗證數據集"""
        print("🔄 準備數據集...")
        
        # 分割數據
        dataset = Dataset.from_pandas(df[['text', 'score']])
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        print(f"  訓練集: {len(split_dataset['train'])} 筆")
        print(f"  驗證集: {len(split_dataset['test'])} 筆")
        
        return split_dataset
    
    def tokenize_function(self, examples, tokenizer):
        """Tokenization 函數"""
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None
        )
    
    def compute_metrics(self, eval_pred):
        """計算評估指標"""
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def get_training_args(self) -> TrainingArguments:
        """獲取訓練參數"""
        # 根據 GPU 記憶體調整批次大小
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 20:  # RTX 3090+ 
                train_batch_size = 32
                eval_batch_size = 64
            elif gpu_memory_gb >= 10:  # RTX 3080
                train_batch_size = 24
                eval_batch_size = 32
            else:  # 較小的 GPU
                train_batch_size = 16
                eval_batch_size = 16
            
            print(f"  GPU 記憶體 {gpu_memory_gb:.1f}GB，批次大小: 訓練={train_batch_size}, 評估={eval_batch_size}")
        else:
            train_batch_size = 8
            eval_batch_size = 8
            print(f"  CPU 模式，批次大小: {train_batch_size}")
        
        # 臨時目錄
        temp_dir = Path("data/_tmp/rl_training")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=str(temp_dir),
            learning_rate=5e-5,
            num_train_epochs=3,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_mse',
            greater_is_better=False,
            logging_steps=50,
            logging_dir=str(temp_dir / "logs"),
            report_to=[],
            seed=42,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # 混合精度訓練
            dataloader_num_workers=0,  # 避免多進程問題
            remove_unused_columns=False,
        )
    
    def train(self) -> Dict:
        """執行訓練"""
        print("🚀 開始訓練...")
        
        # 1. 載入數據
        df = self.load_data()
        
        # 2. 準備數據集
        datasets = self.prepare_datasets(df)
        
        # 3. 載入 tokenizer 和模型
        print(f"📦 載入模型: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
            problem_type='regression'
        )
        
        # 4. Tokenization
        print("🔤 進行 tokenization...")
        tokenized_datasets = datasets.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        
        # 重命名標籤列
        tokenized_datasets = tokenized_datasets.rename_column('score', 'labels')
        
        # 設置格式
        tokenized_datasets.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        # 5. 移動模型到 GPU
        model = model.to(self.device)
        print(f"📱 模型已移至: {next(model.parameters()).device}")
        
        # 6. 設置訓練參數
        training_args = self.get_training_args()
        
        # 7. 創建 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 8. 開始訓練
        print("🎯 開始訓練...")
        start_time = time.time()
        
        if torch.cuda.is_available():
            print(f"訓練前 GPU 記憶體: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        print(f"⏱️  訓練完成，耗時: {training_time:.2f} 秒")
        
        if torch.cuda.is_available():
            print(f"訓練後 GPU 記憶體: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU 記憶體峰值: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # 9. 最終評估
        print("📊 最終評估...")
        eval_results = trainer.evaluate()
        
        print("評估結果:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # 10. 保存模型
        print(f"💾 保存模型到: {self.output_dir}")
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        
        # 11. 保存訓練記錄
        training_log = {
            'model_name': self.model_name,
            'training_time': training_time,
            'train_samples': len(tokenized_datasets['train']),
            'eval_samples': len(tokenized_datasets['test']),
            'final_eval_results': eval_results,
            'train_results': {
                'train_loss': train_result.training_loss,
                'train_runtime': train_result.metrics['train_runtime'],
                'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            }
        }
        
        log_path = self.output_dir / "training_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        
        print(f"📝 訓練記錄保存到: {log_path}")
        
        return {
            'eval_results': eval_results,
            'training_time': training_time,
            'model_path': self.output_dir
        }

def main():
    """主函數"""
    trainer = RLTrainer()
    
    try:
        results = trainer.train()
        
        print("\n🎉 訓練完成！")
        print("📊 最終結果:")
        for key, value in results['eval_results'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        print(f"  訓練時間: {results['training_time']:.2f} 秒")
        print(f"  模型路徑: {results['model_path']}")
        
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        raise

if __name__ == "__main__":
    main() 