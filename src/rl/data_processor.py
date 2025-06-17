"""
RL 訓練數據處理模組
處理 CMV 數據並生成訓練樣本
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from tqdm import tqdm

class RLDataProcessor:
    """RL 訓練數據處理器"""
    
    def __init__(self, input_path: str = "data/raw/pairs.jsonl", 
                 output_path: str = "data/rl/rl_pairs.csv"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除過長的文本
        if len(text) > 2000:
            text = text[:2000]
        
        # 移除多餘的空白和換行
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除特殊字符但保留標點
        text = re.sub(r'[^\w\s.,!?;:()\-\'"@#$%]', '', text)
        
        return text
    
    def calculate_quality_score(self, comment: Dict, submission: Dict, 
                              is_delta: bool, similarity: float) -> float:
        """計算品質分數"""
        base_score = 1.0 if is_delta else 0.0
        
        # 評論分數 (標準化到 0-1)
        comment_score = max(0, min(comment.get('score', 0), 50)) / 50.0
        
        # 相似度分數
        similarity_score = max(0, min(similarity, 1.0))
        
        # 長度分數 (適中長度獲得更高分數)
        text_length = len(comment.get('body', ''))
        if 50 <= text_length <= 500:
            length_score = 1.0
        elif text_length < 50:
            length_score = text_length / 50.0
        else:
            length_score = max(0.5, 1.0 - (text_length - 500) / 1000.0)
        
        # 結構分數 (包含論證結構的評論獲得更高分數)
        body = comment.get('body', '').lower()
        structure_indicators = ['because', 'therefore', 'however', 'furthermore', 
                               'moreover', 'in addition', 'for example', 'studies show']
        structure_score = min(1.0, sum(1 for indicator in structure_indicators if indicator in body) / 3.0)
        
        # 綜合分數
        if is_delta:
            # Delta 評論：更重視品質指標
            final_score = (
                0.4 * base_score +
                0.25 * comment_score +
                0.15 * similarity_score +
                0.1 * length_score +
                0.1 * structure_score
            )
        else:
            # Non-delta 評論：相對較低的基礎分數
            final_score = (
                0.2 * base_score +
                0.3 * comment_score +
                0.2 * similarity_score +
                0.15 * length_score +
                0.15 * structure_score
            )
        
        # 縮放到 0-2 範圍
        return final_score * 2.0
    
    def extract_features(self, submission: Dict, comment: Dict) -> Dict:
        """提取特徵"""
        # 提交文本
        title = submission.get('title', '')
        selftext = submission.get('selftext', '')
        submission_text = f"{title} {selftext}".strip()
        
        # 評論文本
        comment_text = comment.get('body', '')
        
        # 清理文本
        submission_text = self.clean_text(submission_text)
        comment_text = self.clean_text(comment_text)
        
        # 組合文本
        combined_text = f"Submission: {submission_text} Comment: {comment_text}"
        
        return {
            'text': combined_text,
            'submission_text': submission_text,
            'comment_text': comment_text,
            'comment_score': comment.get('score', 0),
            'submission_score': submission.get('score', 0)
        }
    
    def process_data(self) -> pd.DataFrame:
        """處理數據並生成訓練樣本"""
        print(f"📂 處理數據文件: {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"找不到數據文件: {self.input_path}")
        
        samples = []
        valid_count = 0
        total_count = 0
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="處理數據"):
                try:
                    data = json.loads(line.strip())
                    total_count += 1
                    
                    submission = data.get('submission', {})
                    similarity = data.get('comments_similarity', 0.0)
                    
                    # 處理 delta 評論
                    delta_comment = data.get('delta_comment')
                    if delta_comment and delta_comment.get('body'):
                        features = self.extract_features(submission, delta_comment)
                        if features['text'] and len(features['text']) > 20:
                            score = self.calculate_quality_score(
                                delta_comment, submission, True, similarity
                            )
                            
                            samples.append({
                                'text': features['text'],
                                'score': score,
                                'is_delta': True,
                                'comment_score': features['comment_score'],
                                'similarity': similarity
                            })
                            valid_count += 1
                    
                    # 處理 non-delta 評論
                    nodelta_comment = data.get('nodelta_comment')
                    if nodelta_comment and nodelta_comment.get('body'):
                        features = self.extract_features(submission, nodelta_comment)
                        if features['text'] and len(features['text']) > 20:
                            score = self.calculate_quality_score(
                                nodelta_comment, submission, False, similarity
                            )
                            
                            samples.append({
                                'text': features['text'],
                                'score': score,
                                'is_delta': False,
                                'comment_score': features['comment_score'],
                                'similarity': similarity
                            })
                            valid_count += 1
                            
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        print(f"📊 數據處理完成:")
        print(f"  總記錄數: {total_count}")
        print(f"  有效樣本: {valid_count}")
        print(f"  有效率: {valid_count/total_count*100:.1f}%")
        
        # 轉換為 DataFrame
        df = pd.DataFrame(samples)
        
        if len(df) == 0:
            raise ValueError("沒有生成有效的訓練樣本")
        
        # 數據統計
        print(f"\n📈 數據統計:")
        print(f"  Delta 樣本: {df['is_delta'].sum()} ({df['is_delta'].mean()*100:.1f}%)")
        print(f"  Non-delta 樣本: {(~df['is_delta']).sum()} ({(~df['is_delta']).mean()*100:.1f}%)")
        print(f"  分數範圍: {df['score'].min():.3f} ~ {df['score'].max():.3f}")
        print(f"  平均分數: {df['score'].mean():.3f}")
        print(f"  分數標準差: {df['score'].std():.3f}")
        
        return df
    
    def save_data(self, df: pd.DataFrame):
        """保存處理後的數據"""
        print(f"💾 保存數據到: {self.output_path}")
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        
        # 保存統計信息
        stats_path = self.output_path.parent / "data_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("RL 訓練數據統計\n")
            f.write("=" * 30 + "\n")
            f.write(f"總樣本數: {len(df)}\n")
            f.write(f"Delta 樣本: {df['is_delta'].sum()}\n")
            f.write(f"Non-delta 樣本: {(~df['is_delta']).sum()}\n")
            f.write(f"分數範圍: {df['score'].min():.3f} ~ {df['score'].max():.3f}\n")
            f.write(f"平均分數: {df['score'].mean():.3f}\n")
            f.write(f"分數標準差: {df['score'].std():.3f}\n")
            f.write(f"\n分數分布:\n")
            f.write(str(df['score'].describe()))
        
        print(f"📊 統計信息保存到: {stats_path}")
    
    def run(self) -> pd.DataFrame:
        """執行完整的數據處理流程"""
        print("🚀 開始 RL 數據處理...")
        
        # 處理數據
        df = self.process_data()
        
        # 保存數據
        self.save_data(df)
        
        print("✅ 數據處理完成！")
        return df

def main():
    """主函數"""
    processor = RLDataProcessor()
    df = processor.run()
    print(f"\n🎉 成功生成 {len(df)} 個訓練樣本")

if __name__ == "__main__":
    main() 