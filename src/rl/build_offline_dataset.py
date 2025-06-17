"""
離線數據集構建器 (舊版本，建議使用 data_processor.py)
保留用於向後兼容
"""

import json
import pathlib
import tqdm
import csv
from pathlib import Path
import sys

# 添加 src 路徑以便導入其他模組
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rag.retriever import CMVRetriever
except ImportError:
    print("⚠️  警告：無法導入 CMVRetriever，將使用簡化版本")
    CMVRetriever = None

PAIRS = pathlib.Path("data/raw/pairs.jsonl")
OUT = pathlib.Path("data/rl/offline_rl.jsonl")

def reward(delta: bool, sim: float, score: int) -> float:
    """計算獎勵分數"""
    base = 1.0 if delta else 0.0
    topic = 0.5 * sim
    qual = 0.5 * min(score, 20) / 20
    return base + (topic + qual if delta else topic)

def simple_retrieve(query: str, k: int = 5) -> list:
    """簡化版檢索器，當 CMVRetriever 不可用時使用"""
    # 這裡應該實現簡單的檢索邏輯
    # 暫時返回空列表
    return [f"Retrieved snippet {i+1} for: {query[:50]}..." for i in range(k)]

def main():
    """主函數"""
    print("🚀 構建離線 RL 數據集...")
    print("⚠️  注意：這是舊版本，建議使用 data_processor.py")
    
    # 檢查輸入文件
    if not PAIRS.exists():
        print(f"❌ 找不到輸入文件: {PAIRS}")
        print("請確保 data/raw/pairs.jsonl 存在")
        return
    
    # 創建輸出目錄
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化檢索器
    if CMVRetriever is not None:
        try:
            retr = CMVRetriever(quality="high", k=5)
            print("✅ CMVRetriever 初始化成功")
        except Exception as e:
            print(f"⚠️  CMVRetriever 初始化失敗: {e}")
            print("使用簡化版檢索器")
            retr = None
    else:
        retr = None
    
    processed_count = 0
    success_count = 0
    
    with OUT.open("w", encoding="utf-8") as w:
        with PAIRS.open(encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="處理數據對"):
                try:
                    p = json.loads(line.strip())
                    processed_count += 1
                    
                    sub = p.get("submission", {})
                    state = sub.get("title", "") + "\n" + sub.get("selftext", "")
                    
                    # 檢索相關片段
                    if retr is not None:
                        try:
                            pool = retr.retrieve(state)
                        except:
                            pool = simple_retrieve(state)
                    else:
                        pool = simple_retrieve(state)
                    
                    if len(pool) < 5:
                        continue
                    
                    # 處理成功樣本 (delta comment)
                    dc = p.get("delta_comment")
                    if dc and dc.get("body"):
                        sim = p.get("comments_similarity", 0.0)
                        r = reward(True, sim, dc.get("score", 0))
                        
                        # 找到最佳匹配的片段
                        best = 0  # 簡化版本，總是選擇第一個
                        if isinstance(pool, list) and len(pool) > 0:
                            # 嘗試找到最相關的片段
                            dc_body = dc.get("body", "").lower()
                            for i, snippet in enumerate(pool):
                                if isinstance(snippet, str) and any(word in snippet.lower() for word in dc_body.split()[:5]):
                                    best = i
                                    break
                        
                        sample = {
                            "state": state,
                            "pool": pool,
                            "action": best,
                            "reward": r,
                            "type": "delta"
                        }
                        w.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        success_count += 1
                    
                    # 處理失敗樣本 (non-delta comment)
                    nc = p.get("nodelta_comment")
                    if nc and nc.get("body"):
                        sim = p.get("comments_similarity", 0.0)
                        r = reward(False, sim, nc.get("score", 0))
                        
                        sample = {
                            "state": state,
                            "pool": pool,
                            "action": 0,  # 簡化版本
                            "reward": r,
                            "type": "nodelta"
                        }
                        w.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        success_count += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    continue
                except Exception as e:
                    print(f"處理錯誤: {e}")
                    continue
    
    print(f"✅ 離線數據集構建完成")
    print(f"  處理記錄: {processed_count}")
    print(f"  生成樣本: {success_count}")
    print(f"  輸出文件: {OUT}")
    print(f"  成功率: {success_count/processed_count*100:.1f}%" if processed_count > 0 else "  成功率: 0%")
    
    # 生成統計信息
    stats_file = OUT.parent / "offline_dataset_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("離線 RL 數據集統計\n")
        f.write("=" * 30 + "\n")
        f.write(f"處理記錄: {processed_count}\n")
        f.write(f"生成樣本: {success_count}\n")
        f.write(f"成功率: {success_count/processed_count*100:.1f}%\n" if processed_count > 0 else "成功率: 0%\n")
        f.write(f"輸出文件: {OUT}\n")
    
    print(f"📊 統計信息保存至: {stats_file}")
    
    # 建議使用新版本
    print("\n💡 建議：")
    print("  此腳本為舊版本，建議使用新的訓練 pipeline：")
    print("  python src/rl/pipeline.py")
    print("  或單獨使用數據處理器：")
    print("  python src/rl/data_processor.py")

if __name__ == "__main__":
    main() 