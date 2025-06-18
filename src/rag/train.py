"""
RAG 索引構建主程式
構建向量資料庫索引以支援檢索增強生成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.build_index import build_chroma_index, build_simple_index
from src.utils.config_loader import ConfigLoader
import argparse
import json

def main():
    """主構建函數"""
    parser = argparse.ArgumentParser(description="構建 RAG 檢索索引")
    parser.add_argument("--type", type=str, choices=["chroma", "simple", "both"], 
                       default="both", help="索引類型")
    parser.add_argument("--data_path", type=str, default="data/raw/pairs.jsonl", 
                       help="原始數據路徑")
    parser.add_argument("--output_dir", type=str, default="data/chroma/social_debate", 
                       help="Chroma 索引輸出目錄")
    parser.add_argument("--simple_output", type=str, default="src/rag/data/rag/simple_index.json",
                       help="簡單索引輸出路徑")
    parser.add_argument("--max_docs", type=int, default=None, help="最大文檔數（用於測試）")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 RAG 索引構建程式")
    print("=" * 50)
    
    # 載入配置
    config = ConfigLoader.load("rag")
    
    # 合併配置和命令行參數
    chroma_config = config.get("chroma", {})
    indexing_config = config.get("indexing", {})
    embedding_config = chroma_config.get("embedding", {})
    
    data_path = args.data_path or indexing_config.get("data_source", "data/raw/pairs.jsonl")
    batch_size = args.batch_size or embedding_config.get("batch_size", 500)
    
    print(f"\n配置參數:")
    print(f"  - 索引類型: {args.type}")
    print(f"  - 數據路徑: {data_path}")
    print(f"  - 批次大小: {batch_size}")
    if args.max_docs:
        print(f"  - 最大文檔數: {args.max_docs}")
    print("-" * 50)
    
    try:
        # 構建 Chroma 索引
        if args.type in ["chroma", "both"]:
            print("\n📚 構建 Chroma 向量索引...")
            
            # 確保輸出目錄存在
            output_dir = Path(args.output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # 執行構建
            stats = build_chroma_index(
                data_path=data_path,
                output_dir=str(output_dir),
                max_docs=args.max_docs,
                batch_size=batch_size
            )
            
            print(f"\n✅ Chroma 索引構建完成！")
            print(f"  - 總文檔數: {stats.get('total_docs', 0)}")
            print(f"  - 索引位置: {output_dir}")
        
        # 構建簡單索引
        if args.type in ["simple", "both"]:
            print("\n📄 構建簡單 JSON 索引...")
            
            # 確保輸出目錄存在
            simple_output = Path(args.simple_output)
            simple_output.parent.mkdir(parents=True, exist_ok=True)
            
            # 執行構建
            docs = build_simple_index(
                data_path=data_path,
                output_path=str(simple_output),
                max_docs=args.max_docs  # 不設定預設值，處理所有數據
            )
            
            print(f"\n✅ 簡單索引構建完成！")
            print(f"  - 文檔數: {len(docs)}")
            print(f"  - 索引位置: {simple_output}")
        
        print("\n🎉 所有索引構建完成！")
        
    except Exception as e:
        print(f"\n❌ 構建失敗: {e}")
        raise

def build_demo_index():
    """構建演示用的簡單索引"""
    demo_docs = [
        {
            "id": "doc_001",
            "content": "人工智慧的監管是一個複雜的議題。支持者認為，適當的監管可以防止 AI 被濫用，保護公民隱私和安全。例如，歐盟的 AI 法案就是一個嘗試建立全面監管框架的例子。",
            "metadata": {
                "type": "expert_opinion",
                "topic": "AI監管",
                "stance": "支持",
                "quality_score": 0.85
            }
        },
        {
            "id": "doc_002",
            "content": "反對政府監管 AI 的論點主要集中在創新和競爭力方面。過度監管可能會扼殺創新，使企業難以快速迭代和改進技術。矽谷的許多科技公司都擔心嚴格的監管會降低他們在全球市場的競爭力。",
            "metadata": {
                "type": "industry_perspective",
                "topic": "AI監管",
                "stance": "反對",
                "quality_score": 0.82
            }
        },
        {
            "id": "doc_003",
            "content": "根據麻省理工學院的研究，平衡的 AI 監管方法可能是最佳選擇。這種方法既保護公眾利益，又不會過度限制技術發展。研究建議採用風險導向的監管框架，對高風險應用實施更嚴格的控制。",
            "metadata": {
                "type": "research",
                "topic": "AI監管",
                "stance": "中立",
                "quality_score": 0.90
            }
        }
    ]
    
    # 保存演示索引
    demo_path = Path("src/rag/data/rag/simple_index.json")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(demo_path, 'w', encoding='utf-8') as f:
        json.dump({
            "documents": demo_docs,
            "metadata": {
                "version": "1.0",
                "created_at": "2024-01-01",
                "total_documents": len(demo_docs)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 演示索引已保存到: {demo_path}")

if __name__ == "__main__":
    # 如果沒有原始數據，先建立演示索引
    if not Path("data/raw/pairs.jsonl").exists():
        print("⚠️ 找不到原始數據，建立演示索引...")
        build_demo_index()
    else:
        main() 