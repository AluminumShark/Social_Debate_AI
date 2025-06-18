"""
Social Debate AI 統一訓練管理腳本
支援訓練所有模組：GNN、RL、RAG
"""

import sys
import argparse
from pathlib import Path

# 添加 src 到路徑
sys.path.append(str(Path(__file__).parent / "src"))

def train_gnn():
    """訓練 GNN 模型"""
    print("\n" + "="*50)
    print("🌐 開始訓練 GNN 模型...")
    print("="*50)
    
    from src.gnn.train import main as gnn_main
    sys.argv = ['train.py']  # 重置參數
    gnn_main()

def train_rl():
    """訓練 RL 模型"""
    print("\n" + "="*50)
    print("🤖 開始訓練 RL 模型...")
    print("="*50)
    
    from src.rl.train import main as rl_main
    sys.argv = ['train.py', '--process_data']  # 自動處理數據
    rl_main()

def build_rag(index_type='both'):
    """構建 RAG 索引"""
    print("\n" + "="*50)
    print("📚 開始構建 RAG 索引...")
    print("="*50)
    
    from src.rag.train import main as rag_main
    sys.argv = ['train.py', '--type', index_type]  # 支援 simple, chroma, both
    rag_main()

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="Social Debate AI 統一訓練管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python train_all.py --all              # 訓練所有模組
  python train_all.py --gnn              # 只訓練 GNN
  python train_all.py --rl               # 只訓練 RL
  python train_all.py --rag              # 只構建 RAG 索引（簡單索引）
  python train_all.py --rag-chroma       # 構建 Chroma 向量索引
  python train_all.py --rag-both         # 構建兩種索引
  python train_all.py --gnn --rl         # 訓練 GNN 和 RL
        """
    )
    
    parser.add_argument("--all", action="store_true", help="訓練所有模組")
    parser.add_argument("--gnn", action="store_true", help="訓練 GNN 模型")
    parser.add_argument("--rl", action="store_true", help="訓練 RL 模型")
    parser.add_argument("--rag", action="store_true", help="構建 RAG 索引（簡單索引）")
    parser.add_argument("--rag-chroma", action="store_true", help="構建 Chroma 向量索引")
    parser.add_argument("--rag-both", action="store_true", help="構建兩種索引")
    
    args = parser.parse_args()
    
    # 如果沒有指定任何選項，顯示幫助
    if not any([args.all, args.gnn, args.rl, args.rag, args.rag_chroma, args.rag_both]):
        parser.print_help()
        return
    
    print("🚀 Social Debate AI 訓練管理系統")
    print("=" * 50)
    
    try:
        # 訓練 GNN
        if args.all or args.gnn:
            train_gnn()
        
        # 訓練 RL
        if args.all or args.rl:
            train_rl()
        
        # 構建 RAG
        if args.all or args.rag:
            build_rag('simple')
        elif args.rag_chroma:
            build_rag('chroma')
        elif args.rag_both:
            build_rag('both')
        
        print("\n" + "="*50)
        print("✅ 所有訓練任務完成！")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 