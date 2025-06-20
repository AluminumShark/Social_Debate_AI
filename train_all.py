"""
Social Debate AI 統一訓練管理腳本
支援訓練所有模組：GNN、RL、RAG
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time

# 添加 src 到路徑
sys.path.append(str(Path(__file__).parent / "src"))

def run_command(cmd, description):
    """執行命令並顯示進度"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"執行命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} 完成！(耗時: {elapsed:.1f}秒)")
    else:
        print(f"\n❌ {description} 失敗！")
        sys.exit(1)
    
    return result

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
    print("訓練 RL 策略網路（PPO）...")
    from src.rl.train_ppo import main as train_ppo_main
    
    # 創建參數列表
    import sys
    original_argv = sys.argv
    try:
        # 設置 PPO 訓練參數
        sys.argv = ['train_ppo.py', '--episodes', '1000', '--output_dir', 'data/models/ppo']
        train_ppo_main()
    finally:
        sys.argv = original_argv

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
    parser.add_argument("--skip-rag", action="store_true", help="跳過 RAG 訓練")
    parser.add_argument("--skip-rl", action="store_true", help="跳過 RL 訓練")
    parser.add_argument("--skip-gnn", action="store_true", help="跳過 GNN 訓練")
    parser.add_argument("--quick", action="store_true", help="快速訓練模式（減少 epochs）")
    
    args = parser.parse_args()
    
    # 如果沒有指定任何選項，顯示幫助
    if not any([args.all, args.gnn, args.rl, args.rag, args.rag_chroma, args.rag_both]):
        parser.print_help()
        return
    
    print("🎯 Social Debate AI 統一訓練腳本")
    print(f"配置: RAG={'跳過' if args.skip_rag else '訓練'}, "
          f"RL={'跳過' if args.skip_rl else '訓練'}, "
          f"GNN={'跳過' if args.skip_gnn else '訓練'}")
    
    print("🚀 Social Debate AI 訓練管理系統")
    print("=" * 50)
    
    try:
        # 訓練 GNN
        if args.all or args.gnn:
            if args.skip_gnn:
                print("🚫 跳過 GNN 訓練")
            else:
                # 訓練監督式模型
                gnn_supervised_cmd = [sys.executable, "-m", "src.gnn.train_supervised"]
                if args.quick:
                    gnn_supervised_cmd = [sys.executable, "-c", 
                        "from src.gnn.train_supervised import train_supervised_gnn; "
                        "train_supervised_gnn(epochs=10)"]
                else:
                    gnn_supervised_cmd = [sys.executable, "-c", 
                        "from src.gnn.train_supervised import train_supervised_gnn; "
                        "train_supervised_gnn(epochs=50)"]
                run_command(gnn_supervised_cmd, "訓練 GNN 說服力預測（監督式）")
        
        # 訓練 RL
        if args.all or args.rl:
            if args.skip_rl:
                print("🚫 跳過 RL 訓練")
            else:
                if args.quick:
                    # 快速模式：較少的訓練回合
                    rl_cmd = [sys.executable, "-m", "src.rl.train_ppo", "--episodes", "100"]
                else:
                    rl_cmd = [sys.executable, "-m", "src.rl.train_ppo", "--episodes", "1000"]
                run_command(rl_cmd, "訓練 RL 策略網路（PPO）")
        
        # 構建 RAG
        if args.all or args.rag or args.rag_chroma or args.rag_both:
            if args.skip_rag:
                print("🚫 跳過 RAG 訓練")
            else:
                if args.rag_chroma:
                    build_rag('chroma')
                elif args.rag_both:
                    build_rag('both')
                else:
                    build_rag('simple')
        
        # 測試系統
        print(f"\n{'='*60}")
        print("🧪 執行系統整合測試...")
        print(f"{'='*60}")
        
        test_cmd = [sys.executable, "test_system_integrity.py"]
        result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("\n✅ 所有訓練完成！系統測試通過！")
            print("\n📊 訓練總結：")
            if not args.skip_rag:
                print("  - RAG: 檢索索引已建立")
            if not args.skip_rl:
                print("  - RL: 策略網路已訓練")
            if not args.skip_gnn:
                print("  - GNN: 監督式說服力預測模型已訓練")
            
            print("\n🚀 現在可以運行 UI:")
            print("  python run_flask.py")
        else:
            print("\n⚠️ 訓練完成但系統測試失敗")
            if result.stderr:
                print("錯誤輸出:")
                print(result.stderr)
        
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 