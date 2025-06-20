"""
PPO 訓練主程式
使用真正的強化學習訓練辯論策略
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rl.ppo_trainer import PPOTrainer, DebateEnvironment
from src.utils.config_loader import ConfigLoader
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def plot_training_curve(rewards, save_path):
    """繪製訓練曲線"""
    plt.figure(figsize=(10, 6))
    
    # 計算移動平均
    window = 100
    if len(rewards) >= window:
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), avg_rewards, label=f'{window}-episode average')
    
    plt.plot(rewards, alpha=0.3, label='Episode reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"訓練曲線保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="訓練 PPO 辯論策略")
    parser.add_argument("--episodes", type=int, default=1000, help="訓練回合數")
    parser.add_argument("--lr", type=float, default=3e-4, help="學習率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO epsilon clipping")
    parser.add_argument("--k_epochs", type=int, default=4, help="PPO 更新次數")
    parser.add_argument("--output_dir", type=str, default="data/models/ppo", help="輸出目錄")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 PPO 辯論策略訓練")
    print("=" * 60)
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入配置
    config = ConfigLoader.load("rl")
    
    # 初始化 tokenizer 和 encoder（用於環境）
    print("載入語言模型...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    encoder = AutoModel.from_pretrained('distilbert-base-uncased')
    encoder.eval()
    
    # 創建環境
    env = DebateEnvironment(tokenizer, encoder)
    
    # 創建 PPO 訓練器
    trainer = PPOTrainer(
        state_dim=768,
        hidden_dim=256,
        num_actions=4,
        learning_rate=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs
    )
    
    print(f"\n訓練參數:")
    print(f"  訓練回合: {args.episodes}")
    print(f"  學習率: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Epsilon clip: {args.eps_clip}")
    print(f"  K epochs: {args.k_epochs}")
    print("-" * 60)
    
    # 訓練
    print("\n開始訓練...")
    episode_rewards = trainer.train(env, num_episodes=args.episodes)
    
    # 保存模型
    model_path = output_dir / "ppo_debate_policy.pt"
    trainer.save(str(model_path))
    
    # 保存訓練記錄
    training_log = {
        'episodes': args.episodes,
        'final_avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
        'max_reward': max(episode_rewards),
        'min_reward': min(episode_rewards),
        'hyperparameters': {
            'learning_rate': args.lr,
            'gamma': args.gamma,
            'eps_clip': args.eps_clip,
            'k_epochs': args.k_epochs
        }
    }
    
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # 繪製訓練曲線
    plot_path = output_dir / "training_curve.png"
    plot_training_curve(episode_rewards, plot_path)
    
    print(f"\n✅ 訓練完成！")
    print(f"  最終平均獎勵: {training_log['final_avg_reward']:.3f}")
    print(f"  最高獎勵: {training_log['max_reward']:.3f}")
    print(f"  最低獎勵: {training_log['min_reward']:.3f}")

if __name__ == "__main__":
    main() 