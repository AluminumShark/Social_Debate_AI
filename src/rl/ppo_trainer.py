"""
PPO (Proximal Policy Optimization) 訓練器
用於訓練辯論策略的強化學習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModel
from collections import deque
import random

@dataclass
class DebateTransition:
    """辯論轉移記錄"""
    state: torch.Tensor  # 當前狀態
    action: int  # 選擇的策略
    reward: float  # 獲得的獎勵
    next_state: torch.Tensor  # 下一個狀態
    done: bool  # 是否結束
    log_prob: float  # 動作的對數概率
    value: float  # 狀態價值

class DebatePPONetwork(nn.Module):
    """PPO 網路架構（Actor-Critic）"""
    
    def __init__(self, state_dim: int = 768, hidden_dim: int = 256, num_actions: int = 4):
        super().__init__()
        
        # 共享層
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor 頭（策略網路）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Critic 頭（價值網路）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向傳播"""
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """選擇動作"""
        action_logits, state_value = self.forward(state)
        
        # 創建動作分布
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()

class DebateEnvironment:
    """辯論環境"""
    
    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
        
        # 辯論狀態
        self.current_stance = 0.0  # -1 到 1
        self.opponent_stance = 0.0
        self.round = 0
        self.max_rounds = 10
        self.history = []
        
    def reset(self) -> torch.Tensor:
        """重置環境"""
        self.current_stance = np.random.uniform(-0.5, 0.5)
        self.opponent_stance = np.random.uniform(-0.5, 0.5)
        self.round = 0
        self.history = []
        
        # 返回初始狀態
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """獲取當前狀態表示"""
        # 簡化的狀態表示（實際應該包含更多信息）
        state_features = torch.zeros(768)
        
        # 添加立場信息
        state_features[0] = self.current_stance
        state_features[1] = self.opponent_stance
        state_features[2] = self.round / self.max_rounds
        state_features[3] = len(self.history) / 20.0
        
        # 添加歷史策略信息
        if self.history:
            recent_strategies = self.history[-5:]
            for i, (strategy, _) in enumerate(recent_strategies):
                state_features[10 + i] = self.strategies.index(strategy) / 4.0
        
        return state_features
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """執行動作"""
        strategy = self.strategies[action]
        self.history.append((strategy, self.round))
        
        # 計算獎勵
        reward = self._calculate_reward(strategy)
        
        # 更新狀態
        self._update_stances(strategy)
        self.round += 1
        
        # 檢查是否結束
        done = (self.round >= self.max_rounds) or abs(self.opponent_stance) > 0.9
        
        # 獲取新狀態
        next_state = self._get_state()
        
        info = {
            'strategy': strategy,
            'current_stance': self.current_stance,
            'opponent_stance': self.opponent_stance,
            'round': self.round
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, strategy: str) -> float:
        """計算獎勵"""
        reward = 0.0
        
        # 基於策略和當前狀態的獎勵
        stance_diff = abs(self.current_stance - self.opponent_stance)
        
        if strategy == 'aggressive':
            # 攻擊策略：立場差異大時有效
            if stance_diff > 0.5:
                reward += 0.3
            else:
                reward -= 0.1
                
        elif strategy == 'defensive':
            # 防禦策略：保護自己的立場
            if abs(self.current_stance) > 0.7:
                reward += 0.2
            
        elif strategy == 'analytical':
            # 分析策略：中性時有效
            if stance_diff < 0.3:
                reward += 0.4
                
        elif strategy == 'empathetic':
            # 同理策略：對方立場弱時有效
            if abs(self.opponent_stance) < 0.3:
                reward += 0.5
        
        # 說服獎勵
        if abs(self.opponent_stance) < 0.1:  # 對方接近中立
            reward += 1.0
        
        # 避免重複使用同一策略
        if len(self.history) >= 2 and self.history[-2][0] == strategy:
            reward -= 0.2
        
        return reward
    
    def _update_stances(self, strategy: str):
        """更新立場"""
        # 簡化的立場更新邏輯
        if strategy == 'aggressive':
            self.opponent_stance *= 0.95
            self.current_stance *= 1.05
        elif strategy == 'defensive':
            self.current_stance *= 0.98
        elif strategy == 'analytical':
            self.opponent_stance *= 0.9
        elif strategy == 'empathetic':
            self.opponent_stance *= 0.85
            
        # 限制範圍
        self.current_stance = np.clip(self.current_stance, -1, 1)
        self.opponent_stance = np.clip(self.opponent_stance, -1, 1)

class PPOTrainer:
    """PPO 訓練器"""
    
    def __init__(self, 
                 state_dim: int = 768,
                 hidden_dim: int = 256,
                 num_actions: int = 4,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化網路
        self.policy = DebatePPONetwork(state_dim, hidden_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # PPO 參數
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 記憶體
        self.memory = []
        
        print(f"PPO 訓練器初始化完成")
        print(f"  設備: {self.device}")
        print(f"  學習率: {learning_rate}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon clip: {eps_clip}")
        
    def store_transition(self, transition: DebateTransition):
        """存儲轉移"""
        self.memory.append(transition)
    
    def compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """計算折扣回報"""
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        return returns
    
    def update(self):
        """PPO 更新"""
        if len(self.memory) == 0:
            return
        
        # 提取數據
        states = torch.stack([t.state for t in self.memory]).to(self.device)
        actions = torch.tensor([t.action for t in self.memory]).to(self.device)
        rewards = [t.reward for t in self.memory]
        dones = [t.done for t in self.memory]
        old_log_probs = torch.tensor([t.log_prob for t in self.memory]).to(self.device)
        old_values = torch.tensor([t.value for t in self.memory]).to(self.device)
        
        # 計算回報和優勢
        returns = torch.tensor(self.compute_returns(rewards, dones)).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新 K 次
        for _ in range(self.k_epochs):
            # 獲取新的動作概率和價值
            action_logits, state_values = self.policy(states)
            dist = Categorical(F.softmax(action_logits, dim=-1))
            new_log_probs = dist.log_prob(actions)
            
            # 計算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 計算 surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 計算損失
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # 總損失
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # 清空記憶體
        self.memory = []
    
    def train(self, env: DebateEnvironment, num_episodes: int = 1000):
        """訓練 PPO"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            while True:
                # 選擇動作
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(self.device)
                    action, log_prob, value = self.policy.get_action(state_tensor)
                
                # 執行動作
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # 存儲轉移
                transition = DebateTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                )
                self.store_transition(transition)
                
                state = next_state
                
                if done:
                    break
            
            # 更新策略
            if (episode + 1) % 10 == 0:
                self.update()
            
            episode_rewards.append(episode_reward)
            
            # 打印進度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, 平均獎勵: {avg_reward:.3f}")
        
        return episode_rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型保存到: {path}")
    
    def load(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型載入自: {path}") 