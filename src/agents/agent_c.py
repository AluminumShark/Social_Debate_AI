"""
Agent C - 中立分析型智能體
保持中立立場，提供平衡的分析觀點
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent
from orchestrator.parallel_orchestrator import ParallelOrchestrator as Orchestrator
from gpt_interface.gpt_client import chat

class AgentC(BaseAgent):
    """中立分析型智能體"""
    
    def __init__(self, name: str, config: dict):
        """
        初始化 Agent C
        
        Args:
            name: 智能體名稱
            config: 配置字典
        """
        super().__init__(name, config)
        self.orch = Orchestrator()
    
    def select_action(self, state):
        """
        選擇行動策略
        
        Args:
            state: 當前辯論狀態
            
        Returns:
            生成的回應文本
        """
        return self.orch.get_reply(state, self.name)