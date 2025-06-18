"""
基礎智能體類別
定義所有辯論智能體的共同接口
"""

from typing import Dict, Any

class BaseAgent:
    """基礎智能體類別"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化智能體
        
        Args:
            name: 智能體名稱
            config: 配置字典
        """
        self.name = name
        self.config = config
        self.history = []

    def select_action(self, state: Dict[str, Any]) -> str:
        """
        選擇行動（需由子類實現）
        
        Args:
            state: 當前狀態
            
        Returns:
            行動字串
            
        Raises:
            NotImplementedError: 子類必須實現此方法
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_history(self, message: str):
        """
        更新對話歷史
        
        Args:
            message: 新的訊息
        """
        self.history.append(message)