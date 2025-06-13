from typing import Dict, Any

class BaseAgent:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.history = []

    def select_action(self, state: Dict[str, Any]) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_history(self, message: str):
        self.history.append(message)