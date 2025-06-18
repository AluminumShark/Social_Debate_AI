from .base_agent import BaseAgent
from typing import List, Dict, Optional
from dataclasses import dataclass
from orchestrator.parallel_orchestrator import ParallelOrchestrator as Orchestrator
from gpt_interface.gpt_client import chat

class AgentB(BaseAgent):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.orch = Orchestrator()
    
    def select_action(self, state):
        return self.orch.get_reply(state, self.name)