from .base_agent import BaseAgent
try:
    from ..orchestrator.orchestrator import EnhancedOrchestrator as Orchestrator
except ImportError:
    from orchestrator.orchestrator import EnhancedOrchestrator as Orchestrator

class AgentA(BaseAgent):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.orch = Orchestrator()
    
    def select_action(self, state):
        return self.orch.get_reply(state, self.name)