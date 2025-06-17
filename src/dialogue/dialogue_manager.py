try:
    from ..agents.agent_a import AgentA
    from ..agents.agent_b import AgentB
    from ..agents.agent_c import AgentC
    from ..utils.config_loader import Config
except ImportError:
    from agents.agent_a import AgentA
    from agents.agent_b import AgentB
    from agents.agent_c import AgentC
    from utils.config_loader import Config
import textwrap

class DialogueManager:
    def __init__(self):
        dfg = Config.load()
        self.rounds = dfg.get('debate', {}).get('rounds', 3)
        self.topic = dfg.get('debate', {}).get('topic', 'Should we adopt universal basic income?')
        self.agents = {
            'A': AgentA(name='A', config=dfg),
            'B': AgentB(name='B', config=dfg),
            'C': AgentC(name='C', config=dfg),
        }
        self.state = {
            'last_message': self.topic,
            'topic': self.topic,
            'history': []
        }


    def run(self):
        bar = "─" * 60
        print(f"\n{bar}\n🗣️  Debate Topic: {self.topic}\n{bar}")

        for rnd in range(1, self.rounds + 1):
            print(f"\n⏰ Round {rnd}\n{bar}")
            for agent in self.agents.values():
                reply = agent.select_action(self.state)

                # ⬇︎ 文字過長自動換行，最多 80 字一行
                wrapped = textwrap.fill(reply, width=80, subsequent_indent=" " * 9)
                print(f"{agent.name:>6} │ {wrapped}")

                # 更新所有 Agent 的歷史
                for ag in self.agents.values():
                    ag.update_history(reply)

                # 更新狀態
                self.state["last_message"] = reply
                self.state["history"].append({"speaker": agent.name, "text": reply})

        print(f"\n{bar}\n🏁 Debate Ended\n{bar}")