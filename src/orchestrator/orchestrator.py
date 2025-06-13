from gpt_interface.gpt_client import chat

class Orchestrator:
    def build_prompt(self, state, agent_name):
        return f"""Topic: {state['topic']}
history: {state['history'][-3:]}
You are {agent_name}, please give your opinion and response on the topic or other agents' opinion.
"""
    
    def get_reply(self, state, agent_name):
        prompt = self.build_prompt(state, agent_name)
        return chat(prompt)