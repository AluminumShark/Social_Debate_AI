"""
平行處理辯論協調器
支援 RL + GNN + RAG 平行運行，動態說服/反駁機制
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import time
from pathlib import Path

try:
    from ..rl.policy_network import select_strategy, choose_snippet, PolicyNetwork
    from ..gnn.social_encoder import social_vec
    from ..rag.retriever import create_enhanced_retriever
    from ..gpt_interface.gpt_client import chat
except ImportError:
    # 回退導入
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from rl.policy_network import select_strategy, choose_snippet, PolicyNetwork
    from gnn.social_encoder import social_vec
    from rag.retriever import create_enhanced_retriever
    from gpt_interface.gpt_client import chat

@dataclass
class AgentState:
    """Agent 狀態"""
    agent_id: str
    current_stance: float  # -1.0 到 1.0，立場強度
    conviction: float      # 0.0 到 1.0，信念堅定度
    social_context: List[float]  # 社會背景向量
    persuasion_history: List[float]  # 被說服歷史
    attack_history: List[float]     # 攻擊歷史
    
    def update_stance(self, persuasion_score: float, attack_score: float):
        """更新立場和信念"""
        # 計算說服效果
        persuasion_effect = persuasion_score * (1.0 - self.conviction)
        
        # 計算攻擊抵抗
        attack_resistance = self.conviction * 0.8
        attack_effect = max(0, attack_score - attack_resistance)
        
        # 更新立場 (說服使立場趨向中性，攻擊使立場極化)
        if persuasion_score > 0.6:  # 被說服
            self.current_stance *= (1.0 - persuasion_effect * 0.3)
            self.conviction *= 0.9  # 信念減弱
        
        if attack_effect > 0.3:  # 被攻擊
            self.current_stance *= (1.0 + attack_effect * 0.2)  # 立場更極端
            self.conviction = min(1.0, self.conviction * 1.1)  # 信念增強
        
        # 記錄歷史
        self.persuasion_history.append(persuasion_score)
        self.attack_history.append(attack_score)
        
        # 保持歷史長度
        if len(self.persuasion_history) > 10:
            self.persuasion_history.pop(0)
        if len(self.attack_history) > 10:
            self.attack_history.pop(0)

@dataclass
class DebateRound:
    """辯論回合"""
    round_number: int
    topic: str
    agent_states: Dict[str, AgentState]
    history: List[Dict]
    
class ParallelOrchestrator:
    """平行處理辯論協調器"""
    
    def __init__(self):
        self.policy_network = PolicyNetwork()
        self.retriever = create_enhanced_retriever()
        self.agent_states = {}
        self.debate_history = []
        
        # 執行器池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        print("✅ 平行辯論協調器初始化完成")
    
    def initialize_agents(self, agent_configs: List[Dict]) -> Dict[str, AgentState]:
        """初始化 Agent 狀態"""
        agents = {}
        
        for config in agent_configs:
            agent_id = config['id']
            agents[agent_id] = AgentState(
                agent_id=agent_id,
                current_stance=config.get('initial_stance', 0.0),
                conviction=config.get('initial_conviction', 0.7),
                social_context=config.get('social_context', [0.0] * 128),
                persuasion_history=[],
                attack_history=[]
            )
        
        self.agent_states = agents
        print(f"✅ 初始化 {len(agents)} 個 Agent")
        return agents
    
    async def parallel_analysis(self, agent_id: str, topic: str, 
                              history: List[str]) -> Dict:
        """平行執行 RL + GNN + RAG 分析"""
        
        # 構建查詢上下文
        recent_turns = history[-3:] if history else []
        context = f"Topic: {topic}\nRecent: {' '.join(recent_turns)}"
        agent_state = self.agent_states[agent_id]
        
        # 創建異步任務
        loop = asyncio.get_event_loop()
        
        # 1. RL 策略選擇
        rl_task = loop.run_in_executor(
            self.executor,
            self._rl_analysis,
            context, agent_state.social_context
        )
        
        # 2. GNN 社會分析
        gnn_task = loop.run_in_executor(
            self.executor,
            self._gnn_analysis,
            agent_id, agent_state
        )
        
        # 3. RAG 證據檢索
        rag_task = loop.run_in_executor(
            self.executor,
            self._rag_analysis,
            context, topic
        )
        
        # 等待所有任務完成
        try:
            rl_result, gnn_result, rag_result = await asyncio.gather(
                rl_task, gnn_task, rag_task
            )
            
            return {
                'rl': rl_result,
                'gnn': gnn_result,
                'rag': rag_result,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"❌ 平行分析失敗: {e}")
            return self._fallback_analysis(context, agent_id)
    
    def _rl_analysis(self, context: str, social_context: List[float]) -> Dict:
        """RL 策略分析"""
        try:
            strategy = select_strategy(context, "", social_context)
            
            # 預測品質分數
            quality_score = self.policy_network.predict_quality(context)
            
            return {
                'strategy': strategy,
                'quality_score': quality_score,
                'confidence': 0.8  # 可以從模型獲取
            }
        except Exception as e:
            print(f"⚠️ RL 分析失敗: {e}")
            return {'strategy': 'analytical', 'quality_score': 0.5, 'confidence': 0.3}
    
    def _gnn_analysis(self, agent_id: str, agent_state: AgentState) -> Dict:
        """GNN 社會關係分析"""
        try:
            # 獲取社會向量
            social_vector = social_vec(agent_id)
            
            # 分析社會影響力
            influence_score = sum(social_vector[:10]) / 10  # 簡化計算
            
            # 分析立場變化趨勢
            stance_trend = 0.0
            if len(agent_state.persuasion_history) >= 2:
                recent_persuasion = sum(agent_state.persuasion_history[-3:]) / 3
                stance_trend = recent_persuasion - 0.5
            
            return {
                'social_vector': social_vector,
                'influence_score': influence_score,
                'stance_trend': stance_trend,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction
            }
        except Exception as e:
            print(f"⚠️ GNN 分析失敗: {e}")
            return {
                'social_vector': [0.0] * 128,
                'influence_score': 0.5,
                'stance_trend': 0.0,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction
            }
    
    def _rag_analysis(self, context: str, topic: str) -> Dict:
        """RAG 證據檢索分析"""
        try:
            # 檢索證據池
            evidence_pool = self.retriever.retrieve(
                query=context,
                k=8,
                index_type='high_quality'
            )
            
            # 選擇最佳證據
            best_evidence = choose_snippet(context, evidence_pool)
            
            # 分析證據類型分布
            evidence_types = {}
            for ev in evidence_pool:
                ev_type = ev.get('type', 'unknown')
                evidence_types[ev_type] = evidence_types.get(ev_type, 0) + 1
            
            return {
                'evidence_pool': evidence_pool,
                'best_evidence': best_evidence,
                'evidence_types': evidence_types,
                'total_evidence': len(evidence_pool)
            }
        except Exception as e:
            print(f"⚠️ RAG 分析失敗: {e}")
            return {
                'evidence_pool': [],
                'best_evidence': "No evidence available",
                'evidence_types': {},
                'total_evidence': 0
            }
    
    def _fallback_analysis(self, context: str, agent_id: str) -> Dict:
        """回退分析"""
        return {
            'rl': {'strategy': 'analytical', 'quality_score': 0.5, 'confidence': 0.3},
            'gnn': {'social_vector': [0.0] * 128, 'influence_score': 0.5, 
                   'stance_trend': 0.0, 'current_stance': 0.0, 'conviction': 0.7},
            'rag': {'evidence_pool': [], 'best_evidence': "No evidence available",
                   'evidence_types': {}, 'total_evidence': 0},
            'timestamp': time.time()
        }
    
    def fuse_analysis_results(self, analysis_results: Dict, agent_id: str) -> Dict:
        """融合分析結果"""
        rl_result = analysis_results['rl']
        gnn_result = analysis_results['gnn']
        rag_result = analysis_results['rag']
        
        # 策略調整：根據社會影響力和立場調整策略
        base_strategy = rl_result['strategy']
        influence_score = gnn_result['influence_score']
        current_stance = gnn_result['current_stance']
        
        # 高影響力 + 強立場 = 更積極
        if influence_score > 0.6 and abs(current_stance) > 0.5:
            if base_strategy == 'analytical':
                adjusted_strategy = 'aggressive'
            else:
                adjusted_strategy = base_strategy
        # 低影響力 + 弱立場 = 更謹慎
        elif influence_score < 0.4 and abs(current_stance) < 0.3:
            if base_strategy == 'aggressive':
                adjusted_strategy = 'defensive'
            else:
                adjusted_strategy = base_strategy
        else:
            adjusted_strategy = base_strategy
        
        # 證據選擇：根據策略調整證據選擇
        evidence = rag_result['best_evidence']
        evidence_confidence = min(1.0, rag_result['total_evidence'] / 5.0)
        
        return {
            'final_strategy': adjusted_strategy,
            'evidence': evidence,
            'evidence_confidence': evidence_confidence,
            'social_influence': influence_score,
            'stance_strength': abs(current_stance),
            'conviction': gnn_result['conviction'],
            'fusion_timestamp': time.time()
        }
    
    async def generate_response(self, agent_id: str, topic: str, 
                              history: List[str], target_agents: List[str]) -> str:
        """生成辯論回覆"""
        
        # 1. 平行分析
        print(f"🔄 Agent {agent_id} 開始平行分析...")
        analysis_start = time.time()
        
        analysis_results = await self.parallel_analysis(agent_id, topic, history)
        
        analysis_time = time.time() - analysis_start
        print(f"⚡ 平行分析完成 ({analysis_time:.2f}s)")
        
        # 2. 融合結果
        fused_results = self.fuse_analysis_results(analysis_results, agent_id)
        
        # 3. 構建提示
        agent_state = self.agent_states[agent_id]
        recent_history = history[-4:] if history else []
        
        # 分析目標 Agent 的弱點
        target_analysis = self._analyze_targets(target_agents, history)
        
        prompt = self._build_enhanced_prompt(
            agent_id, topic, recent_history, fused_results, target_analysis
        )
        
        # 4. 生成回覆
        print(f"🤖 Agent {agent_id} 使用 {fused_results['final_strategy']} 策略生成回覆...")
        response = chat(prompt)
        
        # 5. 評估回覆效果
        response_effects = self._evaluate_response(response, target_agents)
        
        return response
    
    def _analyze_targets(self, target_agents: List[str], history: List[str]) -> Dict:
        """分析目標 Agent 的弱點和機會"""
        target_analysis = {}
        
        for target_id in target_agents:
            if target_id in self.agent_states:
                target_state = self.agent_states[target_id]
                
                # 分析說服機會
                persuasion_opportunity = 1.0 - target_state.conviction
                
                # 分析攻擊機會
                attack_opportunity = abs(target_state.current_stance)
                
                # 分析歷史趨勢
                recent_persuasion = 0.0
                if target_state.persuasion_history:
                    recent_persuasion = sum(target_state.persuasion_history[-2:]) / 2
                
                target_analysis[target_id] = {
                    'persuasion_opportunity': persuasion_opportunity,
                    'attack_opportunity': attack_opportunity,
                    'recent_persuasion': recent_persuasion,
                    'stance': target_state.current_stance,
                    'conviction': target_state.conviction
                }
        
        return target_analysis
    
    def _build_enhanced_prompt(self, agent_id: str, topic: str, history: List[str],
                              fused_results: Dict, target_analysis: Dict) -> str:
        """構建增強提示"""
        
        agent_state = self.agent_states[agent_id]
        strategy = fused_results['final_strategy']
        evidence = fused_results['evidence']
        
        # 歷史對話
        history_text = '\n'.join(f"Turn {i+1}: {turn}" for i, turn in enumerate(history))
        
        # 目標分析
        target_info = ""
        for target_id, analysis in target_analysis.items():
            target_info += f"\n- {target_id}: 立場{analysis['stance']:.2f}, 信念{analysis['conviction']:.2f}, 說服機會{analysis['persuasion_opportunity']:.2f}"
        
        # 策略指導
        strategy_guidance = {
            'aggressive': "採用強勢攻擊，指出對方論點的漏洞和矛盾",
            'defensive': "防禦性回應，保護自己的立場並反駁攻擊",
            'analytical': "理性分析，使用邏輯和證據進行論證",
            'empathetic': "同理心說服，尋找共同點並溫和地改變對方觀點"
        }
        
        return f"""你是辯論 Agent {agent_id}，參與關於 "{topic}" 的辯論。

當前狀態：
- 你的立場強度：{agent_state.current_stance:.2f} (-1到1)
- 你的信念堅定度：{agent_state.conviction:.2f} (0到1)
- 選定策略：{strategy}

對話歷史：
{history_text}

證據支持：
{evidence}

目標對手分析：{target_info}

策略指導：{strategy_guidance.get(strategy, '')}

請生成你的下一輪發言（≤150字）：
1. 根據 {strategy} 策略行動
2. 利用提供的證據支持你的論點
3. 針對對手的弱點進行攻擊或說服
4. 引用證據時使用 [CITE] 標記
5. 保持你的立場但允許適度調整

發言："""
    
    def _evaluate_response(self, response: str, target_agents: List[str]) -> Dict:
        """評估回覆的說服力和攻擊性"""
        
        # 簡單的啟發式評估
        persuasion_indicators = ['however', 'consider', 'understand', 'perspective', 'common']
        attack_indicators = ['wrong', 'flawed', 'mistake', 'ignore', 'fail']
        evidence_indicators = ['[CITE]', 'study', 'research', 'data', 'evidence']
        
        persuasion_score = sum(1 for indicator in persuasion_indicators if indicator in response.lower()) / len(persuasion_indicators)
        attack_score = sum(1 for indicator in attack_indicators if indicator in response.lower()) / len(attack_indicators)
        evidence_score = sum(1 for indicator in evidence_indicators if indicator in response.lower()) / len(evidence_indicators)
        
        return {
            'persuasion_score': min(1.0, persuasion_score),
            'attack_score': min(1.0, attack_score),
            'evidence_score': min(1.0, evidence_score),
            'length_score': min(1.0, len(response.split()) / 100)
        }
    
    async def run_debate_round(self, round_number: int, topic: str, 
                             agent_order: List[str]) -> DebateRound:
        """執行一輪辯論"""
        
        print(f"\n🎭 開始第 {round_number} 輪辯論")
        print(f"主題: {topic}")
        print(f"發言順序: {' → '.join(agent_order)}")
        
        round_responses = []
        
        for i, agent_id in enumerate(agent_order):
            print(f"\n--- Agent {agent_id} 發言 ---")
            
            # 確定目標對手
            other_agents = [aid for aid in agent_order if aid != agent_id]
            
            # 生成回覆
            response = await self.generate_response(
                agent_id, topic, 
                [r['content'] for r in round_responses], 
                other_agents
            )
            
            # 評估效果
            effects = self._evaluate_response(response, other_agents)
            
            # 記錄回覆
            round_responses.append({
                'agent_id': agent_id,
                'content': response,
                'effects': effects,
                'timestamp': time.time()
            })
            
            print(f"Agent {agent_id}: {response[:100]}...")
            print(f"效果評估: 說服{effects['persuasion_score']:.2f}, 攻擊{effects['attack_score']:.2f}")
            
            # 更新其他 Agent 的狀態
            for target_id in other_agents:
                if target_id in self.agent_states:
                    self.agent_states[target_id].update_stance(
                        effects['persuasion_score'],
                        effects['attack_score']
                    )
        
        # 創建回合記錄
        debate_round = DebateRound(
            round_number=round_number,
            topic=topic,
            agent_states=dict(self.agent_states),
            history=round_responses
        )
        
        self.debate_history.append(debate_round)
        
        # 顯示回合結果
        self._display_round_summary(debate_round)
        
        return debate_round
    
    def _display_round_summary(self, debate_round: DebateRound):
        """顯示回合總結"""
        print(f"\n📊 第 {debate_round.round_number} 輪總結")
        print("=" * 50)
        
        for agent_id, state in debate_round.agent_states.items():
            print(f"Agent {agent_id}:")
            print(f"  立場: {state.current_stance:+.2f} | 信念: {state.conviction:.2f}")
            if state.persuasion_history:
                avg_persuasion = sum(state.persuasion_history[-3:]) / min(3, len(state.persuasion_history))
                print(f"  近期被說服度: {avg_persuasion:.2f}")
    
    def get_debate_summary(self) -> Dict:
        """獲取辯論總結"""
        if not self.debate_history:
            return {"message": "尚未開始辯論"}
        
        # 分析立場變化
        stance_changes = {}
        for agent_id in self.agent_states:
            initial_stance = 0.0  # 假設初始立場為中性
            current_stance = self.agent_states[agent_id].current_stance
            stance_changes[agent_id] = current_stance - initial_stance
        
        # 找出最有說服力的 Agent
        most_persuasive = max(stance_changes.keys(), 
                            key=lambda x: abs(stance_changes[x]))
        
        return {
            "total_rounds": len(self.debate_history),
            "stance_changes": stance_changes,
            "most_persuasive_agent": most_persuasive,
            "final_states": {aid: {
                "stance": state.current_stance,
                "conviction": state.conviction
            } for aid, state in self.agent_states.items()}
        }

# 便利函數
def create_parallel_orchestrator():
    """創建平行協調器"""
    return ParallelOrchestrator() 