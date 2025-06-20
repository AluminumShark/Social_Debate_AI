"""
平行辯論協調器
整合 RL、GNN、RAG 三個模組的平行處理
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import random
import numpy as np

# 延遲導入，避免循環依賴
# PolicyNetwork 和 Retriever 將在需要時動態導入

# 導入 GPT 接口
try:
    from gpt_interface.gpt_client import chat
except ImportError:
    # 定義虛擬函數，以防 GPT 不可用
    def chat(prompt: str) -> str:
        """虛擬 chat 函數"""
        # 這裡應該調用實際的 GPT 接口
        # 為了演示，返回一個模擬回應
        responses = [
            "基於深入分析，我認為這個議題需要從多個角度來考慮。首先，我們必須承認其複雜性，並理解不同立場背後的合理關切。",
            "讓我從另一個角度來闡述這個問題。雖然對方提出了一些觀點，但我認為他們忽略了幾個關鍵因素。",
            "我理解對方的擔憂，但我們需要基於事實和數據來討論。根據最新的研究顯示，這個議題的影響遠比表面看起來更深遠。"
        ]
        return random.choice(responses)

# 嘗試導入所需模組
try:
    from rl.policy_network import select_strategy as _select_strategy, choose_snippet as _choose_snippet, PolicyNetwork as _PolicyNetwork
    from gnn.social_encoder import social_vec as _social_vec, get_social_influence_score, predict_persuasion
    from rag.retriever import create_enhanced_retriever as _create_enhanced_retriever
    from utils.config_loader import ConfigLoader as _ConfigLoader
    
    # 創建包裝函數
    def select_strategy(query: str, context: str = "", social_context: List[float] = None) -> str:
        """包裝 select_strategy 函數"""
        return _select_strategy(query, context, social_context)
    
    def choose_snippet(state_text: str, pool: List[Dict]) -> str:
        """包裝 choose_snippet 函數"""
        return _choose_snippet(state_text, pool)
    
    def social_vec(agent_id: str) -> List[float]:
        """包裝 social_vec 函數"""
        return _social_vec(agent_id)
        
except ImportError as e:
    print(f"⚠️ 部分模組導入失敗: {e}")
    print("⚠️ 使用虛擬函數運行")
    
    # 定義虛擬函數
    def select_strategy(query: str, context: str = "", social_context: List[float] = None) -> str:
        """虛擬 select_strategy 函數"""
        strategies = ['analytical', 'aggressive', 'defensive', 'empathetic']
        return random.choice(strategies)
    
    def choose_snippet(state_text: str, pool: List[Dict]) -> str:
        """虛擬 choose_snippet 函數"""
        if pool:
            return pool[0].get('content', 'No evidence available')
        return "No evidence available"
    
    def social_vec(agent_id: str) -> List[float]:
        """虛擬 social_vec 函數"""
        return [random.random() for _ in range(128)]

@dataclass
class AgentState:
    """Agent 狀態"""
    agent_id: str
    current_stance: float  # -1.0 到 1.0，立場強度
    conviction: float      # 0.0 到 1.0，信念堅定度
    social_context: List[float]  # 社會背景向量
    persuasion_history: List[float]  # 被說服歷史
    attack_history: List[float]     # 攻擊歷史
    has_surrendered: bool = False  # 是否已投降
    
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
            self.conviction *= 0.85  # 信念減弱更多
        
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
        
        # 檢查是否應該投降（條件放寬）
        if len(self.persuasion_history) >= 2:
            recent_persuasion = sum(self.persuasion_history[-2:]) / 2
            # 條件1：高說服度 + 低信念
            if recent_persuasion > 0.6 and self.conviction < 0.4:
                self.has_surrendered = True
                print(f"💔 {self.agent_id} 被說服了，選擇投降！")
            # 條件2：立場已經接近中立
            elif abs(self.current_stance) < 0.2 and self.conviction < 0.5:
                self.has_surrendered = True
                print(f"💔 {self.agent_id} 立場動搖，選擇投降！")
            # 條件3：連續被高度說服
            elif len(self.persuasion_history) >= 3:
                consecutive_high = all(score > 0.5 for score in self.persuasion_history[-3:])
                if consecutive_high:
                    self.has_surrendered = True
                    print(f"💔 {self.agent_id} 連續被說服，選擇投降！")

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
        self.policy_network = None  # 延遲載入
        self.retriever = None  # 延遲載入
        self.agent_states = {}
        self.debate_history = []
        
        # 執行器池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        print("   ✅ 平行辯論協調器初始化完成")
        print("   ⚡ 執行器池: 4 個工作執行緒")
        print("   💾 模型延遲載入: 將在首次使用時載入")
    
    def _get_policy_network(self):
        """延遲載入 PolicyNetwork"""
        if self.policy_network is None:
            print("📦 首次使用，正在載入 RL 策略網路...")
            print("   🔍 載入 DistilBERT 模型...")
            start_time = time.time()
            from rl.policy_network import PolicyNetwork
            self.policy_network = PolicyNetwork()
            load_time = time.time() - start_time
            print(f"   ✅ RL 策略網路載入完成 ({load_time:.2f}s)")
            print(f"   📊 模型大小: ~66M 參數")
        return self.policy_network
    
    def _get_retriever(self):
        """延遲載入 Retriever"""
        if self.retriever is None:
            print("📦 首次使用，正在載入 RAG 檢索器...")
            start_time = time.time()
            try:
                print("   🔍 檢查 Chroma 向量資料庫...")
                from rag.retriever import create_enhanced_retriever
                self.retriever = create_enhanced_retriever()
                load_time = time.time() - start_time
                print(f"   ✅ RAG 檢索器載入完成 ({load_time:.2f}s)")
            except Exception as e:
                print(f"   ⚠️ 增強檢索器載入失敗: {e}")
                print("   🔄 使用簡單檢索器...")
                from rag.simple_retriever import SimpleRetriever
                self.retriever = SimpleRetriever()
                load_time = time.time() - start_time
                print(f"   ✅ 簡單檢索器載入完成 ({load_time:.2f}s)")
                # 獲取統計信息
                stats = self.retriever.get_stats()
                print(f"   📊 索引大小: {stats.get('total_documents', 0):,} 個文檔")
        return self.retriever
    
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
            print(f"  📊 RL: 選擇策略 = {strategy}")
            
            # 預測品質分數
            quality_score = self._get_policy_network().predict_quality(context)
            print(f"  📊 RL: 品質分數 = {quality_score:.2f}")
            
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
            print(f"  🌐 GNN: 社會向量維度 = {len(social_vector)}")
            
            # 使用新的影響力計算方法
            influence_score = get_social_influence_score(agent_id)
            print(f"  🌐 GNN: 影響力分數 = {influence_score:.2f}")
            
            # 如果有最近的發言，預測說服策略
            if hasattr(agent_state, 'last_response') and agent_state.last_response:
                # 簡化的文本特徵提取（實際應使用 BERT）
                text_features = np.random.randn(768)  # 臨時使用隨機特徵
                persuasion_pred = predict_persuasion(text_features, agent_id)
                
                print(f"  🌐 GNN: 預測 Delta 概率 = {persuasion_pred['delta_probability']:.2f}")
                print(f"  🌐 GNN: 建議策略 = {persuasion_pred['best_strategy']}")
            else:
                persuasion_pred = {
                    'delta_probability': 0.5,
                    'best_strategy': 'analytical',
                    'strategy_scores': {}
                }
            
            # 分析立場變化趨勢
            stance_trend = 0.0
            if len(agent_state.persuasion_history) >= 2:
                recent_persuasion = sum(agent_state.persuasion_history[-3:]) / 3
                stance_trend = recent_persuasion - 0.5
            
            print(f"  🌐 GNN: 立場趨勢 = {stance_trend:.2f}")
            
            return {
                'social_vector': social_vector,
                'influence_score': influence_score,
                'stance_trend': stance_trend,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction,
                'persuasion_prediction': persuasion_pred
            }
        except Exception as e:
            print(f"⚠️ GNN 分析失敗: {e}")
            return {
                'social_vector': [0.0] * 128,
                'influence_score': 0.5,
                'stance_trend': 0.0,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction,
                'persuasion_prediction': {
                    'delta_probability': 0.5,
                    'best_strategy': 'analytical',
                    'strategy_scores': {}
                }
            }
    
    def _rag_analysis(self, context: str, topic: str) -> Dict:
        """RAG 證據檢索分析"""
        try:
            # 檢索證據池
            retrieval_results = self._get_retriever().retrieve(
                query=context,
                top_k=8
            )
            print(f"  📚 RAG: 檢索到 {len(retrieval_results)} 個證據")
            
            # 轉換為字典格式供 choose_snippet 使用
            evidence_pool = []
            for result in retrieval_results:
                # 檢查 result 是否已經是字典（來自 SimpleRetrieverAdapter）
                if isinstance(result, dict):
                    evidence_dict = {
                        'content': result.get('content', ''),
                        'similarity_score': result.get('score', 0.0),
                        'metadata': result.get('metadata', {}),
                        'doc_id': result.get('doc_id', '')
                    }
                else:
                    # 如果是物件（來自 EnhancedRetriever）
                    evidence_dict = {
                        'content': getattr(result, 'content', ''),
                        'similarity_score': getattr(result, 'score', 0.0),
                        'metadata': getattr(result, 'metadata', {}),
                        'doc_id': getattr(result, 'doc_id', '')
                    }
                evidence_pool.append(evidence_dict)
            
            # 選擇最佳證據
            best_evidence = choose_snippet(context, evidence_pool)
            print(f"  📚 RAG: 最佳證據長度 = {len(best_evidence)} 字")
            
            # 分析證據類型分布
            evidence_types = {}
            for result in retrieval_results:
                # 安全地獲取 metadata
                if isinstance(result, dict):
                    metadata = result.get('metadata', {})
                else:
                    metadata = getattr(result, 'metadata', {})
                
                ev_type = metadata.get('type', 'unknown')
                evidence_types[ev_type] = evidence_types.get(ev_type, 0) + 1
            
            print(f"  📚 RAG: 證據類型分布 = {evidence_types}")
            
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
        
        # 策略調整：結合 RL 和 GNN 的建議
        base_strategy = rl_result['strategy']
        gnn_strategy = gnn_result['persuasion_prediction']['best_strategy']
        influence_score = gnn_result['influence_score']
        current_stance = gnn_result['current_stance']
        delta_probability = gnn_result['persuasion_prediction']['delta_probability']
        
        # 策略融合邏輯
        if delta_probability > 0.7:
            # 高說服成功率，優先使用 GNN 建議的策略
            adjusted_strategy = gnn_strategy
            print(f"  🔄 策略調整: {base_strategy} → {adjusted_strategy} (基於高 Delta 概率)")
        elif influence_score > 0.6 and abs(current_stance) > 0.5:
            # 高影響力 + 強立場 = 更積極
            if base_strategy == 'analytical' and gnn_strategy == 'aggressive':
                adjusted_strategy = 'aggressive'
                print(f"  🔄 策略調整: {base_strategy} → {adjusted_strategy} (高影響力+強立場)")
            else:
                adjusted_strategy = base_strategy
        elif influence_score < 0.4 and abs(current_stance) < 0.3:
            # 低影響力 + 弱立場 = 更謹慎
            if base_strategy == 'aggressive':
                adjusted_strategy = 'defensive'
                print(f"  🔄 策略調整: {base_strategy} → {adjusted_strategy} (低影響力+弱立場)")
            else:
                adjusted_strategy = base_strategy
        else:
            # 權衡 RL 和 GNN 的建議
            strategy_scores = gnn_result['persuasion_prediction']['strategy_scores']
            if strategy_scores.get(base_strategy, 0) < 0.2:
                # 如果 RL 選擇的策略在 GNN 中得分很低，考慮切換
                adjusted_strategy = gnn_strategy
                print(f"  🔄 策略調整: {base_strategy} → {adjusted_strategy} (GNN 建議)")
            else:
                adjusted_strategy = base_strategy
        
        # 證據選擇：根據策略和預測的說服力調整
        evidence = rag_result['best_evidence']
        evidence_confidence = min(1.0, rag_result['total_evidence'] / 5.0)
        
        # 根據預測的 Delta 概率調整證據信心度
        adjusted_confidence = evidence_confidence * (0.5 + 0.5 * delta_probability)
        
        print(f"  ✨ 融合結果: 最終策略={adjusted_strategy}, 證據信心={adjusted_confidence:.2f}, Delta概率={delta_probability:.2f}")
        
        return {
            'final_strategy': adjusted_strategy,
            'evidence': evidence,
            'evidence_confidence': adjusted_confidence,
            'social_influence': influence_score,
            'stance_strength': abs(current_stance),
            'conviction': gnn_result['conviction'],
            'delta_probability': delta_probability,
            'gnn_suggested_strategy': gnn_strategy,
            'fusion_timestamp': time.time()
        }
    
    async def generate_response(self, agent_id: str, topic: str, 
                              history: List[str], target_agents: List[str]) -> str:
        """生成辯論回覆"""
        
        # 1. 平行分析
        print(f"\n🔄 Agent {agent_id} 開始平行分析...")
        print(f"   📝 主題: {topic}")
        print(f"   📊 歷史回合數: {len(history)}")
        analysis_start = time.time()
        
        analysis_results = await self.parallel_analysis(agent_id, topic, history)
        
        analysis_time = time.time() - analysis_start
        print(f"⚡ 平行分析完成 ({analysis_time:.2f}s)")
        
        # 2. 融合結果
        print(f"🔀 開始融合分析結果...")
        fused_results = self.fuse_analysis_results(analysis_results, agent_id)
        print(f"✅ 融合完成")
        
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
        generation_start = time.time()
        response = chat(prompt)
        generation_time = time.time() - generation_start
        print(f"✅ 回覆生成完成 ({generation_time:.2f}s)")
        
        # 檢查回應是否被截斷（檢查是否以句號、問號或驚嘆號結尾）
        if response and not response.rstrip().endswith(('。', '！', '？', '.', '!', '?')):
            print(f"⚠️ 檢測到回應可能被截斷，嘗試補充完整...")
            # 如果回應被截斷，添加結尾
            response += "。總之，基於以上分析，我堅持我的立場。"
        
        # 5. 評估回覆效果
        response_effects = self._evaluate_response(response, target_agents)
        
        total_time = time.time() - analysis_start
        print(f"⏱️ Agent {agent_id} 總處理時間: {total_time:.2f}s")
        
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
        
        # 判斷是否為第一回合
        is_first_round = len(history) == 0
        
        # 歷史對話
        history_text = '\n'.join(f"Turn {i+1}: {turn}" for i, turn in enumerate(history))
        
        # 分析已使用的論點（避免重複）
        used_arguments = self._extract_used_arguments(history, agent_id)
        
        # 目標分析（第一回合不需要）
        target_info = ""
        if not is_first_round:
            for target_id, analysis in target_analysis.items():
                target_info += f"\n- {target_id}: 立場{analysis['stance']:.2f}, 信念{analysis['conviction']:.2f}, 說服機會{analysis['persuasion_opportunity']:.2f}"
        
        # 增強的策略指導
        strategy_guidance = {
            'aggressive': "採用批判性分析策略：深入剖析對方論點的邏輯缺陷，用有力的反例和數據挑戰其核心假設。",
            'defensive': "採用穩健論證策略：鞏固自己的核心論點，系統性地回應質疑，用更多證據強化立場。",
            'analytical': "採用理性分析策略：運用邏輯推理、實證數據和案例研究，客觀地評估各種觀點的優劣。",
            'empathetic': "採用建設性對話策略：理解對方的合理關切，尋找共同點，提出兼顧各方利益的解決方案。"
        }
        
        # 辯論風格提示
        if is_first_round:
            debate_style = """
這是關於公共議題的理性辯論，你需要：
- 清楚陳述你對議題的立場（支持或反對）
- 提出你的核心論點和理由
- 使用事實和邏輯支持你的觀點
- 展現批判性思維能力
- 不要提及其他參與者（因為你還沒聽過他們的發言）
"""
            round_instruction = """
第一回合要求：
1. 明確表明你是支持還是反對該議題
2. 提出3-4個核心論點
3. 使用具體事實和數據支持你的立場
4. 從社會影響、經濟效益、長遠發展等角度分析
5. 不要提及其他參與者
6. 字數要求：200-250字
7. 必須完成完整的論述，不要中途截斷
"""
        else:
            # 根據回合數調整策略
            round_num = len([h for h in history if agent_id in h]) + 1
            if round_num >= 3 and agent_state.conviction < 0.4:
                debate_style = """
經過深入討論，你開始重新思考這個議題：
- 承認對方某些論點確實有道理
- 反思自己立場的局限性
- 嘗試尋找中間立場或妥協方案
- 展現開放和理性的態度
"""
            elif round_num >= 4:
                debate_style = """
辯論進入總結階段，你需要：
- 整合各方觀點，提出全面的分析
- 指出核心分歧點和共識
- 提出可行的解決方案或建議
- 為這個公共議題的討論做出建設性貢獻
"""
            else:
                debate_style = """
這是一場深入的公共議題辯論，你需要：
- 回應對方的具體論點
- 提供更多證據和分析
- 從不同角度深化你的論證
- 保持理性和專業的討論氛圍
"""
            
            # 避免重複的指導
            avoid_repetition = ""
            if used_arguments:
                avoid_repetition = f"""
注意：你已經討論過以下方面，請探討新的角度：
{chr(10).join(f"- {arg}" for arg in used_arguments[:3])}

嘗試從其他維度分析，如：社會公平、可持續發展、國際比較、歷史經驗等。
"""
            
            round_instruction = f"""
辯論要求：
1. 回應對方的核心論點，指出邏輯漏洞或提供反證
2. 深化你的論證，提供新的視角和證據
3. 使用「但是」、「然而」、「另一方面」等連接詞
4. 引用具體案例或數據時使用 [CITE] 標記
5. 保持客觀理性，避免人身攻擊
6. 如果被說服，可以適當調整立場
7. 避免重複已經討論過的內容
8. 字數要求：200-250字
9. 必須完成完整的論述，確保論點有頭有尾

{avoid_repetition}
"""
        
        # 構建提示
        if is_first_round:
            return f"""你正在參與一場關於「{topic}」的公共議題辯論。

{debate_style}

你的角色設定：
- 立場傾向：{agent_state.current_stance:.2f} (正值傾向支持，負值傾向反對)
- 堅定程度：{agent_state.conviction:.2f} (越高越不易改變立場)
- 論證風格：{strategy}

可用論據：
{evidence}

{round_instruction}

請發表你的觀點："""
        else:
            return f"""你正在參與一場關於「{topic}」的公共議題辯論。

{debate_style}

你的當前狀態：
- 立場傾向：{agent_state.current_stance:.2f} (正值傾向支持，負值傾向反對)
- 堅定程度：{agent_state.conviction:.2f} (越高越不易改變立場)
- 論證風格：{strategy}

討論記錄：
{history_text}

可用論據：
{evidence}

其他參與者狀態：{target_info}

論證策略：{strategy_guidance.get(strategy, '')}

{round_instruction}

請發表你的觀點："""
    
    def _extract_used_arguments(self, history: List[str], agent_id: str) -> List[str]:
        """提取已使用的論點關鍵詞"""
        used_arguments = []
        
        # 簡單的關鍵詞提取
        keywords = ['經濟', '失業率', '外交', '貿易', '環境', '氣候', '安全', '政策', 
                   '財政', '赤字', '國際', '領導', '社會', '分裂']
        
        for turn in history:
            if agent_id in turn:
                for keyword in keywords:
                    if keyword in turn and keyword not in used_arguments:
                        used_arguments.append(keyword)
        
        return used_arguments
    
    def _evaluate_response(self, response: str, target_agents: List[str]) -> Dict:
        """評估回覆的說服力和攻擊性"""
        
        # 中英文關鍵詞評估
        persuasion_indicators = [
            'however', 'consider', 'understand', 'perspective', 'common',
            '但是', '考慮', '理解', '觀點', '共同', '認同', '同意'
        ]
        attack_indicators = [
            'wrong', 'flawed', 'mistake', 'ignore', 'fail',
            '錯誤', '缺陷', '謬誤', '忽視', '失敗', '荒謬', '不合理'
        ]
        evidence_indicators = [
            '[CITE]', 'study', 'research', 'data', 'evidence',
            '研究', '數據', '證據', '事實', '統計', '報告', '調查'
        ]
        
        # 計算分數（使用更敏感的計算方式）
        response_lower = response.lower()
        persuasion_count = sum(1 for indicator in persuasion_indicators if indicator in response_lower)
        attack_count = sum(1 for indicator in attack_indicators if indicator in response_lower)
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in response_lower)
        
        # 調整分數計算，使其更容易獲得非零分數
        persuasion_score = min(1.0, persuasion_count * 0.3)
        attack_score = min(1.0, attack_count * 0.4)
        evidence_score = min(1.0, evidence_count * 0.35)
        
        # 長度分數
        word_count = len(response.split())
        length_score = min(1.0, word_count / 80)
        
        return {
            'persuasion_score': persuasion_score,
            'attack_score': attack_score,
            'evidence_score': evidence_score,
            'length_score': length_score
        }
    
    async def run_debate_round(self, round_number: int, topic: str, 
                             agent_order: List[str]) -> DebateRound:
        """執行一輪辯論"""
        
        print(f"\n🎭 開始第 {round_number} 輪辯論")
        print(f"主題: {topic}")
        print(f"發言順序: {' → '.join(agent_order)}")
        
        # 獲取所有之前的歷史記錄
        all_history = []
        for past_round in self.debate_history:
            for response in past_round.history:
                all_history.append(f"{response['agent_id']}: {response['content']}")
        
        # 當前回合的回應
        round_responses = []
        
        for i, agent_id in enumerate(agent_order):
            print(f"\n--- Agent {agent_id} 發言 ---")
            
            # 確定目標對手
            other_agents = [aid for aid in agent_order if aid != agent_id]
            
            # 組合歷史：之前所有回合 + 當前回合已發言
            current_history = all_history + [f"{r['agent_id']}: {r['content']}" for r in round_responses]
            
            # 生成回覆
            response = await self.generate_response(
                agent_id, topic, 
                current_history, 
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
        """獲取辯論總結和勝負判定"""
        if not self.debate_history:
            return {"message": "尚未開始辯論"}
        
        # 統計投降情況
        surrendered_agents = [aid for aid, state in self.agent_states.items() 
                            if state.has_surrendered]
        
        # 計算每個 Agent 的綜合得分
        agent_scores = {}
        for agent_id, state in self.agent_states.items():
            # 基礎分數
            score = 0
            
            # 立場堅定度得分（立場越極端且信念越強得分越高）
            stance_score = abs(state.current_stance) * state.conviction * 30
            score += stance_score
            
            # 說服他人得分（根據其他人的投降和立場改變）
            persuasion_score = 0
            for other_id, other_state in self.agent_states.items():
                if other_id != agent_id:
                    if other_state.has_surrendered:
                        persuasion_score += 20
                    # 計算對其他人的影響
                    if len(other_state.persuasion_history) > 0:
                        avg_persuasion = sum(other_state.persuasion_history) / len(other_state.persuasion_history)
                        persuasion_score += avg_persuasion * 10
            score += persuasion_score
            
            # 抗壓能力得分（被攻擊但仍保持立場）
            if len(state.attack_history) > 0:
                avg_attack = sum(state.attack_history) / len(state.attack_history)
                resistance_score = (1 - avg_attack) * state.conviction * 20
                score += resistance_score
            
            # 投降扣分
            if state.has_surrendered:
                score -= 50
            
            agent_scores[agent_id] = score
        
        # 判定獲勝者
        winner = max(agent_scores.keys(), key=lambda x: agent_scores[x])
        
        # 生成總結報告
        summary = {
            "total_rounds": len(self.debate_history),
            "winner": winner,
            "scores": agent_scores,
            "surrendered_agents": surrendered_agents,
            "final_states": {},
            "verdict": ""
        }
        
        # 添加最終狀態
        for aid, state in self.agent_states.items():
            summary["final_states"][aid] = {
                "stance": state.current_stance,
                "conviction": state.conviction,
                "has_surrendered": state.has_surrendered,
                "final_position": "支持" if state.current_stance > 0 else "反對"
            }
        
        # 生成裁決詞
        if len(surrendered_agents) > 0:
            summary["verdict"] = f"🏆 {winner} 獲得壓倒性勝利！成功說服 {', '.join(surrendered_agents)} 投降。"
        else:
            score_diff = agent_scores[winner] - sorted(agent_scores.values())[-2]
            if score_diff > 30:
                summary["verdict"] = f"🏆 {winner} 以明顯優勢獲勝！展現了卓越的辯論技巧。"
            else:
                summary["verdict"] = f"🏆 {winner} 險勝！這是一場勢均力敵的精彩辯論。"
        
        return summary

# 便利函數
def create_parallel_orchestrator():
    """創建平行協調器"""
    return ParallelOrchestrator() 