"""
Enhanced Debate Orchestrator
Supports diverse retrieval strategies, topic analysis, and argument pattern recognition
"""

try:
    from ..gpt_interface.gpt_client import chat
except ImportError:
    from gpt_interface.gpt_client import chat

try:
    from ..rag.retriever import create_enhanced_retriever
except ImportError:
    from rag.retriever import create_enhanced_retriever

from typing import List, Dict, Optional
import re

try:
    from ..gnn.social_encoder import social_vec
except ImportError:
    try:
        from gnn.social_encoder import social_vec
    except ImportError:
        def social_vec(author):
            """Placeholder function for social vector"""
            return [0.0] * 128

try:
    from ..rl.policy_network import select_strategy, choose_snippet
except ImportError:
    try:
        from rl.policy_network import select_strategy, choose_snippet
    except ImportError:
        def select_strategy(query):
            """Placeholder function for strategy selection"""
            return 'balanced'
        
        def choose_snippet(state_text, pool):
            """Placeholder function for snippet selection"""
            if not pool:
                return "No evidence available."
            return max(pool, key=lambda x: x.get('similarity_score', 0.0))['content']

class EnhancedOrchestrator:
    """Enhanced Debate Orchestrator"""
    
    def __init__(self):
        try:
            self.retriever = create_enhanced_retriever()
            print("✅ Enhanced debate orchestrator initialized successfully")
            
            # Retrieval strategy configurations
            self.strategies = {
                'balanced': {'hq_ratio': 0.6, 'diverse': True},
                'expert': {'hq_ratio': 0.9, 'persuasion_only': True},
                'comprehensive': {'hq_ratio': 0.3, 'index_type': 'comprehensive'}
            }
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent and topics"""
        # Simple intent analysis
        intent = {
            'topics': [],
            'complexity': 'intermediate',
            'debate_style': 'balanced',
            'requires_evidence': True
        }
        
        # Detect topic distribution
        topic_dist = self.retriever.get_topic_distribution(query)
        intent['topics'] = list(topic_dist.keys())[:3]  # Top 3 relevant topics
        
        # Detect complexity requirements
        if any(word in query.lower() for word in ['simple', 'basic', 'explain']):
            intent['complexity'] = 'simple'
        elif any(word in query.lower() for word in ['complex', 'detailed', 'analysis']):
            intent['complexity'] = 'complex'
        
        # Detect debate style
        if any(word in query.lower() for word in ['evidence', 'proof', 'data']):
            intent['debate_style'] = 'expert'
        elif any(word in query.lower() for word in ['opinion', 'various', 'different']):
            intent['debate_style'] = 'comprehensive'
        
        return intent
    
    def gather_evidence_adaptive(self, query: str, strategy: str = 'balanced') -> List[Dict]:
        """Adaptive evidence gathering"""
        config = self.strategies.get(strategy, self.strategies['balanced'])
        
        evidence = []
        
        # High-quality evidence
        if config.get('hq_ratio', 0.5) > 0:
            hq_count = max(1, int(5 * config['hq_ratio']))
            hq_evidence = self.retriever.retrieve(
                query=query,
                k=hq_count,
                index_type='high_quality',
                persuasion_only=config.get('persuasion_only', False)
            )
            evidence.extend(hq_evidence)
        
        # Diverse evidence
        if config.get('diverse', False):
            diverse_count = 5 - len(evidence)
            if diverse_count > 0:
                diverse_evidence = self.retriever.retrieve_diverse_perspectives(
                    query=query,
                    k=diverse_count
                )
                evidence.extend(diverse_evidence)
        
        # Comprehensive evidence
        if config.get('index_type') == 'comprehensive':
            comp_count = 5 - len(evidence)
            if comp_count > 0:
                comp_evidence = self.retriever.retrieve(
                    query=query,
                    k=comp_count,
                    index_type='comprehensive'
                )
                evidence.extend(comp_evidence)
        
        # Deduplication and sorting
        seen = set()
        unique_evidence = []
        for ev in evidence:
            if ev['content'] not in seen:
                seen.add(ev['content'])
                unique_evidence.append(ev)
        
        # Sort by similarity and quality
        unique_evidence.sort(
            key=lambda x: (x['similarity_score'], x['score']), 
            reverse=True
        )
        
        print(f"📚 Gathered {len(unique_evidence)} adaptive evidence pieces")
        return unique_evidence[:5]
    
    def build_enhanced_prompt(self, topic: str, history: List[str], agent: str, 
                            evidence: List[Dict], intent: Dict) -> str:
        """Build enhanced debate prompt"""
        
        # Adjust prompt based on intent
        if intent['debate_style'] == 'expert':
            role_desc = "expert debater"
            tone_req = "Use professional terminology and data-driven arguments"
        elif intent['debate_style'] == 'comprehensive':
            role_desc = "multi-perspective analyst"
            tone_req = "Consider various viewpoints and positions"
        else:
            role_desc = "rational discussant"
            tone_req = "Balance different perspectives fairly"
        
        # Format history
        if history:
            recent_history = history[-6:]
            hist_text = '\n'.join(f"- {turn}" for turn in recent_history)
        else:
            hist_text = '(Debate beginning)'
        
        # Categorize and format evidence
        evidence_by_type = {'submission': [], 'delta_comment': []}
        for ev in evidence:
            ev_type = ev.get('type', 'submission')
            evidence_by_type[ev_type].append(ev)
        
        evidence_text = ""
        if evidence_by_type['delta_comment']:
            evidence_text += "**Successful Persuasion Cases:**\n"
            for i, ev in enumerate(evidence_by_type['delta_comment'], 1):
                topics_str = ', '.join(ev.get('topics', []))
                evidence_text += f"[SUCCESS-{i}] ({topics_str}) {ev['content'][:200]}...\n"
        
        if evidence_by_type['submission']:
            evidence_text += "\n**Related Discussions:**\n"
            for i, ev in enumerate(evidence_by_type['submission'], 1):
                topics_str = ', '.join(ev.get('topics', []))
                evidence_text += f"[DISCUSS-{i}] ({topics_str}) {ev['content'][:200]}...\n"
        
        if not evidence_text:
            evidence_text = "(No relevant evidence found)"
        
        # Topic context
        topic_context = ""
        if intent['topics']:
            topic_context = f"**Related Topic Areas:** {', '.join(intent['topics'])}\n"
        
        return f"""You are an {role_desc} participating in a rational debate about "{topic}", playing Agent {agent}.

{topic_context}
### 💬 Conversation History
{hist_text}

### 📚 Evidence Database
{evidence_text}

### 🎯 Response Requirements
Write your next response (≤180 words):

1. **Argument Strategy**: {tone_req}
2. **Evidence Citation**: Use **[SUCCESS-X]** or **[DISCUSS-X]** format for citations
3. **Complexity**: Suitable for {intent['complexity']}-level discussion
4. **Objective**: Present persuasive arguments that promote constructive dialogue

⚠️ Important: Only use the evidence provided above, maintain factual accuracy."""
    
    def get_reply(self, topic: str, history: List[str], agent: str, 
                  strategy: str = 'balanced') -> str:
        """Get enhanced debate response"""
        print(f"🤖 Agent {agent} starting enhanced analysis...")
        
        # Analyze query intent
        query = history[-1] if history else topic
        intent = self.analyze_query_intent(query)
        print(f"🎯 Intent identified: {intent['debate_style']}, topics: {intent['topics']}")
        
        # 1. 取適應證據
        evidence = self.gather_evidence_adaptive(query, strategy)

        # 2. 取得社交向量 & 策略
        author = agent # 這裡假設 agent 名 = 作者名；若不同請傳入
        social_feature = social_vec(author)
        dyn_strategy = select_strategy(query)

        # 3. Build prompt
        prompt = self.build_enhanced_prompt(
            topic, history, agent, evidence, intent,
            ) + f'\n\nSocial embedding: {social_feature}\nSuggested strategy: {dyn_strategy}'

        # Generate response
        try:
            response = chat(prompt)
            print(f"✅ Agent {agent} enhanced response generated successfully")
            return response
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I apologize, but I encountered a technical issue while analyzing this complex topic. Let me reorganize my thoughts..."
    
    def get_rl_enhanced_reply(self, topic: str, history: List[str], agent: str,
                            social_context: List[float] = None) -> str:
        """使用 RL 策略增強的回覆生成"""
        try:
            print(f"🤖 Agent {agent} 開始 RL 增強分析...")
            
            # 構建狀態文本
            recent_turns = history[-3:] if history else []
            recent = '\n'.join(recent_turns) if recent_turns else "(Debate beginning)"
            state_text = f"Topic: {topic}\nRecent turns: {recent}"
            
            # 使用 RL 選擇策略
            selected_strategy = select_strategy(state_text, recent, social_context)
            print(f"🎯 RL 選擇策略: {selected_strategy}")
            
            # 根據策略調整檢索參數
            strategy_config = self._get_strategy_config(selected_strategy)
            
            # 檢索證據池
            pool = self.retriever.retrieve(
                query=state_text,
                k=strategy_config['k'],
                index_type=strategy_config.get('index_type', 'high_quality'),
                persuasion_only=strategy_config.get('persuasion_only', False)
            )
            
            # 使用 RL 選擇最佳片段
            chosen = choose_snippet(state_text, pool)
            print(f"📝 RL 選擇證據片段: {chosen[:100]}...")
            
            # 獲取社會背景信息
            social_info = ""
            if social_context:
                social_info = f"Social context: {social_context[:5]}..."  # 顯示前5個維度
            
            # 構建增強提示
            prompt = f"""
Topic: {topic}
Recent turns:
{recent}

Social: {social_info}
Evidence Snippet: "{chosen}"

Strategy: {selected_strategy}

Write ≤120 words persuading the opponent using {selected_strategy} approach. Cite as [CITE].
"""
            
            # 生成回覆
            reply = chat(prompt)
            
            print(f"🎭 Agent {agent} 使用 {selected_strategy} 策略生成回覆 ({len(reply.split())} words)")
            return reply
            
        except Exception as e:
            print(f"❌ RL 增強回覆生成失敗: {e}")
            # 回退到普通方法
            return self.get_reply(topic, history, agent, 'balanced')
    
    def _get_strategy_config(self, strategy: str) -> Dict:
        """獲取策略配置"""
        configs = {
            'aggressive': {
                'k': 3,
                'index_type': 'high_quality',
                'persuasion_only': True
            },
            'defensive': {
                'k': 5,
                'index_type': 'comprehensive',
                'persuasion_only': False
            },
            'analytical': {
                'k': 4,
                'index_type': 'high_quality',
                'persuasion_only': False
            },
            'empathetic': {
                'k': 3,
                'index_type': 'comprehensive',
                'persuasion_only': True
            }
        }
        return configs.get(strategy, configs['analytical'])
    
    def analyze_debate_flow(self, topic: str, history: List[str]) -> Dict:
        """Analyze debate flow and suggestions"""
        if not history:
            return {'stage': 'opening', 'suggestion': 'Present opening statement and position'}
        
        # Simple flow analysis
        turns = len(history)
        if turns < 4:
            stage = 'opening'
            suggestion = 'Establish basic positions and core arguments'
        elif turns < 8:
            stage = 'development'
            suggestion = 'Deepen arguments with specific evidence'
        else:
            stage = 'conclusion'
            suggestion = 'Summarize key points and seek common ground'
        
        # Analyze recent arguments
        recent_topics = []
        if len(history) >= 2:
            recent_query = ' '.join(history[-2:])
            topic_dist = self.retriever.get_topic_distribution(recent_query)
            recent_topics = list(topic_dist.keys())[:3]
        
        return {
            'stage': stage,
            'turns': turns,
            'suggestion': suggestion,
            'recent_topics': recent_topics,
            'recommended_strategy': 'expert' if stage == 'conclusion' else 'balanced'
        }
    
    def get_topic_suggestions(self, current_topic: str) -> List[str]:
        """Get related topic suggestions"""
        try:
            # Retrieve related discussions
            related = self.retriever.retrieve(current_topic, k=10)
            
            # Extract related topics
            all_topics = []
            for item in related:
                all_topics.extend(item.get('topics', []))
            
            # Count and filter
            from collections import Counter
            topic_count = Counter(all_topics)
            
            # Exclude current topic-related terms
            current_lower = current_topic.lower()
            suggestions = []
            for topic, count in topic_count.most_common(10):
                if topic not in current_lower and count >= 2:
                    suggestions.append(f"{topic} ({count} related discussions)")
            
            return suggestions[:5]
            
        except Exception as e:
            print(f"⚠️ Error getting topic suggestions: {e}")
            return []
    


# Convenience function
def create_enhanced_orchestrator():
    """Create enhanced orchestrator instance"""
    try:
        return EnhancedOrchestrator()
    except Exception as e:
        print(f"❌ Failed to create enhanced orchestrator: {e}")
        raise 