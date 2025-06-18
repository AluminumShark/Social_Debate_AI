"""
Flask 後端 API for Social Debate AI
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import asyncio
import json
import os
from datetime import datetime
import uuid

# 添加專案路徑
import sys
from pathlib import Path
# 修正路徑：從 ui 目錄回到專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 導入必要模組
try:
    from orchestrator.parallel_orchestrator import create_parallel_orchestrator
    from dialogue.dialogue_manager import DialogueManager
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"導入錯誤: {e}")
    sys.exit(1)

# 初始化 Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'social-debate-ai-secret-key-2024')
CORS(app)

# 全局變數
orchestrator = None
dialogue_manager = None
config = None

def init_system():
    """初始化系統"""
    global orchestrator, dialogue_manager, config
    
    try:
        # 步驟 1: 載入配置
        print("📋 [1/4] 載入系統配置...")
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "debate.yaml"
        if config_path.exists():
            config = ConfigLoader.load("debate", str(project_root / "configs"))
            print(f"   ✅ 成功載入配置: {config_path}")
            print(f"   📊 最大回合數: {config.get('debate', {}).get('max_rounds', 5)}")
            print(f"   👥 參與者: {', '.join(config.get('debate', {}).get('agents', []))}")
        else:
            config = {
                'debate': {
                    'max_rounds': 5,
                    'agents': ['Agent_A', 'Agent_B', 'Agent_C']
                }
            }
            print("   ⚠️ 配置檔案不存在，使用預設配置")
        
        # 步驟 2: 初始化協調器
        print("\n🎯 [2/4] 初始化平行協調器...")
        orchestrator = create_parallel_orchestrator()
        print("   ✅ 協調器初始化成功")
        print("   📌 支援模組: RL策略選擇、GNN社會編碼、RAG證據檢索")
        
        # 步驟 3: 初始化對話管理器
        print("\n💬 [3/4] 初始化對話管理器...")
        dialogue_manager = DialogueManager()
        print("   ✅ 對話管理器初始化成功")
        print("   📝 支援功能: 對話歷史管理、回合控制、狀態追蹤")
        
        # 步驟 4: 初始化 Agents
        print("\n🤖 [4/4] 初始化智能體...")
        agent_configs = [
            {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
            {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.6},
            {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.5}
        ]
        orchestrator.initialize_agents(agent_configs)
        print("   ✅ 成功初始化 3 個智能體:")
        print("   🔴 Agent_A: 積極支持型 (立場: +0.8)")
        print("   🔵 Agent_B: 反對質疑型 (立場: -0.6)")
        print("   🟢 Agent_C: 中立分析型 (立場: 0.0)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 系統初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """主頁"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def api_init():
    """初始化 API"""
    success = init_system()
    if success:
        # 初始化 session（只存儲必要的信息）
        session['debate_id'] = str(uuid.uuid4())
        session['current_round'] = 0
        session['topic'] = ""
        
        return jsonify({
            'success': True,
            'message': '系統初始化成功',
            'debate_id': session['debate_id']
        })
    else:
        return jsonify({
            'success': False,
            'message': '系統初始化失敗'
        }), 500

@app.route('/api/set_topic', methods=['POST'])
def api_set_topic():
    """設置辯論主題"""
    data = request.json
    topic = data.get('topic', '')
    
    if not topic:
        return jsonify({
            'success': False,
            'message': '請輸入辯論主題'
        }), 400
    
    session['topic'] = topic
    session['current_round'] = 0
    
    # 重置 orchestrator 的辯論歷史和 Agent 狀態
    if orchestrator:
        orchestrator.debate_history = []
        # 重新初始化 Agents
        agent_configs = [
            {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
            {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.6},
            {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.5}
        ]
        orchestrator.initialize_agents(agent_configs)
    
    return jsonify({
        'success': True,
        'topic': topic,
        'message': f'辯論主題已設置: {topic}'
    })

@app.route('/api/debate_round', methods=['POST'])
def api_debate_round():
    """執行一輪辯論"""
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統未初始化'
        }), 500
    
    topic = session.get('topic', '')
    if not topic:
        return jsonify({
            'success': False,
            'message': '請先設置辯論主題'
        }), 400
    
    # 增加回合數
    session['current_round'] = session.get('current_round', 0) + 1
    current_round = session['current_round']
    
    # 檢查是否已有人投降
    has_surrender = any(state.has_surrendered for state in orchestrator.agent_states.values())
    max_rounds = config.get('debate', {}).get('max_rounds', 5)
    
    # 檢查是否應該在執行回合前就結束辯論
    if current_round > max_rounds:
        # 辯論已超過最大回合數
        summary = orchestrator.get_debate_summary()
        
        # 獲取當前 Agent 狀態
        agent_states = {}
        for agent_id, state in orchestrator.agent_states.items():
            agent_states[agent_id] = {
                'stance': state.current_stance,
                'conviction': state.conviction,
                'has_surrendered': state.has_surrendered,
                'persuasion_avg': sum(state.persuasion_history[-3:]) / min(3, len(state.persuasion_history)) if state.persuasion_history else 0
            }
        
        return jsonify({
            'success': True,
            'round': current_round - 1,  # 返回實際的最後一輪
            'responses': [],  # 空回應列表
            'agent_states': agent_states,
            'debate_ended': True,
            'summary': summary,
            'message': '辯論已結束（達到最大回合數）'
        })
    
    try:
        # 使用異步執行器來運行異步函數
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 執行辯論回合
        debate_round = loop.run_until_complete(
            orchestrator.run_debate_round(
                round_number=current_round,
                topic=topic,
                agent_order=['Agent_A', 'Agent_B', 'Agent_C']
            )
        )
        
        loop.close()
        
        # 整理回合數據
        round_data = {
            'round': current_round,
            'responses': []
        }
        
        # 確保 debate_round.history 存在
        if hasattr(debate_round, 'history') and debate_round.history:
            for response in debate_round.history:
                round_data['responses'].append({
                    'agent_id': response.get('agent_id', ''),
                    'content': response.get('content', ''),
                    'effects': response.get('effects', {'persuasion_score': 0, 'attack_score': 0}),
                    'timestamp': response.get('timestamp', '')
                })
        
        # 獲取 Agent 狀態
        agent_states = {}
        if hasattr(debate_round, 'agent_states') and debate_round.agent_states:
            for agent_id, state in debate_round.agent_states.items():
                agent_states[agent_id] = {
                    'stance': state.current_stance,
                    'conviction': state.conviction,
                    'has_surrendered': state.has_surrendered,
                    'persuasion_avg': sum(state.persuasion_history[-3:]) / min(3, len(state.persuasion_history)) if state.persuasion_history else 0
                }
        else:
            # 如果沒有新的狀態，使用 orchestrator 的當前狀態
            for agent_id, state in orchestrator.agent_states.items():
                agent_states[agent_id] = {
                    'stance': state.current_stance,
                    'conviction': state.conviction,
                    'has_surrendered': state.has_surrendered,
                    'persuasion_avg': sum(state.persuasion_history[-3:]) / min(3, len(state.persuasion_history)) if state.persuasion_history else 0
                }
        
        # 不再將完整歷史存儲在 session 中
        # 只存儲回合數和基本信息
        session['current_round'] = current_round
        session.modified = True
        
        # 檢查是否有人投降或達到最大回合數
        debate_ended = any(state['has_surrendered'] for state in agent_states.values()) or current_round >= max_rounds
        
        response_data = {
            'success': True,
            'round': current_round,
            'responses': round_data['responses'],
            'agent_states': agent_states,
            'debate_ended': debate_ended,
            'message': f'第 {current_round} 輪辯論完成'
        }
        
        # 如果辯論結束，添加總結
        if debate_ended:
            response_data['summary'] = orchestrator.get_debate_summary()
            response_data['message'] = '辯論已結束！'
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"❌ 辯論執行失敗: {str(e)}")
        print(traceback.format_exc())
        
        # 返回錯誤時也要包含必要的數據結構
        return jsonify({
            'success': False,
            'round': current_round,
            'responses': [],
            'agent_states': {},
            'debate_ended': False,
            'message': f'辯論執行失敗: {str(e)}'
        }), 500

@app.route('/api/debate_history', methods=['GET'])
def api_debate_history():
    """獲取辯論歷史"""
    # 從 orchestrator 獲取歷史，而不是從 session
    history = []
    if orchestrator and hasattr(orchestrator, 'debate_history'):
        for debate_round in orchestrator.debate_history:
            round_data = {
                'round': debate_round.round_number,
                'responses': []
            }
            for response in debate_round.history:
                round_data['responses'].append({
                    'agent_id': response['agent_id'],
                    'content': response['content'],
                    'effects': response['effects']
                })
            history.append(round_data)
    
    return jsonify({
        'success': True,
        'topic': session.get('topic', ''),
        'current_round': session.get('current_round', 0),
        'history': history
    })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置辯論"""
    session.clear()
    if init_system():
        session['debate_id'] = str(uuid.uuid4())
        return jsonify({
            'success': True,
            'message': '辯論已重置',
            'debate_id': session['debate_id']
        })
    else:
        return jsonify({
            'success': False,
            'message': '重置失敗'
        }), 500

@app.route('/api/export', methods=['GET'])
def api_export():
    """導出辯論記錄"""
    # 從 orchestrator 獲取完整歷史
    history = []
    if orchestrator and hasattr(orchestrator, 'debate_history'):
        for debate_round in orchestrator.debate_history:
            round_data = {
                'round': debate_round.round_number,
                'responses': []
            }
            for response in debate_round.history:
                round_data['responses'].append({
                    'agent_id': response['agent_id'],
                    'content': response['content'],
                    'effects': response['effects'],
                    'timestamp': response.get('timestamp', '')
                })
            history.append(round_data)
    
    debate_data = {
        'debate_id': session.get('debate_id', ''),
        'topic': session.get('topic', ''),
        'total_rounds': session.get('current_round', 0),
        'history': history,
        'exported_at': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'data': debate_data
    })

@app.route('/api/debate_summary', methods=['GET'])
def api_debate_summary():
    """獲取辯論總結和勝負判定"""
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統未初始化'
        }), 500
    
    summary = orchestrator.get_debate_summary()
    
    return jsonify({
        'success': True,
        'summary': summary
    })

if __name__ == '__main__':
    # 初始化系統
    if init_system():
        print("🚀 Flask 伺服器啟動中...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ 無法啟動伺服器") 