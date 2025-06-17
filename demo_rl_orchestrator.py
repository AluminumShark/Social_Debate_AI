#!/usr/bin/env python3
"""
RL 增強辯論系統使用示例
展示如何使用 RL 策略選擇和證據片段選擇
"""

import sys
from pathlib import Path

# 添加 src 路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_rl_policy_network():
    """示範 RL Policy Network 功能"""
    print("🎯 RL Policy Network 示範")
    print("=" * 50)
    
    from rl.policy_network import choose_snippet, select_strategy, PolicyNetwork
    
    # 1. 策略選擇示例
    print("\n1️⃣ 策略選擇示例:")
    queries = [
        "Should we implement universal healthcare?",
        "Is climate change caused by human activities?", 
        "Should social media be regulated?"
    ]
    
    for query in queries:
        strategy = select_strategy(query)
        print(f"   查詢: {query}")
        print(f"   選擇策略: {strategy}")
        print()
    
    # 2. 片段選擇示例
    print("2️⃣ 證據片段選擇示例:")
    query = "Should we implement universal healthcare?"
    pool = [
        {
            'content': 'Universal healthcare systems reduce administrative costs by eliminating the complex insurance bureaucracy, leading to significant savings.',
            'similarity_score': 0.85,
            'score': 1.2,
            'type': 'delta_comment'
        },
        {
            'content': 'Private healthcare encourages innovation and competition, resulting in better quality care and faster access to new treatments.',
            'similarity_score': 0.78,
            'score': 1.0,
            'type': 'submission'
        },
        {
            'content': 'Healthcare is a fundamental human right that should be accessible to all citizens regardless of their economic status.',
            'similarity_score': 0.72,
            'score': 0.9,
            'type': 'delta_comment'
        },
        {
            'content': 'Government-run healthcare systems often face long waiting times and resource allocation challenges.',
            'similarity_score': 0.68,
            'score': 0.8,
            'type': 'submission'
        }
    ]
    
    print(f"   查詢: {query}")
    print(f"   可選證據片段: {len(pool)} 個")
    
    chosen = choose_snippet(query, pool)
    print(f"   選擇的片段: {chosen[:100]}...")
    print()

def demo_orchestrator_integration():
    """示範 Orchestrator 整合"""
    print("🎭 Orchestrator 整合示範")
    print("=" * 50)
    
    try:
        from orchestrator.orchestrator import create_enhanced_orchestrator
        
        # 創建 orchestrator
        orchestrator = create_enhanced_orchestrator()
        print("✅ Enhanced Orchestrator 創建成功")
        
        # 辯論場景設定
        topic = "Climate change mitigation strategies"
        history = [
            "Agent A: We need immediate aggressive action on climate change, including carbon taxes and renewable energy mandates.",
            "Agent B: While climate action is important, we must balance environmental goals with economic realities and job preservation.",
            "Agent A: The economic costs of inaction far exceed the costs of transition. Studies show green jobs can replace fossil fuel jobs.",
            "Agent B: Rapid transitions can devastate communities dependent on traditional industries. We need gradual, market-based solutions."
        ]
        
        print(f"\n辯論主題: {topic}")
        print(f"歷史對話: {len(history)} 輪")
        
        # 社會背景向量 (模擬)
        social_context = [0.2, -0.1, 0.5, 0.3, -0.2, 0.4, 0.1, -0.3]
        
        # 測試策略配置
        print("\n3️⃣ 策略配置測試:")
        strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
        for strategy in strategies:
            config = orchestrator._get_strategy_config(strategy)
            print(f"   {strategy}: k={config['k']}, type={config['index_type']}, persuasion={config['persuasion_only']}")
        
        print("\n4️⃣ RL 增強回覆生成 (模擬):")
        print("   (實際生成需要 RAG 索引和 GPT API)")
        
        # 模擬 RL 增強流程
        from rl.policy_network import select_strategy
        recent = '\n'.join(history[-2:])
        state_text = f"Topic: {topic}\nRecent turns: {recent}"
        
        selected_strategy = select_strategy(state_text, recent, social_context)
        print(f"   RL 選擇策略: {selected_strategy}")
        
        strategy_config = orchestrator._get_strategy_config(selected_strategy)
        print(f"   策略配置: {strategy_config}")
        
        # 模擬證據池和片段選擇
        mock_pool = [
            {
                'content': 'Carbon pricing mechanisms have proven effective in reducing emissions while maintaining economic growth in countries like Sweden and British Columbia.',
                'similarity_score': 0.88,
                'score': 1.4,
                'type': 'delta_comment'
            },
            {
                'content': 'Just transition policies can help retrain workers from fossil fuel industries for renewable energy jobs, ensuring no community is left behind.',
                'similarity_score': 0.82,
                'score': 1.2,
                'type': 'delta_comment'
            }
        ]
        
        from rl.policy_network import choose_snippet
        chosen = choose_snippet(state_text, mock_pool)
        print(f"   選擇的證據: {chosen[:80]}...")
        
        # 構建提示範例
        prompt_example = f"""
Topic: {topic}
Recent turns:
{recent}

Social: {social_context[:5]}...
Evidence Snippet: "{chosen[:100]}..."

Strategy: {selected_strategy}

Write ≤120 words persuading the opponent using {selected_strategy} approach. Cite as [CITE].
"""
        
        print(f"   生成的提示範例:")
        print("   " + "─" * 40)
        for line in prompt_example.strip().split('\n'):
            print(f"   {line}")
        print("   " + "─" * 40)
        
    except Exception as e:
        print(f"❌ Orchestrator 示範失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主示範函數"""
    print("🚀 RL 增強辯論系統示範")
    print("=" * 60)
    
    try:
        # 示範 RL Policy Network
        demo_rl_policy_network()
        
        print("\n" + "=" * 60)
        
        # 示範 Orchestrator 整合
        demo_orchestrator_integration()
        
        print("\n🎉 示範完成！")
        print("\n📋 總結:")
        print("✅ RL Policy Network: 策略選擇和片段選擇")
        print("✅ Orchestrator 整合: RL 增強的回覆生成")
        print("✅ 策略配置: 4種辯論策略 (aggressive, defensive, analytical, empathetic)")
        print("✅ 社會背景: 支援社會向量輸入")
        print("✅ 證據選擇: 基於品質和相關性的智能選擇")
        
    except Exception as e:
        print(f"❌ 示範失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 