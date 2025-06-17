#!/usr/bin/env python3
"""
平行辯論系統完整示例
展示 RL + GNN + RAG 平行運行和動態說服/反駁機制
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demo_parallel_debate():
    """平行辯論系統示例"""
    
    print("🚀 平行辯論系統示例")
    print("=" * 60)
    
    try:
        from orchestrator.parallel_orchestrator import create_parallel_orchestrator
        
        # 1. 創建平行協調器
        orchestrator = create_parallel_orchestrator()
        
        # 2. 初始化 Agent 配置
        agent_configs = [
            {
                'id': 'Agent_A',
                'initial_stance': 0.7,    # 支持立場
                'initial_conviction': 0.8, # 高信念度
                'social_context': [0.2, 0.5, -0.1, 0.3, 0.4] + [0.0] * 123  # 社會背景
            },
            {
                'id': 'Agent_B', 
                'initial_stance': -0.6,   # 反對立場
                'initial_conviction': 0.7, # 中等信念度
                'social_context': [-0.3, 0.1, 0.4, -0.2, 0.6] + [0.0] * 123
            },
            {
                'id': 'Agent_C',
                'initial_stance': 0.1,    # 中性偏支持
                'initial_conviction': 0.5, # 低信念度，容易被說服
                'social_context': [0.0, 0.2, 0.1, 0.3, -0.1] + [0.0] * 123
            }
        ]
        
        # 3. 初始化 Agent
        agents = orchestrator.initialize_agents(agent_configs)
        
        # 4. 辯論主題
        topic = "Should artificial intelligence development be regulated by government?"
        
        # 5. Agent 發言順序
        agent_order = ['Agent_A', 'Agent_B', 'Agent_C']
        
        print(f"\n🎯 辯論主題: {topic}")
        print(f"🎭 參與者: {', '.join(agent_order)}")
        
        # 顯示初始狀態
        print(f"\n📊 初始狀態:")
        for agent_id, state in agents.items():
            print(f"  {agent_id}: 立場 {state.current_stance:+.2f}, 信念 {state.conviction:.2f}")
        
        # 6. 執行多輪辯論
        total_rounds = 3
        
        for round_num in range(1, total_rounds + 1):
            print(f"\n{'='*60}")
            
            # 執行辯論回合
            debate_round = await orchestrator.run_debate_round(
                round_number=round_num,
                topic=topic,
                agent_order=agent_order
            )
            
            # 檢查立場變化
            if round_num > 1:
                print(f"\n📈 立場變化分析:")
                for agent_id in agent_order:
                    current_state = orchestrator.agent_states[agent_id]
                    prev_round = orchestrator.debate_history[-2]
                    prev_state = prev_round.agent_states[agent_id]
                    
                    stance_change = current_state.current_stance - prev_state.current_stance
                    conviction_change = current_state.conviction - prev_state.conviction
                    
                    print(f"  {agent_id}: 立場變化 {stance_change:+.3f}, 信念變化 {conviction_change:+.3f}")
            
            # 模擬思考時間
            await asyncio.sleep(1)
        
        # 7. 最終總結
        print(f"\n🏆 辯論總結")
        print("=" * 60)
        
        summary = orchestrator.get_debate_summary()
        
        print(f"總回合數: {summary['total_rounds']}")
        print(f"最有說服力的 Agent: {summary['most_persuasive_agent']}")
        
        print(f"\n📊 最終狀態:")
        for agent_id, final_state in summary['final_states'].items():
            stance_change = summary['stance_changes'][agent_id]
            print(f"  {agent_id}:")
            print(f"    最終立場: {final_state['stance']:+.3f} (變化: {stance_change:+.3f})")
            print(f"    最終信念: {final_state['conviction']:.3f}")
        
        # 8. 分析說服效果
        print(f"\n🎯 說服效果分析:")
        
        most_persuaded = min(summary['stance_changes'].keys(), 
                           key=lambda x: abs(summary['stance_changes'][x]))
        least_persuaded = max(summary['stance_changes'].keys(),
                            key=lambda x: abs(summary['stance_changes'][x]))
        
        print(f"  最容易被說服: {most_persuaded} (立場變化: {summary['stance_changes'][most_persuaded]:+.3f})")
        print(f"  最難被說服: {least_persuaded} (立場變化: {summary['stance_changes'][least_persuaded]:+.3f})")
        
        # 9. 顯示辯論歷程
        print(f"\n📜 辯論歷程回顧:")
        for i, round_data in enumerate(orchestrator.debate_history, 1):
            print(f"\n  第 {i} 輪:")
            for response in round_data.history:
                agent_id = response['agent_id']
                content = response['content'][:80] + "..." if len(response['content']) > 80 else response['content']
                effects = response['effects']
                print(f"    {agent_id}: {content}")
                print(f"      效果 - 說服: {effects['persuasion_score']:.2f}, 攻擊: {effects['attack_score']:.2f}")
        
        print(f"\n🎉 平行辯論示例完成！")
        
    except Exception as e:
        print(f"❌ 示例執行失敗: {e}")
        import traceback
        traceback.print_exc()

def demo_architecture_explanation():
    """解釋系統架構"""
    
    print("\n🏗️ 系統架構說明")
    print("=" * 60)
    
    print("""
📋 平行處理流程:

1. 🔄 每個回合開始時，所有 Agent 同時進行分析:
   ├── RL 策略選擇 (選擇辯論策略)
   ├── GNN 社會分析 (分析社會關係和影響力)  
   └── RAG 證據檢索 (搜尋相關證據)

2. 🔀 融合分析結果:
   ├── 根據社會影響力調整策略
   ├── 結合證據品質選擇最佳片段
   └── 生成個性化的辯論提示

3. 🤖 GPT-4o 生成回覆:
   ├── 整合所有分析結果
   ├── 考慮目標對手的弱點
   └── 生成策略性回覆

4. 📊 效果評估和狀態更新:
   ├── 評估說服力和攻擊性
   ├── 更新所有 Agent 的立場和信念
   └── 記錄歷史以供下回合參考

🎯 動態說服/反駁機制:

• 說服機制:
  - 高說服力回覆 → 降低對手信念度 → 立場趨向中性
  - 信念度低的 Agent 更容易被說服
  - 溫和的策略更容易產生說服效果

• 反駁機制:
  - 高攻擊性回覆 → 增強對手信念度 → 立場更極端
  - 信念度高的 Agent 更能抵抗攻擊
  - 激進的策略容易引發反彈

• 適應性學習:
  - Agent 根據歷史經驗調整策略
  - 社會背景影響策略選擇
  - 立場和信念動態變化影響後續行為
""")

async def main():
    """主函數"""
    
    # 解釋架構
    demo_architecture_explanation()
    
    # 執行示例
    await demo_parallel_debate()

if __name__ == "__main__":
    # 運行異步主函數
    asyncio.run(main()) 