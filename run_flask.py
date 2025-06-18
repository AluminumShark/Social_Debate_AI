#!/usr/bin/env python
"""
Flask 版本的 Social Debate AI 啟動腳本
"""

import sys
import os
import traceback
from datetime import datetime

# 確保可以導入 app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """印出啟動橫幅"""
    print("\n" + "=" * 60)
    print("🚀 Social Debate AI - Flask Web Interface")
    print("=" * 60)
    print(f"📅 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

def print_step(step_name, status="開始"):
    """印出步驟狀態"""
    icons = {
        "開始": "⏳",
        "成功": "✅",
        "失敗": "❌",
        "警告": "⚠️"
    }
    icon = icons.get(status, "📌")
    print(f"{icon} {step_name}...")

if __name__ == '__main__':
    try:
        print_banner()
        
        # 步驟 1: 導入模組
        print_step("導入 Flask 應用模組", "開始")
        from ui.app import app, init_system
        print_step("導入 Flask 應用模組", "成功")
        
        # 步驟 2: 初始化系統
        print("\n" + "-" * 40)
        print("📦 系統初始化")
        print("-" * 40)
        
        init_result = init_system()
        
        if init_result:
            print("\n" + "-" * 40)
            print("✅ 所有系統元件初始化完成！")
            print("-" * 40)
            
            # 顯示服務資訊
            print("\n📡 Flask 服務資訊:")
            print(f"   - 本地訪問: http://localhost:5000")
            print(f"   - 網路訪問: http://0.0.0.0:5000")
            print(f"   - 調試模式: 開啟")
            print(f"   - 自動重載: 關閉")
            
            print("\n💡 使用提示:")
            print("   - 在瀏覽器中打開上述網址")
            print("   - 按 Ctrl+C 停止服務")
            print("\n" + "=" * 60 + "\n")
            
            # 啟動 Flask（關閉自動重載以避免 socket 錯誤）
            app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        else:
            print("\n❌ 系統初始化失敗")
            print("請檢查以下項目:")
            print("   1. 環境變數是否正確設置 (OPENAI_API_KEY)")
            print("   2. 所需的模型檔案是否存在")
            print("   3. 配置檔案是否正確")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  收到中斷信號，正在關閉服務...")
        print("👋 感謝使用 Social Debate AI！\n")
    except Exception as e:
        print(f"\n❌ 啟動失敗: {e}")
        print("\n詳細錯誤信息:")
        traceback.print_exc()
        sys.exit(1) 