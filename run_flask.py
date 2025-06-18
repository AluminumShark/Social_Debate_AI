#!/usr/bin/env python
"""
Flask 版本的 Social Debate AI 啟動腳本
"""

import sys
import os

# 確保可以導入 app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    try:
        from ui.app import app, init_system
        
        print("=" * 50)
        print("🚀 Social Debate AI - Flask 版本")
        print("=" * 50)
        
        # 初始化系統
        if init_system():
            print("\n✅ 系統初始化成功！")
            print(f"\n🌐 在瀏覽器中打開: http://localhost:5000")
            print("=" * 50)
            
            # 啟動 Flask（關閉自動重載以避免 socket 錯誤）
            app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        else:
            print("\n❌ 系統初始化失敗，請檢查配置")
            
    except Exception as e:
        print(f"\n❌ 啟動失敗: {e}")
        sys.exit(1) 