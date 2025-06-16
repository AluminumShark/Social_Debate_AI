#!/usr/bin/env python3
"""測試整個 RAG + Orchestrator 系統"""

import os
from pathlib import Path

def test_api_key():
    """測試 API key 設置"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ API Key 已設置: {api_key[:8]}...{api_key[-4:]}")
        return True
    else:
        print("❌ 請設置 OPENAI_API_KEY 環境變數")
        return False

def test_index_exists():
    """測試索引是否存在"""
    index_path = Path("data/index/pairs_high")
    if index_path.exists():
        print(f"✅ 索引目錄存在: {index_path}")
        return True
    else:
        print(f"❌ 索引目錄不存在: {index_path}")
        print("💡 請先運行: python src/rag/build_index.py")
        return False

def test_retriever():
    """測試檢索器"""
    try:
        from src.rag.retriever import PairsRetriever
        retriever = PairsRetriever()
        
        # 測試檢索
        results = retriever.retrieve("universal basic income", k=3)
        print(f"✅ 檢索測試成功，找到 {len(results)} 個結果")
        
        if results:
            print("📄 檢索結果預覽:")
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 檢索器測試失敗: {e}")
        return False

def test_orchestrator():
    """測試協調器"""
    try:
        from src.orchestrator.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        
        # 測試回應生成（不需要實際調用 GPT API）
        topic = "Universal Basic Income should be implemented"
        history = ["I think UBI would reduce work incentives"]
        
        print("✅ 協調器初始化成功")
        print("💡 如果有 GPT API，可以測試 get_reply 方法")
        
        # 測試證據收集
        evidence = orchestrator.gather_evidence("basic income work incentives")
        print(f"✅ 證據收集測試成功，找到 {len(evidence)} 條證據")
        
        return True
        
    except Exception as e:
        print(f"❌ 協調器測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔧 開始系統測試...")
    print("=" * 50)
    
    tests = [
        ("API Key 檢查", test_api_key),
        ("索引存在檢查", test_index_exists),
        ("檢索器測試", test_retriever),
        ("協調器測試", test_orchestrator)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 {name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ❌ {name} 未通過")
        except Exception as e:
            print(f"   ❌ {name} 出錯: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統運行正常")
    else:
        print("⚠️  部分測試未通過，請檢查上述錯誤")

if __name__ == "__main__":
    main() 