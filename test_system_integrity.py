"""
系統完整性測試腳本
檢查所有模組是否正常運作
"""

import sys
from pathlib import Path

# 添加專案路徑
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """測試所有重要模組的導入"""
    print("🔍 測試模組導入...")
    
    modules_to_test = [
        ("RL 模組", "rl.policy_network", "PolicyNetwork"),
        ("GNN 模組", "gnn.social_encoder", "social_vec"),
        ("RAG 模組", "rag.retriever", "create_enhanced_retriever"),
        ("GPT 介面", "gpt_interface.gpt_client", "chat"),
        ("配置載入器", "utils.config_loader", "ConfigLoader"),
        ("協調器", "orchestrator.parallel_orchestrator", "ParallelOrchestrator"),
        ("對話管理器", "dialogue.dialogue_manager", "DialogueManager"),
    ]
    
    results = []
    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                results.append((name, "✅ 成功"))
            else:
                results.append((name, f"❌ 找不到 {class_name}"))
        except Exception as e:
            results.append((name, f"❌ 錯誤: {str(e)}"))
    
    return results

def test_config_files():
    """測試配置檔案"""
    print("\n🔍 測試配置檔案...")
    
    config_files = [
        "configs/debate.yaml",
        "configs/rag.yaml",
        "configs/gnn.yaml",
        "configs/rl.yaml"
    ]
    
    results = []
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                results.append((config_file, f"✅ 存在 ({size} bytes)"))
            else:
                results.append((config_file, "⚠️ 檔案為空"))
        else:
            results.append((config_file, "❌ 不存在"))
    
    return results

def test_data_files():
    """測試資料檔案"""
    print("\n🔍 測試資料檔案...")
    
    data_files = [
        ("RAG 簡單索引", "src/rag/data/rag/simple_index.json"),
        ("RL 模型", "data/models/policy/pytorch_model.bin"),
        ("GNN 模型", "data/models/gnn_social.pt"),
    ]
    
    results = []
    for name, file_path in data_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            results.append((name, f"✅ 存在 ({size:.1f} KB)"))
        else:
            results.append((name, "⚠️ 不存在（需要訓練）"))
    
    return results

def test_ui_files():
    """測試 UI 檔案"""
    print("\n🔍 測試 UI 檔案...")
    
    ui_files = [
        "ui/app.py",
        "ui/templates/index.html",
        "ui/static/css/style.css",
        "ui/static/js/app.js"
    ]
    
    results = []
    for ui_file in ui_files:
        path = Path(ui_file)
        if path.exists():
            results.append((ui_file, "✅ 存在"))
        else:
            results.append((ui_file, "❌ 不存在"))
    
    return results

def print_results(title, results):
    """打印測試結果"""
    print(f"\n{title}")
    print("-" * 50)
    for name, status in results:
        print(f"{name:<30} {status}")

def main():
    """主函數"""
    print("=" * 50)
    print("🔧 Social Debate AI 系統完整性測試")
    print("=" * 50)
    
    # 執行測試
    import_results = test_imports()
    config_results = test_config_files()
    data_results = test_data_files()
    ui_results = test_ui_files()
    
    # 打印結果
    print_results("模組導入測試", import_results)
    print_results("配置檔案測試", config_results)
    print_results("資料檔案測試", data_results)
    print_results("UI 檔案測試", ui_results)
    
    # 統計總結
    print("\n" + "=" * 50)
    print("📊 測試總結")
    print("=" * 50)
    
    all_results = import_results + config_results + data_results + ui_results
    success_count = sum(1 for _, status in all_results if "✅" in status)
    warning_count = sum(1 for _, status in all_results if "⚠️" in status)
    error_count = sum(1 for _, status in all_results if "❌" in status)
    
    print(f"✅ 成功: {success_count}")
    print(f"⚠️ 警告: {warning_count}")
    print(f"❌ 錯誤: {error_count}")
    print(f"總計: {len(all_results)} 項測試")
    
    # 建議
    if error_count > 0:
        print("\n💡 建議:")
        print("1. 請確保已安裝所有依賴: pip install -r requirements.txt")
        print("2. 如果模型檔案不存在，請執行訓練腳本")
        print("3. 檢查環境變數設定（OPENAI_API_KEY）")
    
    return error_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 