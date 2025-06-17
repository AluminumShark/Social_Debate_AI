import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

try:
    from cmv_dataset import get_pyg_data
except ImportError:
    try:
        import sys
        sys.path.append('src')
        from gnn.cmv_dataset import get_pyg_data
    except ImportError:
        print("無法導入 cmv_dataset")
        sys.exit(1)

CKPT = "data/models/gnn_social.pt"

def evaluate():
    print("=== 開始評估嵌入向量 ===")
    
    try:
        # 1. 載入資料和模型
        print("載入資料...")
        data, G = get_pyg_data()
        print(f"資料載入成功: {data.x.size(0)} 個節點")
        
        print("載入模型...")
        ckpt = torch.load(CKPT, map_location='cpu')
        emb = ckpt['emb'].numpy()
        print(f"嵌入向量載入成功: {emb.shape}")
        
        # 2. 檢查是否有標籤資料
        if hasattr(data, 'y') and data.y is not None:
            print("使用現有的節點標籤...")
            y = data.y.numpy()
        else:
            print("沒有節點標籤，創建基於圖結構的偽標籤...")
            # 基於節點度數創建二元標籤（高度數 vs 低度數）
            from torch_geometric.utils import degree
            node_degrees = degree(data.edge_index[0], num_nodes=data.x.size(0)).numpy()
            median_degree = np.median(node_degrees)
            y = (node_degrees > median_degree).astype(int)
            print(f"創建偽標籤: {np.sum(y)} 個高度數節點, {len(y) - np.sum(y)} 個低度數節點")
        
        # 3. 訓練評估
        print("分割資料...")
        n = len(y)
        np.random.seed(42)  # 確保可重複性
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = perm[:split], perm[split:]
        
        print(f"訓練集: {len(train_idx)} 個樣本, 測試集: {len(test_idx)} 個樣本")
        
        # 4. 訓練分類器
        print("訓練分類器...")
        clf = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear').fit(emb[train_idx], y[train_idx])
        
        # 5. 評估
        print("評估模型...")
        prob = clf.predict_proba(emb[test_idx])[:, 1]
        pred = (prob > 0.5).astype(int)
        
        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], prob)
        
        print(f"\n=== 評估結果 ===")
        print(f"準確率 (Accuracy): {acc:.4f}")
        print(f"AUC 分數: {auc:.4f}")
        
        # 6. 顯示詳細統計資訊
        print(f"\n=== 詳細統計 ===")
        print(f"測試集標籤分佈: 類別0={np.sum(y[test_idx] == 0)}, 類別1={np.sum(y[test_idx] == 1)}")
        print(f"預測分佈: 類別0={np.sum(pred == 0)}, 類別1={np.sum(pred == 1)}")
        
        # 7. 混淆矩陣
        print(f"\n=== 混淆矩陣 ===")
        cm = confusion_matrix(y[test_idx], pred)
        print(f"真負例: {cm[0,0]}, 偽正例: {cm[0,1]}")
        print(f"偽負例: {cm[1,0]}, 真正例: {cm[1,1]}")
        
        # 8. 分類報告
        print(f"\n=== 分類報告 ===")
        print(classification_report(y[test_idx], pred, target_names=['低度數', '高度數']))
        
        # 9. 交叉驗證（可選）
        print(f"\n=== 交叉驗證 ===")
        cv_scores = cross_val_score(clf, emb, y, cv=5, scoring='accuracy')
        print(f"5折交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate()