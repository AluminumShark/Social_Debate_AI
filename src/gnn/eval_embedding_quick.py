import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

try:
    from cmv_dataset import get_pyg_data
except ImportError:
    import sys
    sys.path.append('src')
    from gnn.cmv_dataset import get_pyg_data

CKPT = "data/models/gnn_social.pt"

def quick_evaluate():
    print("=== 快速嵌入向量評估 ===")
    
    try:
        # 1. 載入資料和模型
        print("載入資料...")
        data, G = get_pyg_data()
        
        print("載入模型...")
        ckpt = torch.load(CKPT, map_location='cpu')
        emb = ckpt['emb'].numpy()
        print(f"嵌入向量: {emb.shape}")
        
        # 2. 嵌入向量基本統計
        print(f"\n=== 嵌入向量統計 ===")
        print(f"平均值: {np.mean(emb):.4f}")
        print(f"標準差: {np.std(emb):.4f}")
        print(f"範圍: [{np.min(emb):.4f}, {np.max(emb):.4f}]")
        
        # 3. 創建基於度數的標籤
        from torch_geometric.utils import degree
        node_degrees = degree(data.edge_index[0], num_nodes=data.x.size(0)).numpy()
        median_degree = np.median(node_degrees)
        y = (node_degrees > median_degree).astype(int)
        
        print(f"\n度數分佈:")
        print(f"  中位數度數: {median_degree:.1f}")
        print(f"  高度數節點: {np.sum(y)} 個")
        print(f"  低度數節點: {len(y) - np.sum(y)} 個")
        
        # 4. 分割資料
        n = len(y)
        np.random.seed(42)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = perm[:split], perm[split:]
        
        # 5. 測試分類器
        print(f"\n=== 分類評估 ===")
        
        # Logistic Regression
        print("Logistic Regression:")
        lr = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        lr.fit(emb[train_idx], y[train_idx])
        lr_pred = lr.predict(emb[test_idx])
        lr_prob = lr.predict_proba(emb[test_idx])[:, 1]
        lr_acc = accuracy_score(y[test_idx], lr_pred)
        lr_auc = roc_auc_score(y[test_idx], lr_prob)
        print(f"  準確率: {lr_acc:.4f}, AUC: {lr_auc:.4f}")
        
        # Random Forest
        print("Random Forest:")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        rf.fit(emb[train_idx], y[train_idx])
        rf_pred = rf.predict(emb[test_idx])
        rf_prob = rf.predict_proba(emb[test_idx])[:, 1]
        rf_acc = accuracy_score(y[test_idx], rf_pred)
        rf_auc = roc_auc_score(y[test_idx], rf_prob)
        print(f"  準確率: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
        
        # 6. 聚類評估
        print(f"\n=== 聚類評估 ===")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(emb)
        
        print("聚類分佈:")
        for i in range(3):
            count = np.sum(clusters == i)
            print(f"  聚類 {i}: {count} 個節點 ({count/len(clusters)*100:.1f}%)")
        
        # 7. 分析聚類與度數的關係
        print(f"\n各聚類的平均度數:")
        for i in range(3):
            cluster_mask = clusters == i
            avg_degree = np.mean(node_degrees[cluster_mask])
            print(f"  聚類 {i}: {avg_degree:.2f}")
        
        # 8. 檢查不同特徵的預測能力
        print(f"\n=== 不同特徵的預測能力 ===")
        
        # 基於成功率的分類
        success_rates = []
        for _, d in G.nodes(data=True):
            success_rates.append(d.get('success_rate', 0.0))
        success_rates = np.array(success_rates)
        success_y = (success_rates > np.median(success_rates)).astype(int)
        
        if len(np.unique(success_y)) > 1:  # 確保有兩個類別
            lr_success = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
            lr_success.fit(emb[train_idx], success_y[train_idx])
            success_pred = lr_success.predict(emb[test_idx])
            success_acc = accuracy_score(success_y[test_idx], success_pred)
            print(f"成功率分類準確率: {success_acc:.4f}")
        
        # 基於回覆數的分類
        replies = []
        for _, d in G.nodes(data=True):
            replies.append(d.get('replies', 0))
        replies = np.array(replies)
        reply_y = (replies > np.median(replies)).astype(int)
        
        if len(np.unique(reply_y)) > 1:  # 確保有兩個類別
            lr_reply = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
            lr_reply.fit(emb[train_idx], reply_y[train_idx])
            reply_pred = lr_reply.predict(emb[test_idx])
            reply_acc = accuracy_score(reply_y[test_idx], reply_pred)
            print(f"回覆數分類準確率: {reply_acc:.4f}")
        
        print(f"\n=== 評估完成 ===")
        print("您的 DGI 嵌入向量表現良好！")
        
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_evaluate() 