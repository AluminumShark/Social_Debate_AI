import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

try:
    from cmv_dataset import get_pyg_data
except ImportError:
    import sys
    sys.path.append('src')
    from gnn.cmv_dataset import get_pyg_data

CKPT = "data/models/gnn_social.pt"

def evaluate_classification(emb, y, test_name="度數分類"):
    """評估分類任務"""
    print(f"\n=== {test_name} ===")
    
    n = len(y)
    np.random.seed(42)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    
    # 測試多種分類器
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        clf.fit(emb[train_idx], y[train_idx])
        
        prob = clf.predict_proba(emb[test_idx])[:, 1]
        pred = (prob > 0.5).astype(int)
        
        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], prob)
        
        results[name] = {'accuracy': acc, 'auc': auc}
        print(f"準確率: {acc:.4f}, AUC: {auc:.4f}")
        
        # 交叉驗證
        cv_scores = cross_val_score(clf, emb, y, cv=5, scoring='accuracy')
        print(f"5折交叉驗證: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_clustering(emb, n_clusters=5):
    """評估聚類任務"""
    print(f"\n=== 聚類評估 ===")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(emb)
    
    print(f"聚類數量: {n_clusters}")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        print(f"聚類 {i}: {count} 個節點 ({count/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels

def visualize_embeddings(emb, labels=None, save_path="embedding_visualization.png"):
    """可視化嵌入向量"""
    print(f"\n=== 可視化嵌入向量 ===")
    
    # 使用 t-SNE 降維到 2D
    print("執行 t-SNE 降維...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(emb[:5000])  # 只可視化前5000個點以提高速度
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        labels_subset = labels[:5000]
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_subset, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('嵌入向量 t-SNE 可視化（有標籤）')
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6)
        plt.title('嵌入向量 t-SNE 可視化')
    
    plt.xlabel('t-SNE 維度 1')
    plt.ylabel('t-SNE 維度 2')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可視化圖表已儲存至: {save_path}")
    
    return emb_2d

def analyze_embedding_properties(emb):
    """分析嵌入向量的屬性"""
    print(f"\n=== 嵌入向量屬性分析 ===")
    
    print(f"嵌入維度: {emb.shape}")
    print(f"平均值: {np.mean(emb):.4f}")
    print(f"標準差: {np.std(emb):.4f}")
    print(f"最小值: {np.min(emb):.4f}")
    print(f"最大值: {np.max(emb):.4f}")
    
    # 計算相似性分佈
    from sklearn.metrics.pairwise import cosine_similarity
    print("計算餘弦相似性分佈...")
    sample_size = min(1000, emb.shape[0])
    sample_idx = np.random.choice(emb.shape[0], sample_size, replace=False)
    sample_emb = emb[sample_idx]
    
    cos_sim = cosine_similarity(sample_emb)
    cos_sim_flat = cos_sim[np.triu_indices_from(cos_sim, k=1)]
    
    print(f"餘弦相似性統計 (樣本: {sample_size}):")
    print(f"  平均: {np.mean(cos_sim_flat):.4f}")
    print(f"  標準差: {np.std(cos_sim_flat):.4f}")
    print(f"  中位數: {np.median(cos_sim_flat):.4f}")

def main():
    parser = argparse.ArgumentParser(description='進階嵌入向量評估')
    parser.add_argument('--no-viz', action='store_true', help='跳過可視化')
    parser.add_argument('--clusters', type=int, default=5, help='聚類數量')
    args = parser.parse_args()
    
    print("=== 進階嵌入向量評估 ===")
    
    try:
        # 1. 載入資料和模型
        print("載入資料...")
        data, G = get_pyg_data()
        print(f"資料載入成功: {data.x.size(0)} 個節點")
        
        print("載入模型...")
        ckpt = torch.load(CKPT, map_location='cpu')
        emb = ckpt['emb'].numpy()
        print(f"嵌入向量載入成功: {emb.shape}")
        
        # 2. 分析嵌入向量屬性
        analyze_embedding_properties(emb)
        
        # 3. 創建多種標籤進行評估
        from torch_geometric.utils import degree
        node_degrees = degree(data.edge_index[0], num_nodes=data.x.size(0)).numpy()
        
        # 基於度數的二元分類
        median_degree = np.median(node_degrees)
        degree_labels = (node_degrees > median_degree).astype(int)
        
        # 基於成功率的分類（如果有的話）
        success_rates = []
        for _, d in G.nodes(data=True):
            success_rates.append(d.get('success_rate', 0.0))
        success_rates = np.array(success_rates)
        success_labels = (success_rates > np.median(success_rates)).astype(int)
        
        # 基於回覆數的分類
        replies = []
        for _, d in G.nodes(data=True):
            replies.append(d.get('replies', 0))
        replies = np.array(replies)
        reply_labels = (replies > np.median(replies)).astype(int)
        
        # 4. 評估分類任務
        evaluate_classification(emb, degree_labels, "節點度數分類")
        evaluate_classification(emb, success_labels, "成功率分類")
        evaluate_classification(emb, reply_labels, "回覆數分類")
        
        # 5. 聚類評估
        cluster_labels = evaluate_clustering(emb, args.clusters)
        
        # 6. 可視化（可選）
        if not args.no_viz:
            try:
                visualize_embeddings(emb, degree_labels, "embedding_viz_degree.png")
                visualize_embeddings(emb, cluster_labels, "embedding_viz_clusters.png")
            except ImportError:
                print("跳過可視化 (需要 matplotlib)")
        
        print(f"\n=== 評估完成 ===")
        
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 