import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
# from torch_geometric.loader import NeighborLoader  # 註解掉，使用完整圖訓練
from cmv_dataset import get_pyg_data, NUM_FEATURES
from pathlib import Path
from tqdm import tqdm

class SAGE(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.conv1 = tgnn.SAGEConv(in_dim, hid)
        self.conv2 = tgnn.SAGEConv(hid, hid)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu_()
        return self.conv2(x, edge_index)
    
def train_dgi(epochs=30,
                hid=128,
                batch_size=4096,
                fanout=[15, 10],
                out="data/models/gnn_social.pt"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    out = Path(out)

    # 0. 設定隨機種子
    torch.manual_seed(517466)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(517466)

    # 1. 載入圖資料
    print("載入圖資料...")
    data, G = get_pyg_data()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    print(f"圖節點數: {data.x.size(0)}, 邊數: {data.edge_index.size(1)}")

    # 2. 定義模型
    print("初始化模型...")
    encoder = SAGE(NUM_FEATURES, hid).to(device)
    
    # 定義損壞函數（corruption function）
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index
    
    dgi = tgnn.DeepGraphInfomax(
        hidden_channels=hid,
        encoder=encoder,
        summary=lambda z, *k: z.mean(dim=0),
        corruption=corruption
    ).to(device)
    
    optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3)

    print(f"開始訓練 {epochs} 個 epochs（使用完整圖）...")
    
    # 3. 訓練迴圈（使用完整圖，不使用小批次）
    for epoch in tqdm(range(epochs), desc="訓練進度"):
        dgi.train()
        
        # 前向傳播
        optimizer.zero_grad()
        pos_z, neg_z, summary = dgi(data.x, data.edge_index)
        loss = dgi.loss(pos_z, neg_z, summary)
        
        # 反向傳播和優化
        loss.backward()
        optimizer.step()
        
        # 每 5 個 epoch 顯示一次損失
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs}, 損失: {loss.item():.4f}")
    
    print(f"訓練完成！最終損失: {loss.item():.4f}")

    # 5. 計算完整圖的嵌入向量
    print("計算完整圖的嵌入向量...")
    dgi.eval()
    with torch.no_grad():
        emb = dgi.encoder(data.x, data.edge_index).cpu()

    # 6. 儲存模型和嵌入向量
    print(f"儲存模型到 {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': dgi.encoder.state_dict(),
        'emb': emb,
        'node2idx': {n: i for i, n in enumerate(G.nodes)},
        'config': {
            'hidden_dim': hid,
            'num_features': NUM_FEATURES,
            'fanout': fanout
        }
    }, str(out))
    
    print("訓練完成！")

if __name__ == "__main__":
    print("=== 開始執行 DGI 訓練主程式 ===")
    try:
        train_dgi()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    print("=== 主程式執行結束 ===")