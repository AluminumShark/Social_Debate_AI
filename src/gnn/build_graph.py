import json, math, random, tiktoken
import networkx as nx
from pathlib import Path
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")

def tok_len(text):
    return len(enc.encode(text))

def build_graph(pairs_path='data/raw/pairs.jsonl'):
    G = nx.Graph()
    pairs_path = Path(pairs_path)
    with open(str(pairs_path), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Scan Pairs'):
            p = json.loads(line)
            op = p['submission']['title']
            
            # 確保原作者節點存在
            if op not in G.nodes:
                G.add_node(op)

            for key, w in (('delta_comment', 3), ('nodelta_comment', 1)):
                c = p.get(key) or {}
                if not c:
                    continue
                author = c['author']
                body = c.get('body', '')
                score = c.get('score', 0)

                # 更新 node basic stats
                if author not in G.nodes:
                    G.add_node(author)
                n = G.nodes[author]
                n['tok_lens'] = n.get('tok_lens', []) + [tok_len(body)]
                n['score_list'] = n.get('score_list', []) + [score]

                if key == 'delta_comment':
                    n['success'] = n.get('success', 0) + 1

                # 增加 Edge
                if author != op:
                    G.add_edge(
                        op,
                        author,
                        weight=G[op][author].get('weight', 0) if G.has_edge(op, author) else w
                    )
    
    # 將 List 聚合成統計，確保所有節點都有相同的屬性
    for node, d in G.nodes(data=True):
        lens = d.get('tok_lens', [])
        scores = d.get('score_list', [])
        
        # 設定預設值，防止除零錯誤
        if len(lens) > 0:
            d['average_body_len'] = sum(lens) / len(lens)
            d['std_body_len'] = math.sqrt(sum((l - d['average_body_len']) ** 2 for l in lens) / len(lens))
        else:
            d['average_body_len'] = 0.0
            d['std_body_len'] = 0.0
            
        if len(scores) > 0:
            d['avg_score'] = sum(scores) / len(scores)
        else:
            d['avg_score'] = 0.0
            
        d['success_count'] = d.get('success', 0)
        d['replies'] = len(lens)
        d['success_rate'] = d['success_count'] / d['replies'] if d['replies'] > 0 else 0.0
        
        # 清理臨時屬性，避免 from_networkx 的屬性不一致問題
        d.pop('tok_lens', None)
        d.pop('score_list', None)
        d.pop('success', None)

    return G