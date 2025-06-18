"""
簡化版檢索器
當 Chroma 向量資料庫不可用時使用
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """檢索結果"""
    content: str
    score: float
    metadata: Dict
    doc_id: str

class SimpleRetriever:
    """簡化版檢索器，使用 JSON 索引"""
    
    def __init__(self, index_path: str = "src/rag/data/rag/simple_index.json"):
        self.index_path = Path(index_path)
        self.documents = []
        self._load_index()
    
    def _load_index(self):
        """載入 JSON 索引"""
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
                print(f"✅ 載入簡單索引: {len(self.documents)} 個文檔")
        else:
            print(f"⚠️ 找不到索引檔案: {self.index_path}")
            self.documents = []
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        簡單的關鍵字檢索
        
        Args:
            query: 查詢字串
            top_k: 返回結果數量
            
        Returns:
            檢索結果列表
        """
        if not self.documents:
            print("⚠️ 沒有可用的文檔")
            return []
        
        # 簡單的關鍵字匹配評分
        query_terms = query.lower().split()
        results = []
        
        for doc in self.documents:
            content = doc.get('content', '').lower()
            metadata = doc.get('metadata', {})
            
            # 計算匹配分數
            score = 0.0
            for term in query_terms:
                if term in content:
                    score += 1.0
            
            # 考慮文檔品質分數
            quality_score = metadata.get('quality_score', 0.5)
            final_score = (score / len(query_terms)) * 0.7 + quality_score * 0.3
            
            results.append(RetrievalResult(
                content=doc.get('content', ''),
                score=final_score,
                metadata=metadata,
                doc_id=doc.get('id', '')
            ))
        
        # 按分數排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前 k 個結果
        top_results = results[:top_k]
        print(f"🔍 檢索到 {len(top_results)} 個結果")
        
        return top_results
    
    def retrieve_by_topic(self, topic: str, top_k: int = 5) -> List[RetrievalResult]:
        """按主題檢索"""
        results = []
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            doc_topic = metadata.get('topic', '')
            
            if topic.lower() in doc_topic.lower():
                results.append(RetrievalResult(
                    content=doc.get('content', ''),
                    score=metadata.get('quality_score', 0.5),
                    metadata=metadata,
                    doc_id=doc.get('id', '')
                ))
        
        # 按品質分數排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """獲取統計資訊"""
        topics = {}
        types = {}
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            
            # 統計主題
            topic = metadata.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
            
            # 統計類型
            doc_type = metadata.get('type', 'unknown')
            types[doc_type] = types.get(doc_type, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'topics': topics,
            'types': types,
            'index_path': str(self.index_path)
        } 