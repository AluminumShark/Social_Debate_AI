"""
Enhanced Multi-layered Retriever
Supports topic filtering, complexity screening, argument strength ranking
"""

from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import numpy as np

class EnhancedRetriever:
    """Enhanced Multi-layered Retriever"""
    
    def __init__(self):
        self.base_dir = Path('data/index/enhanced')
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self.stores = {}
        self._load_stores()
    
    def _load_stores(self):
        """Load all available vector stores"""
        configs = {
            'high_quality': {
                'path': self.base_dir / 'high_quality',
                'collection': 'hq_pairs'
            },
            'comprehensive': {
                'path': self.base_dir / 'comprehensive', 
                'collection': 'all_discussions'
            }
        }
        
        for name, config in configs.items():
            if config['path'].exists():
                try:
                    self.stores[name] = Chroma(
                        persist_directory=str(config['path']),
                        embedding_function=self.embeddings,
                        collection_name=config['collection']
                    )
                    print(f"✅ Loaded {name} index")
                except Exception as e:
                    print(f"⚠️ Failed to load {name} index: {e}")
        
        if not self.stores:
            raise RuntimeError("❌ No available indexes found")
    
    def retrieve(self, 
                 query: str, 
                 k: int = 5,
                 index_type: str = 'high_quality',
                 topics: Optional[List[str]] = None,
                 complexity: Optional[str] = None,
                 min_score: int = 0,
                 persuasion_only: bool = False) -> List[Dict]:
        """
        Enhanced retrieval functionality
        
        Args:
            query: Query string
            k: Number of results to return
            index_type: Index type ('high_quality', 'comprehensive')
            topics: Topic filter (['politics', 'economics'])
            complexity: Complexity filter ('simple', 'intermediate', 'complex')
            min_score: Minimum score filter
            persuasion_only: Whether to return only successful persuasion cases
        """
        
        if index_type not in self.stores:
            print(f"⚠️ Index {index_type} not found, using default index")
            index_type = list(self.stores.keys())[0]
        
        store = self.stores[index_type]
        
        try:
            # Basic retrieval
            docs = store.similarity_search_with_score(query, k=k*2)  # Retrieve more for filtering
            
            results = []
            seen_content = set()
            
            for doc, score in docs:
                content = doc.page_content.strip()
                if content in seen_content:
                    continue
                seen_content.add(content)
                
                metadata = doc.metadata
                
                # Apply filtering conditions
                if self._should_filter(metadata, topics, complexity, min_score, persuasion_only):
                    continue
                
                # Parse topics from string format
                topics_str = metadata.get('topics', '')
                topics_list = topics_str.split(',') if topics_str else []
                
                results.append({
                    'content': content,
                    'metadata': metadata,
                    'similarity_score': float(1 - score),  # Convert to similarity score
                    'type': metadata.get('type', 'unknown'),
                    'topics': topics_list,  # Convert back to list for compatibility
                    'complexity': metadata.get('complexity', 'unknown'),
                    'score': metadata.get('score', 0),
                    'title': metadata.get('title', '')
                })
                
                if len(results) >= k:
                    break
            
            # Sort by relevance and quality
            results.sort(key=lambda x: (x['similarity_score'], x['score']), reverse=True)
            
            print(f"🔍 Retrieved {len(results)} results from {index_type}")
            return results[:k]
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []
    
    def _should_filter(self, metadata, topics, complexity, min_score, persuasion_only):
        """Determine if this document should be filtered"""
        
        # Topic filtering
        if topics:
            doc_topics_str = metadata.get('topics', '')
            doc_topics = doc_topics_str.split(',') if doc_topics_str else []
            if not any(topic in doc_topics for topic in topics):
                return True
        
        # Complexity filtering
        if complexity and metadata.get('complexity') != complexity:
            return True
        
        # Score filtering
        if metadata.get('score', 0) < min_score:
            return True
        
        # Persuasion success filtering
        if persuasion_only and not metadata.get('persuasion_success', False):
            return True
        
        return False
    
    def get_topic_distribution(self, query: str, index_type: str = 'high_quality') -> Dict[str, int]:
        """Get topic distribution of query results"""
        
        if index_type not in self.stores:
            return {}
        
        store = self.stores[index_type]
        docs = store.similarity_search(query, k=50)  # Get more results for statistics
        
        topic_count = {}
        for doc in docs:
            topics_str = doc.metadata.get('topics', '')
            topics = topics_str.split(',') if topics_str else []
            for topic in topics:
                topic = topic.strip()  # Remove any whitespace
                if topic:  # Only count non-empty topics
                    topic_count[topic] = topic_count.get(topic, 0) + 1
        
        return dict(sorted(topic_count.items(), key=lambda x: x[1], reverse=True))
    
    def retrieve_by_topic(self, topic: str, k: int = 10, index_type: str = 'high_quality') -> List[Dict]:
        """Retrieve by topic"""
        return self.retrieve(
            query=f"discussion about {topic}",
            k=k,
            index_type=index_type,
            topics=[topic]
        )
    
    def retrieve_successful_arguments(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve successful argument cases"""
        return self.retrieve(
            query=query,
            k=k,
            index_type='high_quality',
            persuasion_only=True
        )
    
    def retrieve_diverse_perspectives(self, query: str, k: int = 10) -> List[Dict]:
        """Retrieve diverse perspectives"""
        # Get results from different indexes
        results = []
        
        if 'high_quality' in self.stores:
            hq_results = self.retrieve(query, k=k//2, index_type='high_quality')
            results.extend(hq_results)
        
        if 'comprehensive' in self.stores:
            comp_results = self.retrieve(query, k=k//2, index_type='comprehensive')
            results.extend(comp_results)
        
        # Deduplicate and sort by similarity
        seen = set()
        unique_results = []
        for r in results:
            if r['content'] not in seen:
                seen.add(r['content'])
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return unique_results[:k]
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        stats = {
            'available_indexes': list(self.stores.keys()),
            'total_indexes': len(self.stores)
        }
        
        for name, store in self.stores.items():
            try:
                # Get collection info (may need adjustment based on Chroma version)
                collection = store._collection
                stats[f'{name}_count'] = collection.count() if hasattr(collection, 'count') else 'unknown'
            except:
                stats[f'{name}_count'] = 'unknown'
        
        return stats

# Convenience function
def create_enhanced_retriever():
    """Create enhanced retriever instance"""
    try:
        # 檢查是否有 Chroma 索引
        chroma_path = Path('data/chroma/social_debate')
        if chroma_path.exists():
            print("🔍 嘗試載入 Chroma 向量索引...")
            return EnhancedRetriever()
        else:
            print("⚠️ Chroma 索引不存在，使用簡單檢索器")
            from .simple_retriever import SimpleRetriever
            return SimpleRetrieverAdapter()
    except Exception as e:
        print(f"❌ Failed to create retriever: {e}")
        print("💡 使用簡化版檢索器")
        from .simple_retriever import SimpleRetriever
        return SimpleRetrieverAdapter()

class SimpleRetrieverAdapter:
    """
    適配器類，將 SimpleRetriever 的輸出格式轉換為與 EnhancedRetriever 兼容
    """
    def __init__(self):
        from .simple_retriever import SimpleRetriever
        self.simple_retriever = SimpleRetriever()
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """檢索並轉換格式"""
        results = self.simple_retriever.retrieve(query, top_k)
        
        # 轉換為字典格式
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.content,
                'score': result.score,
                'metadata': result.metadata,
                'doc_id': result.doc_id,
                'similarity_score': result.score
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """獲取統計資訊"""
        return self.simple_retriever.get_stats()
