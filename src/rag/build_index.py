"""
Enhanced Multi-layered Index Building System
Supports topic classification, argument types, persuasion levels and other multi-dimensional indexing
"""

import json, os, time, re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from collections import Counter, defaultdict

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = ""  # Enter your API Key here
if not OPENAI_KEY:
    raise SystemExit("❌ Please set OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# === Parameters ===
PAIRS_PATH = Path("data/raw/pairs.jsonl")
DB_BASE_DIR = Path("data/index/enhanced")
THREADS_PATH = Path("data/raw/threads.jsonl")  # Large dataset

# Index configurations
INDEX_CONFIGS = {
    'high_quality': {
        'path': DB_BASE_DIR / 'high_quality',
        'collection': 'hq_pairs',
        'description': 'High-quality successful persuasion cases'
    },
    'by_topic': {
        'path': DB_BASE_DIR / 'by_topic',
        'collection': 'topic_sorted',
        'description': 'Topic-classified index'
    },
    'comprehensive': {
        'path': DB_BASE_DIR / 'comprehensive',
        'collection': 'all_discussions',
        'description': 'Comprehensive discussion database'
    }
}

CHUNK_SIZE, CHUNK_OVERLAP = 1024, 256
EMB_MODEL = "text-embedding-3-small"
COST_PER_1M_TOKENS = 0.02
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Topic classification keywords
TOPIC_KEYWORDS = {
    'politics': ['government', 'politics', 'political', 'democracy', 'voting', 'election', 'policy'],
    'economics': ['economy', 'economic', 'money', 'income', 'tax', 'wealth', 'capitalism', 'socialism', 'market'],
    'technology': ['technology', 'tech', 'AI', 'artificial intelligence', 'automation', 'digital', 'internet'],
    'social_justice': ['equality', 'rights', 'discrimination', 'racism', 'sexism', 'justice', 'inequality'],
    'education': ['education', 'school', 'university', 'learning', 'teaching', 'student'],
    'healthcare': ['health', 'medical', 'healthcare', 'medicine', 'hospital', 'doctor'],
    'environment': ['environment', 'climate', 'global warming', 'pollution', 'renewable', 'sustainability'],
    'ethics': ['ethics', 'moral', 'morality', 'right', 'wrong', 'should', 'ought'],
    'work': ['work', 'job', 'employment', 'labor', 'career', 'workplace', 'worker'],
    'relationships': ['relationship', 'marriage', 'family', 'dating', 'love', 'friendship'],
    'law': ['law', 'legal', 'court', 'judge', 'crime', 'prison', 'justice system'],
    'religion': ['religion', 'religious', 'god', 'church', 'faith', 'belief', 'spiritual']
}

def classify_topics(text):
    """Classify text topics"""
    text_lower = text.lower()
    topics = []
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics if topics else ['general']

def extract_enhanced_metadata(submission, comment=None):
    """Extract enhanced metadata"""
    title = submission.get('title', '')
    selftext = submission.get('selftext', '')
    full_text = f"{title} {selftext}"
    
    # Basic metadata
    topics_list = classify_topics(full_text)
    metadata = {
        'submission_id': submission.get('id', ''),
        'score': submission.get('score', 0),
        'num_comments': submission.get('num_comments', 0),
        'created_utc': submission.get('created_utc', 0),
        'topics': ','.join(topics_list),  # Convert list to comma-separated string
        'primary_topic': topics_list[0] if topics_list else 'general',  # Store primary topic
        'title': title[:100],  # Truncate title
    }
    
    # Calculate complexity (based on text length and sentence count)
    sentences = len(re.findall(r'[.!?]+', full_text))
    words = len(full_text.split())
    complexity = 'simple' if words < 100 else 'intermediate' if words < 300 else 'complex'
    metadata['complexity'] = complexity
    
    # If there's comment data
    if comment:
        metadata.update({
            'type': 'delta_comment',
            'comment_id': comment.get('id', ''),
            'comment_score': comment.get('score', 0),
            'persuasion_success': True,  # Because it's a delta comment
            'argument_strength': min(comment.get('score', 0) / 10.0, 1.0)  # Normalized strength
        })
    else:
        metadata['type'] = 'submission'
    
    return metadata

class EnhancedEmbeddings(OpenAIEmbeddings):
    """Enhanced embedding class with cost calculation and progress bar"""
    def __init__(self, *args, batch_size=500, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_tokens = 0
        self._total_cost = 0.0
        self._batch_size = batch_size
    
    @property
    def total_tokens(self): return self._total_tokens
    @property 
    def total_cost(self): return self._total_cost
    
    def embed_documents(self, texts):
        print(f"🔧 Vectorizing {len(texts):,} documents...")
        total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
        print(f"📊 Estimated tokens: {total_tokens:,}, cost: ${(total_tokens/1_000_000)*COST_PER_1M_TOKENS:.4f}")
        
        batch_size = self._batch_size
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="🔮 Embedding"):
            batch = texts[i:i + batch_size]
            batch_embeddings = super().embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            batch_tokens = sum(len(tokenizer.encode(text)) for text in batch)
            self._total_tokens += batch_tokens
            self._total_cost += (batch_tokens / 1_000_000) * COST_PER_1M_TOKENS
            time.sleep(0.1)
        
        print(f"✅ Complete! Actual cost: ${self._total_cost:.4f}")
        return embeddings

def build_high_quality_index():
    """Build high-quality index (original functionality)"""
    print("\n🏆 Building high-quality pairs index...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    with PAIRS_PATH.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="📄 Processing pairs data"):
            pair = json.loads(line)
            submission = pair["submission"]
            
            # Process submission
            sub_meta = extract_enhanced_metadata(submission)
            body = submission.get("selftext") or submission.get("title", "")
            if body:
                for chunk in splitter.split_text(body):
                    docs.append(Document(page_content=chunk, metadata=sub_meta))
            
            # Process delta_comment
            delta_comment = pair.get("delta_comment", {})
            if delta_comment and delta_comment.get("body"):
                comment_meta = extract_enhanced_metadata(submission, delta_comment)
                for chunk in splitter.split_text(delta_comment["body"]):
                    docs.append(Document(page_content=chunk, metadata=comment_meta))
    
    print(f"📊 Collected {len(docs):,} high-quality document chunks")
    
    # Build index
    config = INDEX_CONFIGS['high_quality']
    config['path'].mkdir(parents=True, exist_ok=True)
    
    embeddings = EnhancedEmbeddings(model=EMB_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(config['path']),
        collection_name=config['collection']
    )
    vectorstore.persist()
    
    print(f"✅ High-quality index completed: {config['path']}")
    return len(docs), embeddings.total_cost

def build_comprehensive_index(sample_size=50000):
    """Build comprehensive index (using sample from full threads.jsonl)"""
    print(f"\n🌍 Building comprehensive index (sample size: {sample_size:,})...")
    
    if not THREADS_PATH.exists():
        print(f"❌ Cannot find {THREADS_PATH}")
        return 0, 0
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    count = 0
    
    with THREADS_PATH.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="📄 Processing full data", total=sample_size):
            if count >= sample_size:
                break
                
            try:
                thread = json.loads(line)
                
                # Simple quality filtering
                if thread.get('score', 0) < 3:  # Filter low-score posts
                    continue
                
                # Process submission
                meta = extract_enhanced_metadata(thread)
                body = thread.get("selftext") or thread.get("title", "")
                if body and len(body) > 50:  # Filter too short content
                    for chunk in splitter.split_text(body):
                        docs.append(Document(page_content=chunk, metadata=meta))
                
                count += 1
                
            except Exception as e:
                continue
    
    print(f"📊 Collected {len(docs):,} comprehensive document chunks")
    
    # Build index
    config = INDEX_CONFIGS['comprehensive']
    config['path'].mkdir(parents=True, exist_ok=True)
    
    embeddings = EnhancedEmbeddings(model=EMB_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(config['path']),
        collection_name=config['collection']
    )
    vectorstore.persist()
    
    print(f"✅ Comprehensive index completed: {config['path']}")
    return len(docs), embeddings.total_cost

def build_all_indexes():
    """Build all indexes"""
    print("🚀 Starting enhanced multi-layered index system...")
    print("=" * 60)
    
    total_docs = 0
    total_cost = 0.0
    
    # 1. Build high-quality index
    docs_hq, cost_hq = build_high_quality_index()
    total_docs += docs_hq
    total_cost += cost_hq
    
    # 2. Build comprehensive index (if needed)
    if input("\nBuild comprehensive index? (y/N): ").lower() == 'y':
        sample_size = int(input("Sample size (default 50000): ") or 50000)
        docs_comp, cost_comp = build_comprehensive_index(sample_size)
        total_docs += docs_comp
        total_cost += cost_comp
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Index building completed!")
    print(f"📊 Total documents: {total_docs:,}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    
    print("\n📁 Built indexes:")
    for name, config in INDEX_CONFIGS.items():
        if config['path'].exists():
            print(f"  ✅ {name}: {config['description']}")
            print(f"     Path: {config['path']}")
            print(f"     Collection: {config['collection']}")

def build_chroma_index(data_path: str, output_dir: str, max_docs: int = None, batch_size: int = 500):
    """構建 Chroma 向量索引"""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"數據檔案不存在: {data_path}")
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    doc_count = 0
    
    with data_path.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="處理文檔"):
            if max_docs and doc_count >= max_docs:
                break
                
            try:
                data = json.loads(line)
                
                # 處理提交
                if "submission" in data:
                    submission = data["submission"]
                    meta = extract_enhanced_metadata(submission)
                    body = submission.get("selftext") or submission.get("title", "")
                    
                    if body and len(body) > 50:
                        for chunk in splitter.split_text(body):
                            docs.append(Document(page_content=chunk, metadata=meta))
                
                # 處理所有評論（包括 delta 和 nodelta）
                if "delta_comment" in data:
                    delta_data = data["delta_comment"]
                    if delta_data and "comments" in delta_data:
                        for comment in delta_data["comments"]:
                            if comment.get("body") and len(comment["body"]) > 50:
                                meta = extract_enhanced_metadata(data["submission"], comment)
                                for chunk in splitter.split_text(comment["body"]):
                                    docs.append(Document(page_content=chunk, metadata=meta))
                
                if "nodelta_comment" in data:
                    nodelta_data = data["nodelta_comment"]
                    if nodelta_data and "comments" in nodelta_data:
                        for comment in nodelta_data["comments"]:
                            if comment.get("body") and len(comment["body"]) > 50:
                                # 標記為非成功說服
                                meta = extract_enhanced_metadata(data["submission"], comment)
                                meta['persuasion_success'] = False
                                meta['type'] = 'nodelta_comment'
                                for chunk in splitter.split_text(comment["body"]):
                                    docs.append(Document(page_content=chunk, metadata=meta))
                
                doc_count += 1
                
            except Exception as e:
                print(f"處理文檔時出錯: {e}")
                continue
    
    print(f"收集了 {len(docs)} 個文檔片段")
    
    # 構建向量索引，使用傳入的批次大小
    embeddings = EnhancedEmbeddings(model=EMB_MODEL, batch_size=batch_size)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(output_dir),
        collection_name="social_debate_collection"
    )
    vectorstore.persist()
    
    return {
        "total_docs": len(docs),
        "embedding_cost": embeddings.total_cost,
        "output_dir": str(output_dir)
    }

def build_simple_index(data_path: str, output_path: str, max_docs: int = None):
    """構建簡單的 JSON 索引"""
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # 創建輸出目錄
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    documents = []
    doc_count = 0
    
    if data_path.exists():
        with data_path.open(encoding="utf-8") as f:
            for line in tqdm(f, desc="處理文檔"):
                if max_docs and doc_count >= max_docs:
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # 處理提交
                    if "submission" in data:
                        submission = data["submission"]
                        body = submission.get("selftext") or submission.get("title", "")
                        
                        if body and len(body) > 50:
                            topics = classify_topics(body)
                            doc = {
                                "id": f"doc_{doc_count:05d}",  # 改為5位數以支援更多文檔
                                "content": body[:500],  # 限制長度
                                "metadata": {
                                    "type": "submission",
                                    "topic": topics[0] if topics else "general",
                                    "stance": "neutral",
                                    "quality_score": min(submission.get("score", 0) / 100, 1.0)
                                }
                            }
                            documents.append(doc)
                            doc_count += 1
                    
                    # 處理 delta 評論
                    if "delta_comment" in data and data["delta_comment"]:
                        delta_comment = data["delta_comment"]
                        # 處理評論列表中的所有評論
                        if "comments" in delta_comment and delta_comment["comments"]:
                            for comment in delta_comment["comments"]:
                                if comment.get("body") and len(comment["body"]) > 50:
                                    topics = classify_topics(comment["body"])
                                    doc = {
                                        "id": f"doc_{doc_count:05d}",
                                        "content": comment["body"][:500],
                                        "metadata": {
                                            "type": "delta_comment",
                                            "topic": topics[0] if topics else "general",
                                            "stance": "persuasive",
                                            "quality_score": min(comment.get("score", 0) / 100, 1.0),
                                            "persuasion_success": True
                                        }
                                    }
                                    documents.append(doc)
                                    doc_count += 1
                                    
                                    if max_docs and doc_count >= max_docs:
                                        break
                    
                    # 處理 nodelta 評論
                    if "nodelta_comment" in data and data["nodelta_comment"]:
                        nodelta_comment = data["nodelta_comment"]
                        if "comments" in nodelta_comment and nodelta_comment["comments"]:
                            for comment in nodelta_comment["comments"]:
                                if comment.get("body") and len(comment["body"]) > 50:
                                    topics = classify_topics(comment["body"])
                                    doc = {
                                        "id": f"doc_{doc_count:05d}",
                                        "content": comment["body"][:500],
                                        "metadata": {
                                            "type": "nodelta_comment",
                                            "topic": topics[0] if topics else "general",
                                            "stance": "argumentative",
                                            "quality_score": min(comment.get("score", 0) / 100, 1.0),
                                            "persuasion_success": False
                                        }
                                    }
                                    documents.append(doc)
                                    doc_count += 1
                                    
                                    if max_docs and doc_count >= max_docs:
                                        break
                    
                except Exception as e:
                    continue
    
    # 如果沒有數據，使用預設文檔
    if not documents:
        documents = [
            {
                "id": "doc_001",
                "content": "人工智慧的監管是一個複雜的議題。支持者認為，適當的監管可以防止 AI 被濫用。",
                "metadata": {
                    "type": "expert_opinion",
                    "topic": "AI監管",
                    "stance": "支持",
                    "quality_score": 0.85
                }
            }
        ]
    
    # 保存索引
    index_data = {
        "documents": documents,
        "metadata": {
            "version": "1.0",
            "created_at": time.strftime("%Y-%m-%d"),
            "total_documents": len(documents)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 簡單索引構建完成，共 {len(documents)} 個文檔")
    
    return documents

if __name__ == "__main__":
    build_all_indexes() 