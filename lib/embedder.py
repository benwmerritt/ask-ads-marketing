"""
Universal Embedding System
Supports OpenAI (best quality) with local fallback
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalEmbedder:
    """Create and manage embeddings for any knowledge base"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        config_path = os.getenv('WIKI_VAULT_CONFIG', config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Chroma
        self.client = chromadb.PersistentClient(
            path=self.config['chroma']['persist_directory'],
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get embedding function
        self.embedding_function = self._get_embedding_function()
        
        # Collection prefix for uniqueness
        self.prefix = self.config['chroma']['collection_prefix']
        
        # Initialize collections
        self.collections = self._init_collections()
        
    def _get_embedding_function(self):
        """Get the best available embedding function"""
        provider = self.config['embeddings']['provider']
        
        # Try OpenAI first (best quality)
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                logger.info(f"Using OpenAI embeddings: {self.config['embeddings']['model']}")
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=self.config['embeddings']['model']
                )
            else:
                logger.warning("OpenAI API key not found, falling back to local embeddings")
        
        # Fallback to local embeddings
        fallback_model = self.config['embeddings'].get('fallback_model', 'all-MiniLM-L6-v2')
        logger.info(f"Using local embeddings: {fallback_model}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=fallback_model
        )
    
    def _init_collections(self) -> Dict[str, chromadb.Collection]:
        """Initialize collections (create if not exist)"""
        collections = {}
        names = self.config['chroma']['collections']
        
        # Main content collection
        content_name = f"{self.prefix}_{names['content']}"
        collections['content'] = self.client.get_or_create_collection(
            name=content_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"{self.config['knowledge_base']['name']} - Main content",
                "creator": self.config['knowledge_base']['creator'],
                "topic": self.config['knowledge_base']['topic']
            }
        )
        
        # Concepts collection
        concepts_name = f"{self.prefix}_{names['concepts']}"
        collections['concepts'] = self.client.get_or_create_collection(
            name=concepts_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"{self.config['knowledge_base']['name']} - Concepts",
                "creator": self.config['knowledge_base']['creator']
            }
        )
        
        # Metadata collection
        metadata_name = f"{self.prefix}_{names['metadata']}"
        collections['metadata'] = self.client.get_or_create_collection(
            name=metadata_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"{self.config['knowledge_base']['name']} - Video metadata"
            }
        )
        
        # Quotes collection (if applicable)
        if 'quotes' in names:
            quotes_name = f"{self.prefix}_{names['quotes']}"
            collections['quotes'] = self.client.get_or_create_collection(
                name=quotes_name,
                embedding_function=self.embedding_function,
                metadata={"description": f"{self.config['knowledge_base']['name']} - Quotes"}
            )
        
        logger.info(f"Initialized {len(collections)} collections")
        return collections
    
    def embed_chunks(self, chunks: List, batch_size: int = None) -> int:
        """Embed content chunks"""
        if batch_size is None:
            batch_size = self.config['embeddings']['batch_size']
        
        collection = self.collections['content']
        embedded_count = 0
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch
            ids = []
            documents = []
            metadatas = []
            
            for chunk in batch:
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                # Clean metadata
                metadata = chunk.metadata.copy()
                metadata['chunk_type'] = chunk.chunk_type
                metadata['token_count'] = chunk.token_count
                
                # Ensure all values are strings or numbers
                for key, value in metadata.items():
                    if value is None:
                        metadata[key] = ""
                    elif isinstance(value, (list, dict)):
                        metadata[key] = str(value)
                    elif hasattr(value, '_type'):
                        # Handle objects with _type attribute
                        metadata[key] = str(value)
                    elif not isinstance(value, (str, int, float, bool)):
                        # Convert any other object to string
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            embedded_count += len(batch)
        
        logger.info(f"✅ Embedded {embedded_count} chunks")
        return embedded_count
    
    def embed_concepts(self, documents: List) -> int:
        """Embed concepts separately"""
        collection = self.collections['concepts']
        concepts_data = []
        
        for doc_idx, doc in enumerate(documents):
            for concept_idx, concept in enumerate(doc.concepts):
                if concept.get('name'):
                    # Add indices to ensure uniqueness
                    cname = str(concept.get('name') or 'concept')
                    concept_id = f"doc{doc_idx:03d}_concept{concept_idx:02d}_{cname.replace(' ', '_')}"
                    
                    # Create concept document
                    concept_doc = f"Concept: {cname}\n"
                    if concept.get('definition'):
                        concept_doc += f"Definition: {concept['definition']}\n"
                    safe_title = (doc.title or Path(doc.file_path).stem)
                    concept_doc += f"Source: {safe_title}\n"
                    concept_doc += f"Creator: {doc.creator or self.config['knowledge_base']['creator']}"
                    
                    # Build safe metadata (no None values)
                    meta = {
                        'concept_name': concept.get('name') or '',
                        'source_title': doc.title or '',
                        'source_url': (doc.url or ''),
                        'creator': (doc.creator or self.config['knowledge_base']['creator'] or '')
                    }
                    # Normalize metadata types
                    for k, v in list(meta.items()):
                        if v is None:
                            meta[k] = ''
                        elif isinstance(v, (list, dict)):
                            meta[k] = str(v)
                        elif not isinstance(v, (str, int, float, bool)):
                            meta[k] = str(v)

                    concepts_data.append({
                        'id': concept_id[:63],  # Chroma has 63 char limit
                        'document': concept_doc,
                        'metadata': meta
                    })
        
        # Batch add concepts
        if concepts_data:
            batch_size = self.config['embeddings']['batch_size']
            
            for i in tqdm(range(0, len(concepts_data), batch_size), desc="Embedding concepts"):
                batch = concepts_data[i:i + batch_size]
                # Final safety pass on metadata
                safe_metas = []
                for c in batch:
                    m = dict(c['metadata'])
                    for k, v in list(m.items()):
                        if v is None:
                            m[k] = ''
                        elif isinstance(v, (list, dict)):
                            m[k] = str(v)
                        elif not isinstance(v, (str, int, float, bool)):
                            m[k] = str(v)
                    safe_metas.append(m)

                collection.add(
                    ids=[c['id'] for c in batch],
                    documents=[c['document'] for c in batch],
                    metadatas=safe_metas
                )
        
        logger.info(f"✅ Embedded {len(concepts_data)} concepts")
        return len(concepts_data)
    
    def embed_metadata(self, documents: List) -> int:
        """Embed document metadata"""
        collection = self.collections['metadata']
        
        for doc in tqdm(documents, desc="Embedding metadata"):
            # Create metadata document
            meta_doc = f"Title: {doc.title}\n"
            meta_doc += f"Type: {doc.doc_type}\n"
            
            if doc.url:
                meta_doc += f"URL: {doc.url}\n"
            
            if doc.summary:
                meta_doc += f"\nSummary: {doc.summary}\n"
            
            if doc.lessons:
                meta_doc += f"\nKey Points: {'; '.join(doc.lessons[:3])}\n"
            
            safe_title = (doc.title or Path(doc.file_path).stem)
            metadata = {
                'title': safe_title,
                'doc_type': doc.doc_type,
                'url': doc.url or '',
                'creator': doc.creator or self.config['knowledge_base']['creator'],
                'video_id': doc.video_id or '',
                'file_path': doc.file_path
            }
            
            # Clean metadata
            for key, value in metadata.items():
                if value is None:
                    metadata[key] = ""
                elif hasattr(value, '_type'):
                    # Handle objects with _type attribute
                    metadata[key] = str(value)
                elif not isinstance(value, (str, int, float, bool)):
                    # Convert any other object to string
                    metadata[key] = str(value)
            
            doc_id_source = (doc.video_id or safe_title)
            doc_id = str(doc_id_source).replace(' ', '_')
            
            collection.add(
                ids=[doc_id[:63]],
                documents=[meta_doc],
                metadatas=[metadata]
            )
        
        logger.info(f"✅ Embedded metadata for {len(documents)} documents")
        return len(documents)
    
    def embed_quotes(self, documents: List) -> int:
        """Embed quotes if collection exists"""
        if 'quotes' not in self.collections:
            return 0
        
        collection = self.collections['quotes']
        quotes_data = []
        
        for doc in documents:
            for i, quote in enumerate(doc.quotes):
                if quote.get('text'):
                    safe_title = (doc.title or Path(doc.file_path).stem)
                    quote_id = f"{str(safe_title).replace(' ', '_')}_q{i:03d}"
                    
                    quote_doc = f'"{quote["text"]}"\n\n'
                    quote_doc += f"From: {safe_title}\n"
                    quote_doc += f"Creator: {doc.creator or self.config['knowledge_base']['creator']}"
                    
                    meta = {
                        'quote_text': (quote.get('text') or '')[:500],
                        'source_title': doc.title or '',
                        'source_url': doc.url or '',
                        'timestamp': quote.get('timestamp') or '',
                        'creator': (doc.creator or self.config['knowledge_base']['creator'] or '')
                    }
                    # Normalize
                    for k, v in list(meta.items()):
                        if v is None:
                            meta[k] = ''
                        elif isinstance(v, (list, dict)):
                            meta[k] = str(v)
                        elif not isinstance(v, (str, int, float, bool)):
                            meta[k] = str(v)

                    quotes_data.append({
                        'id': quote_id[:63],
                        'document': quote_doc,
                        'metadata': meta
                    })
        
        # Batch add quotes
        if quotes_data:
            batch_size = self.config['embeddings']['batch_size']
            
            for i in range(0, len(quotes_data), batch_size):
                batch = quotes_data[i:i + batch_size]
                
                # Safety pass
                safe_metas = []
                for q in batch:
                    m = dict(q['metadata'])
                    for k, v in list(m.items()):
                        if v is None:
                            m[k] = ''
                        elif isinstance(v, (list, dict)):
                            m[k] = str(v)
                        elif not isinstance(v, (str, int, float, bool)):
                            m[k] = str(v)
                    safe_metas.append(m)

                collection.add(
                    ids=[q['id'] for q in batch],
                    documents=[q['document'] for q in batch],
                    metadatas=safe_metas
                )
        
        logger.info(f"✅ Embedded {len(quotes_data)} quotes")
        return len(quotes_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        stats = {}
        
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {
                'name': collection.name,
                'count': count,
                'metadata': collection.metadata
            }
        
        return stats


def create_embeddings(documents: List, chunks: List) -> UniversalEmbedder:
    """Main function to create all embeddings"""
    
    embedder = UniversalEmbedder()
    
    # Embed everything
    logger.info("Creating embeddings...")
    
    content_count = embedder.embed_chunks(chunks)
    concepts_count = embedder.embed_concepts(documents)
    metadata_count = embedder.embed_metadata(documents)
    quotes_count = embedder.embed_quotes(documents)
    
    # Show stats
    stats = embedder.get_statistics()
    
    print("\n📊 Embedding Statistics:")
    for name, stat in stats.items():
        print(f"  {name}: {stat['count']:,} items")
    
    return embedder


if __name__ == "__main__":
    # Test embeddings
    from lib.processor import DocumentProcessor
    from lib.chunker import SmartChunker
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  No OpenAI API key found. Using local embeddings.")
        print("For best results, set OPENAI_API_KEY in .env file")
    
    processor = DocumentProcessor()
    documents = processor.process_all()
    
    if documents:
        chunker = SmartChunker()
        chunks = chunker.chunk_documents(documents)
        
        embedder = create_embeddings(documents, chunks)
        print("\n✅ Embeddings created successfully!")
