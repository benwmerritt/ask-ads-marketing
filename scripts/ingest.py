#!/usr/bin/env python3
"""
Universal Knowledge Base Ingestion Script
Processes any YouTube creator's content into searchable Chroma database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv

from lib.processor import DocumentProcessor
from lib.chunker import SmartChunker
from lib.embedder import create_embeddings

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion pipeline"""
    
    # Load config (supports WIKI_VAULT_CONFIG override)
    import os as _os
    cfg_path = _os.getenv('WIKI_VAULT_CONFIG', 'config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    kb_name = config['knowledge_base']['name']
    creator = config['knowledge_base']['creator']
    topic = config['knowledge_base']['topic']
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           Knowledge Base Ingestion Pipeline               ║
║                                                           ║
║  Name: {kb_name:<50} ║
║  Creator: {creator:<47} ║
║  Topic: {topic:<49} ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Step 1: Process documents
    print("📄 Step 1: Processing documents...")
    print("-" * 50)
    
    processor = DocumentProcessor()
    documents = processor.process_all()
    
    if not documents:
        print("❌ No documents found!")
        print("\nPlease add content to:")
        print("  • source-notes/   (for structured notes)")
        print("  • md-transcripts/ (for video transcripts)")
        return 1
    
    stats = processor.get_statistics()
    print(f"✅ Processed {stats['total_documents']} documents")
    print(f"   - Source notes: {stats['source_notes']}")
    print(f"   - Transcripts: {stats['transcripts']}")
    print(f"   - Concepts found: {stats['total_concepts']}")
    print(f"   - Quotes extracted: {stats['total_quotes']}")
    
    # Step 2: Create chunks
    print("\n📦 Step 2: Creating smart chunks...")
    print("-" * 50)
    
    chunker = SmartChunker()
    chunks = chunker.chunk_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
    
    print("   Chunk types:")
    for ctype, count in chunk_types.items():
        print(f"   - {ctype}: {count}")
    
    # Step 3: Create embeddings
    print("\n🧮 Step 3: Creating embeddings...")
    print("-" * 50)
    
    # Check for OpenAI key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ Using OpenAI embeddings (best quality)")
    else:
        print("⚠️  Using local embeddings (no API key found)")
        print("   For best results, add OPENAI_API_KEY to .env file")
    
    embedder = create_embeddings(documents, chunks)
    
    # Step 4: Show final statistics
    print("\n📊 Step 4: Final Statistics")
    print("-" * 50)
    
    embed_stats = embedder.get_statistics()
    
    total_items = sum(s['count'] for s in embed_stats.values())
    print(f"✅ Total items indexed: {total_items:,}")
    
    for name, stat in embed_stats.items():
        print(f"   - {name}: {stat['count']:,} items")
    
    # Show top concepts
    if stats['top_concepts']:
        print(f"\n💡 Top Concepts:")
        for concept, count in stats['top_concepts'][:5]:
            print(f"   - {concept}: {count} mentions")
    
    # Calculate time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n⏱️  Total time: {minutes}m {seconds}s")
    
    # Success message
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              ✅ INGESTION COMPLETE!                       ║
╚══════════════════════════════════════════════════════════╝

Your {kb_name} is ready!

📊 Database Statistics:
   • Documents: {stats['total_documents']}
   • Chunks: {len(chunks)}
   • Concepts: {stats['total_concepts']}
   • Quotes: {stats['total_quotes']}

🚀 Next Steps:
   1. Search: python scripts/query.py
   2. Ask Expert: python scripts/ask_expert.py
   3. Full Notes: python scripts/full_notes.py

💡 Tip: The first search may take a moment as embeddings are cached.
    """)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion cancelled by user")
        exit_code = 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("  1. Your content is in source-notes/ or md-transcripts/")
        print("  2. Content follows the expected markdown format")
        print("  3. config.yaml is properly configured")
        exit_code = 1
    
    exit(exit_code)
