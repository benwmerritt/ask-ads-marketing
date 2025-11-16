#!/usr/bin/env python3
"""
Universal Query Tool for Knowledge Base
Search any YouTube creator's content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import chromadb
from chromadb.errors import NotFoundError
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import yaml
from typing import List, Dict, Any, Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Silence stdout logging when running under MCP by redirecting rich to stderr
SILENT = os.getenv('WIKI_VAULT_SILENT') == '1'
console = Console(file=sys.stderr) if SILENT else Console()


class KnowledgeQuery:
    """Query interface for any knowledge base"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        import os as _os
        config_path = _os.getenv('WIKI_VAULT_CONFIG', config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get the same embedding function used during ingestion
        self.embedding_function = self._get_embedding_function()
        self._using_fallback = False
        self._fallback_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._fallback_reason: Optional[str] = None
        
        # Connect to Chroma
        self.client = chromadb.PersistentClient(
            path=self.config['chroma']['persist_directory'],
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load collections with correct embedding function
        prefix = self.config['chroma']['collection_prefix']
        names = self.config['chroma']['collections']
        
        try:
            self.content = self.client.get_collection(
                f"{prefix}_{names['content']}",
                embedding_function=self.embedding_function
            )
            self.concepts = self.client.get_collection(
                f"{prefix}_{names['concepts']}",
                embedding_function=self.embedding_function
            )
            self.metadata = self.client.get_collection(
                f"{prefix}_{names['metadata']}",
                embedding_function=self.embedding_function
            )
        except NotFoundError as e:
            raise RuntimeError(
                "Chroma collections not found for this profile. "
                "Please run ingestion for the selected config: "
                f"WIKI_VAULT_CONFIG='{config_path}' python scripts/ingest.py"
            ) from e
        
        # Try to load quotes collection
        try:
            self.quotes = self.client.get_collection(
                f"{prefix}_{names.get('quotes', 'quotes')}",
                embedding_function=self.embedding_function
            )
        except:
            self.quotes = None
        
        # Display info
        kb_name = self.config['knowledge_base']['name']
        creator = self.config['knowledge_base']['creator']
        
        console.print(f"\n[bold cyan]📚 {kb_name}[/bold cyan]")
        console.print(f"[dim]Creator: {creator}[/dim]")
        console.print(f"[dim]Content: {self.content.count():,} chunks | "
                     f"Concepts: {self.concepts.count():,} | "
                     f"Documents: {self.metadata.count():,}[/dim]\n")

        # If embeddings are unavailable, prepare fallback immediately
        if self.embedding_function is None:
            self._prepare_keyword_fallback(
                reason="OpenAI embeddings unavailable; using keyword-based search fallback"
            )
    
    def search(self, query: str, top_k: int = 10, collection: str = 'content') -> List[Dict]:
        """Search a specific collection"""
        
        if collection == 'content':
            coll = self.content
        elif collection == 'concepts':
            coll = self.concepts
        elif collection == 'quotes' and self.quotes:
            coll = self.quotes
        else:
            coll = self.metadata
        
        if self._using_fallback or self.embedding_function is None:
            return self._keyword_search(query, top_k=top_k, collection=collection)

        try:
            results = coll.query(
                query_texts=[query],
                n_results=top_k
            )
        except Exception as e:
            if not self._using_fallback and self.config['embeddings'].get('provider') == 'openai':
                logger.warning(
                    f"Embedding query failed, switching to keyword-based fallback: {e}"
                )
                self._prepare_keyword_fallback(reason=str(e))
                return self._keyword_search(query, top_k=top_k, collection=collection)
            raise

        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Convert cosine distance (0-2 range) to similarity score (0-1 range)
            # Cosine distance: 0 = identical, 2 = opposite
            distance = results['distances'][0][i]
            similarity = max(0, (2 - distance) / 2)
            
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': similarity
            })
        
        return formatted_results
    
    def _get_embedding_function(self):
        """Get the same embedding function used during ingestion"""
        provider = self.config['embeddings']['provider']
        
        # Try OpenAI first (best quality)
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and len(api_key) > 20 and not api_key.endswith('...'):
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=self.config['embeddings']['model']
                )
            else:
                logger.warning("OpenAI API key not found; embeddings disabled")
        
        # No compatible embedding provider – use keyword fallback
        return None

    def _prepare_keyword_fallback(self, reason: Optional[str] = None) -> None:
        """Load documents into memory for keyword-based fallback search."""

        if self._using_fallback and self._fallback_cache:
            return

        if reason:
            self._fallback_reason = reason
            logger.warning(f"Using keyword fallback: {reason}")

        collections = {
            'content': self.content,
            'concepts': self.concepts,
            'metadata': self.metadata,
        }
        if self.quotes:
            collections['quotes'] = self.quotes

        for name, coll in collections.items():
            if not coll:
                self._fallback_cache[name] = []
                continue
            total = coll.count()
            if total == 0:
                self._fallback_cache[name] = []
                continue
            documents: List[Dict[str, Any]] = []
            batch_size = 500
            for offset in range(0, total, batch_size):
                chunk = coll.get(limit=batch_size, offset=offset)
                ids = chunk.get('ids', [])
                docs = chunk.get('documents', [])
                metas = chunk.get('metadatas', [])
                for idx, doc in enumerate(docs):
                    documents.append({
                        'id': ids[idx] if idx < len(ids) else None,
                        'content': doc or '',
                        'metadata': metas[idx] if idx < len(metas) else {}
                    })
            self._fallback_cache[name] = documents

        self._using_fallback = True

    def _keyword_search(self, query: str, top_k: int, collection: str) -> List[Dict[str, Any]]:
        """Return top_k results using simple keyword scoring."""

        self._prepare_keyword_fallback()

        data = self._fallback_cache.get(collection) or self._fallback_cache.get('content', [])
        if not data:
            return []

        keywords = self._extract_keywords(query)
        scored = []
        for item in data:
            text = (item.get('content') or '').lower()
            meta = item.get('metadata') or {}
            score = 0.0

            for word in keywords:
                if not word:
                    continue
                score += text.count(word)
                title = (meta.get('title') or '').lower()
                if word in title:
                    score += 2.0

            if not keywords:
                score = 1.0

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_entries = [entry for _, entry in scored[:top_k]] if scored else []

        results = []
        for entry in top_entries:
            results.append({
                'id': entry.get('id'),
                'content': entry.get('content'),
                'metadata': entry.get('metadata'),
                'score': 0.0
            })

        return results

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[A-Za-z0-9']+", text.lower()) if len(tok) >= 3]
    
    def display_results(self, results: List[Dict], query: str, result_type: str = "content"):
        """Display search results nicely"""
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Results for: '{query}'", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="blue", width=40)
        table.add_column("Content", style="white")
        
        for i, result in enumerate(results[:10], 1):
            score = f"{result['score']:.1%}"
            
            # Get source info
            if result_type == "content":
                source = result['metadata'].get('title', 'Unknown')[:40]
                section = result['metadata'].get('section', '')
                if section:
                    source += f"\n[dim]{section}[/dim]"
            elif result_type == "concepts":
                source = result['metadata'].get('concept_name', 'Unknown')
                video = result['metadata'].get('source_title', '')[:30]
                if video:
                    source += f"\n[dim]{video}[/dim]"
            else:
                source = result['metadata'].get('title', 'Unknown')[:40]
            
            # Get content preview
            content = result['content'].replace('\n', ' ')[:200]
            if len(result['content']) > 200:
                content += "..."
            
            table.add_row(str(i), score, source, content)
        
        console.print(table)
    
    def search_all(self, query: str):
        """Search across all collections"""
        
        console.print(f"\n[bold]🔍 Searching for: '{query}'[/bold]\n")
        
        # Search content
        console.print("[cyan]📄 Content Results:[/cyan]")
        content_results = self.search(query, top_k=5, collection='content')
        self.display_results(content_results, query, "content")
        
        # Search concepts
        console.print("\n[cyan]💡 Concept Results:[/cyan]")
        concept_results = self.search(query, top_k=3, collection='concepts')
        if concept_results:
            for result in concept_results:
                name = result['metadata'].get('concept_name', 'Unknown')
                source = result['metadata'].get('source_title', '')
                console.print(f"  • [bold]{name}[/bold] from {source} [{result['score']:.1%}]")
        else:
            console.print("  [dim]No concepts found[/dim]")
        
        # Search quotes if available
        if self.quotes:
            console.print("\n[cyan]💬 Quote Results:[/cyan]")
            quote_results = self.search(query, top_k=2, collection='quotes')
            if quote_results:
                for result in quote_results:
                    quote = result['metadata'].get('quote_text', result['content'])[:150]
                    source = result['metadata'].get('source_title', '')[:30]
                    console.print(f"  • \"{quote}...\" - {source} [{result['score']:.1%}]")
            else:
                console.print("  [dim]No quotes found[/dim]")
    
    def list_top_concepts(self, limit: int = 20):
        """List all concepts in the knowledge base"""
        
        all_concepts = self.concepts.get(limit=limit * 2)
        
        # Aggregate by concept name
        concept_freq = {}
        for meta in all_concepts['metadatas']:
            name = meta.get('concept_name', 'Unknown')
            if name not in concept_freq:
                concept_freq[name] = 0
            concept_freq[name] += 1
        
        # Sort by frequency
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Display
        table = Table(title="Top Concepts", show_header=True)
        table.add_column("Concept", style="cyan")
        table.add_column("Mentions", style="green")
        
        for name, count in sorted_concepts:
            table.add_row(name, str(count))
        
        console.print(table)
    
    def find_video(self, title_query: str):
        """Find information about a specific video/document"""
        
        results = self.metadata.query(
            query_texts=[title_query],
            n_results=5
        )
        
        if not results['ids'][0]:
            console.print("[yellow]No matching documents found[/yellow]")
            return
        
        console.print(f"\n[bold]📹 Documents matching: '{title_query}'[/bold]\n")
        
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            doc = results['documents'][0][i]
            score = 1 - results['distances'][0][i]
            
            panel = Panel(
                doc,
                title=f"[bold]{meta.get('title', 'Unknown')}[/bold] [{score:.1%}]",
                subtitle=meta.get('url', ''),
                expand=False
            )
            console.print(panel)


def interactive_mode():
    """Interactive query interface"""
    
    query_tool = KnowledgeQuery()
    
    # Load config for personalization
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    kb_name = config['knowledge_base']['name']
    
    console.print(f"\n[bold green]🎯 {kb_name} Query Tool[/bold green]")
    console.print("[dim]Commands: 'help', 'concepts', 'video [title]', 'quit'[/dim]")
    console.print("[dim]Or enter any search query[/dim]\n")
    
    while True:
        try:
            query = console.input("[bold]> [/bold]").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit' or query.lower() == 'exit':
                console.print("[green]Goodbye![/green]")
                break
            
            elif query.lower() == 'help':
                console.print("""
[bold]Available Commands:[/bold]
  • [cyan]Any text[/cyan] - Search across all content
  • [cyan]concepts[/cyan] - List top concepts
  • [cyan]video [title][/cyan] - Find specific video/document
  • [cyan]quit[/cyan] - Exit the tool
                """)
            
            elif query.lower() == 'concepts':
                query_tool.list_top_concepts()
            
            elif query.lower().startswith('video '):
                title = query[6:]
                query_tool.find_video(title)
            
            else:
                # Regular search
                query_tool.search_all(query)
            
            console.print()  # Empty line for readability
            
        except KeyboardInterrupt:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def batch_mode(queries: List[str]):
    """Run batch queries"""
    
    query_tool = KnowledgeQuery()
    
    for query in queries:
        console.rule(f"Query: {query}")
        query_tool.search_all(query)
        console.print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Batch mode with command line queries
        queries = sys.argv[1:]
        batch_mode(queries)
    else:
        # Interactive mode
        interactive_mode()
