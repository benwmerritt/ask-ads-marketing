#!/usr/bin/env python3
"""
Universal Query Tool for Knowledge Base
Search any YouTube creator's content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import chromadb
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

# Optional: Voyage AI for asymmetric embeddings
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

# Optional: Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# Optional: BM25 for hybrid search
try:
    from lib.bm25_index import BM25Index, BM25_AVAILABLE
except ImportError:
    BM25_AVAILABLE = False
    BM25Index = None

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

        # Voyage AI for asymmetric query embeddings
        self._using_voyage = False
        self._voyage_client = None
        self._voyage_model = None
        if self.config['embeddings'].get('provider') == 'voyage':
            voyage_key = os.getenv('VOYAGE_API_KEY')
            if voyage_key and VOYAGE_AVAILABLE:
                self._using_voyage = True
                self._voyage_client = voyageai.Client(api_key=voyage_key)
                self._voyage_model = self.config['embeddings'].get('model', 'voyage-3-lite')
                logger.info(f"Using Voyage asymmetric query embeddings: {self._voyage_model}")

        # Cohere for reranking
        self._cohere_client = None
        rerank_config = self.config.get('reranking', {})
        if rerank_config.get('enabled') and COHERE_AVAILABLE:
            cohere_key = os.getenv('COHERE_API_KEY')
            if cohere_key:
                self._cohere_client = cohere.ClientV2(api_key=cohere_key)
                self._rerank_model = rerank_config.get('model', 'rerank-english-v3.0')
                self._rerank_initial_k = rerank_config.get('initial_k', 30)
                self._rerank_final_k = rerank_config.get('final_k', 10)
                logger.info(f"Cohere reranking enabled: {self._rerank_model}")

        # BM25 for hybrid search
        self._bm25_index = None
        search_config = self.config.get('search', {})
        if search_config.get('hybrid_enabled') and BM25_AVAILABLE:
            bm25_path = search_config.get('bm25_index_path', './data/bm25_index.pkl')
            try:
                from pathlib import Path
                if Path(bm25_path).exists():
                    self._bm25_index = BM25Index.load(bm25_path)
                    self._hybrid_alpha = search_config.get('hybrid_alpha', 0.7)
                    logger.info(f"BM25 hybrid search enabled (alpha={self._hybrid_alpha})")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

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
        except Exception as e:
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
    
    def search(self, query: str, top_k: int = 10, collection: str = 'content') -> List[Dict]:
        """Search with optional Voyage embeddings, BM25 hybrid fusion, and Cohere reranking."""

        if collection == 'content':
            coll = self.content
        elif collection == 'concepts':
            coll = self.concepts
        elif collection == 'quotes' and self.quotes:
            coll = self.quotes
        else:
            coll = self.metadata

        # Determine fetch count based on reranking
        if self._cohere_client and collection == 'content':
            fetch_count = self._rerank_initial_k
        else:
            fetch_count = min(top_k * 3, 100)

        # Query ChromaDB with Voyage embeddings if available
        try:
            if self._using_voyage:
                query_embedding = self._embed_query_voyage(query)
                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_count
                )
            else:
                results = coll.query(
                    query_texts=[query],
                    n_results=fetch_count
                )
        except Exception as e:
            # Fallback to local embeddings if needed
            if not self._using_fallback and self.config['embeddings'].get('provider') in ('openai', 'voyage'):
                logger.warning(f"Embedding query failed, falling back to local embeddings: {e}")
                self._fallback_to_local_embeddings()
                if collection == 'content':
                    coll = self.content
                elif collection == 'concepts':
                    coll = self.concepts
                elif collection == 'quotes' and self.quotes:
                    coll = self.quotes
                else:
                    coll = self.metadata
                results = coll.query(
                    query_texts=[query],
                    n_results=fetch_count
                )
            else:
                raise

        # Format vector results
        vector_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            semantic_score = max(0, (2 - distance) / 2)
            vector_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': semantic_score,
                'vector_rank': i + 1
            })

        # Apply hybrid BM25 fusion if available and for content collection
        if self._bm25_index and collection == 'content':
            formatted_results = self._hybrid_search(query, vector_results, fetch_count)
        else:
            formatted_results = vector_results

        # Apply Cohere reranking if available and for content collection
        if self._cohere_client and collection == 'content':
            formatted_results = self._rerank_results(query, formatted_results)
            return formatted_results[:top_k]

        # Sort and return
        formatted_results.sort(key=lambda x: x['score'], reverse=True)
        return formatted_results[:top_k]

    def _embed_query_voyage(self, query: str) -> List[float]:
        """Embed query with Voyage AI using input_type='query' for asymmetric retrieval."""
        result = self._voyage_client.embed(
            texts=[query],
            model=self._voyage_model,
            input_type="query",
            truncation=True
        )
        return result.embeddings[0]

    def _hybrid_search(self, query: str, vector_results: List[Dict], k: int) -> List[Dict]:
        """Combine vector and BM25 results using Reciprocal Rank Fusion (RRF)."""
        # Get BM25 results
        bm25_results = self._bm25_index.search(query, top_k=k)

        # Build ID -> result mapping from vector results
        id_to_result = {r['id']: r for r in vector_results}

        # RRF constant (standard value)
        rrf_k = 60

        # Calculate RRF scores
        rrf_scores = {}

        # Vector rankings (alpha weight)
        for rank, result in enumerate(vector_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = self._hybrid_alpha * (1 / (rrf_k + rank))

        # BM25 rankings (1-alpha weight)
        for rank, (doc_id, content, bm25_score) in enumerate(bm25_results, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                # Add to result mapping if not from vector search
                id_to_result[doc_id] = {
                    'id': doc_id,
                    'content': content,
                    'metadata': {},  # BM25 doesn't have metadata
                    'score': 0
                }
            rrf_scores[doc_id] += (1 - self._hybrid_alpha) * (1 / (rrf_k + rank))

        # Update scores and sort
        for doc_id, rrf_score in rrf_scores.items():
            if doc_id in id_to_result:
                id_to_result[doc_id]['score'] = rrf_score
                id_to_result[doc_id]['rrf_score'] = rrf_score

        results = list(id_to_result.values())
        results.sort(key=lambda x: x.get('rrf_score', x['score']), reverse=True)
        return results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using Cohere cross-encoder."""
        if not results:
            return results

        documents = [r['content'] for r in results]

        try:
            rerank_response = self._cohere_client.rerank(
                model=self._rerank_model,
                query=query,
                documents=documents,
                top_n=self._rerank_final_k
            )

            reranked = []
            for item in rerank_response.results:
                result = results[item.index].copy()
                result['rerank_score'] = item.relevance_score
                result['original_score'] = result['score']
                result['score'] = item.relevance_score
                reranked.append(result)

            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:self._rerank_final_k]

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword matching score using n-gram overlap"""
        # Extract meaningful query terms (3+ chars, lowercase)
        query_terms = set()
        for word in query.lower().split():
            # Remove common punctuation
            word = word.strip('.,!?;:()[]{}""\'')
            if len(word) >= 3:
                query_terms.add(word)

        if not query_terms:
            return 0.0

        content_lower = content.lower()

        # Count exact term matches
        exact_matches = sum(1 for term in query_terms if term in content_lower)

        # Bonus for multi-word phrase matches
        phrase_bonus = 0
        if len(query.split()) >= 2:
            # Check if full query phrase appears
            if query.lower() in content_lower:
                phrase_bonus = 0.5
            # Check for 2-word phrases
            words = query.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in content_lower:
                    phrase_bonus = max(phrase_bonus, 0.3)

        # Normalize score: (matches / total_terms) + phrase_bonus
        # Max score = 1.5 (perfect match + phrase)
        if not query_terms:
            return 0.0

        keyword_score = (exact_matches / len(query_terms)) + phrase_bonus
        return min(keyword_score, 1.5)  # Cap at 1.5
    
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
                logger.warning("OpenAI API key not found, falling back to local embeddings")
        
        # Fallback to local embeddings
        fallback_model = self.config['embeddings'].get('fallback_model', 'all-MiniLM-L6-v2')
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=fallback_model
        )

    def _fallback_to_local_embeddings(self):
        """Reinitialize collections with local sentence-transformer embeddings"""
        fallback_model = self.config['embeddings'].get('fallback_model', 'all-MiniLM-L6-v2')
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=fallback_model
        )
        self._using_fallback = True
        # Reload collections with the new embedding function
        prefix = self.config['chroma']['collection_prefix']
        names = self.config['chroma']['collections']
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
        try:
            self.quotes = self.client.get_collection(
                f"{prefix}_{names.get('quotes', 'quotes')}",
                embedding_function=self.embedding_function
            )
        except:
            self.quotes = None
    
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
