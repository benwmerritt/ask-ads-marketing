"""BM25 index for keyword-based retrieval."""
import pickle
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index for keyword search over chunks."""

    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.chunk_contents: List[str] = []

    def build(self, chunks: List[Any]) -> None:
        """Build BM25 index from chunks."""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 not installed")

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_contents = [c.content for c in chunks]

        # Tokenize (simple whitespace + lowercase)
        tokenized = [doc.lower().split() for doc in self.chunk_contents]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(f"Built BM25 index with {len(chunks)} documents")

    def search(self, query: str, top_k: int = 30) -> List[Tuple[str, str, float]]:
        """Search and return (chunk_id, content, score) tuples."""
        if self.bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches
                results.append((
                    self.chunk_ids[idx],
                    self.chunk_contents[idx],
                    float(scores[idx])
                ))
        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        data = {
            'bm25': self.bm25,
            'chunk_ids': self.chunk_ids,
            'chunk_contents': self.chunk_contents
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: str) -> 'BM25Index':
        """Load index from disk."""
        instance = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance.bm25 = data['bm25']
        instance.chunk_ids = data['chunk_ids']
        instance.chunk_contents = data['chunk_contents']
        logger.info(f"Loaded BM25 index with {len(instance.chunk_ids)} documents")
        return instance
