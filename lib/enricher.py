"""LLM-based chunk enrichment for improved semantic search."""
import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class EnrichmentResult:
    summary: str
    keywords: List[str]


class ChunkEnricher:
    """Enrich chunks with LLM-generated summaries and keywords."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('metadata_injection', {})
        self.enabled = self.config.get('enabled', False)
        self.model = self.config.get('model', 'claude-3-5-haiku-20241022')
        self.batch_size = self.config.get('batch_size', 5)
        self.client = None

        if self.enabled and ANTHROPIC_AVAILABLE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                logger.warning("ANTHROPIC_API_KEY not set, enrichment disabled")

    def enrich_chunks(self, chunks: List) -> List:
        """Add LLM-generated summaries and keywords to chunks."""
        if not self.enabled:
            return chunks

        if not self.client:
            logger.warning("Enrichment enabled but Anthropic client unavailable")
            return chunks

        enriched = []
        failed_batches = 0

        # Process in batches
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Enriching chunks"):
            batch = chunks[i:i + self.batch_size]
            results = self._enrich_batch(batch)

            if not any(r.summary for r in results):
                failed_batches += 1

            for chunk, result in zip(batch, results):
                # Add to metadata
                chunk.metadata['llm_summary'] = result.summary
                chunk.metadata['llm_keywords'] = result.keywords

                # Create enriched content for embedding
                if result.summary:
                    chunk.enriched_content = (
                        f"Summary: {result.summary}\n"
                        f"Keywords: {', '.join(result.keywords)}\n\n"
                        f"{chunk.content}"
                    )
                else:
                    chunk.enriched_content = chunk.content

                enriched.append(chunk)

        success_rate = ((len(chunks) // self.batch_size) - failed_batches) / max(1, len(chunks) // self.batch_size)
        logger.info(f"Enriched {len(enriched)} chunks (success rate: {success_rate:.0%})")
        return enriched

    def _enrich_batch(self, batch: List) -> List[EnrichmentResult]:
        """Process a batch of chunks with a single LLM call."""
        # Build batch prompt
        chunks_text = ""
        for idx, chunk in enumerate(batch):
            content_preview = chunk.content[:1200]  # Limit per chunk
            chunks_text += f"\n---CHUNK {idx}---\n{content_preview}\n"

        prompt = f"""Analyze these content excerpts and provide a summary and keywords for each.

{chunks_text}

For each chunk, provide:
1. A 1-2 sentence summary of the main point
2. 3-5 key concepts/keywords

Respond in JSON format as an array:
[
  {{"chunk": 0, "summary": "...", "keywords": ["...", "..."]}},
  {{"chunk": 1, "summary": "...", "keywords": ["...", "..."]}}
]

Return ONLY valid JSON, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (```json and ```)
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                if text.startswith("json"):
                    text = text[4:].strip()

            results = json.loads(text)

            # Convert to EnrichmentResult objects
            enrichments = []
            for item in results:
                enrichments.append(EnrichmentResult(
                    summary=item.get('summary', ''),
                    keywords=item.get('keywords', [])
                ))

            # Pad if needed (in case LLM missed some)
            while len(enrichments) < len(batch):
                enrichments.append(EnrichmentResult(summary='', keywords=[]))

            return enrichments

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in batch: {e}")
            return [EnrichmentResult(summary='', keywords=[]) for _ in batch]
        except Exception as e:
            logger.warning(f"Batch enrichment failed: {e}")
            return [EnrichmentResult(summary='', keywords=[]) for _ in batch]
