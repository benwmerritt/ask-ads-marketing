"""
Smart Chunking for Any Content Type
Preserves context and handles both structured notes and transcripts
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of content with metadata"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_type: str  # 'note_section', 'transcript_segment', 'concept', etc.
    

class SmartChunker:
    """Intelligent chunking for any content type"""
    
    def __init__(self, config_path: str = "config.yaml"):
        import os as _os
        config_path = _os.getenv('WIKI_VAULT_CONFIG', config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Chunking settings
        self.notes_chunk_size = self.config['chunking']['notes_chunk_size']
        self.notes_overlap = self.config['chunking']['notes_chunk_overlap']
        self.transcript_minutes = self.config['chunking'].get('transcript_chunk_minutes', 3)
        self.transcript_overlap = self.config['chunking'].get('transcript_overlap_seconds', 30)
        
        # Token encoder
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def chunk_documents(self, documents: List) -> List[Chunk]:
        """Chunk all documents based on their type"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            if doc.doc_type == 'source_note':
                chunks = self._chunk_source_note(doc, doc_idx)
            elif doc.doc_type == 'transcript':
                chunks = self._chunk_transcript(doc, doc_idx)
            else:
                chunks = self._chunk_generic(doc, doc_idx)
            
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_source_note(self, doc, doc_idx: int) -> List[Chunk]:
        """Chunk a structured source note"""
        chunks = []
        chunk_counter = 0
        
        base_metadata = {
            'title': doc.title,
            'file_path': doc.file_path,
            'doc_type': 'source_note',
            'video_id': doc.video_id,
            'url': doc.url,
            'creator': doc.creator
        }
        
        # Chunk 1: Summary/Overview
        if doc.summary:
            chunk = self._create_chunk(
                content=f"# {doc.title}\n\n{doc.summary}",
                metadata={**base_metadata, 'section': 'summary'},
                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                chunk_type='note_section'
            )
            chunks.append(chunk)
            chunk_counter += 1
        
        # Chunk 2: Concepts (each concept separately for precision)
        for concept in doc.concepts:
            if concept.get('name'):
                content = f"Concept: {concept['name']}\n"
                if concept.get('definition'):
                    content += f"Definition: {concept['definition']}\n"
                content += f"From: {doc.title}"
                
                chunk = self._create_chunk(
                    content=content,
                    metadata={
                        **base_metadata,
                        'section': 'concept',
                        'concept_name': concept['name']
                    },
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type='concept'
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        # Chunk 3: Lessons (group related lessons)
        if doc.lessons:
            lesson_chunks = self._chunk_list_content(
                items=doc.lessons,
                title=f"Key Lessons from {doc.title}",
                metadata={**base_metadata, 'section': 'lessons'},
                doc_idx=doc_idx,
                chunk_counter=chunk_counter,
                chunk_type='note_section'
            )
            chunks.extend(lesson_chunks)
            chunk_counter += len(lesson_chunks)
        
        # Chunk 4: Quotes
        if doc.quotes:
            quotes_content = f"## Quotes from {doc.title}\n\n"
            for quote in doc.quotes[:10]:  # Limit quotes per chunk
                quotes_content += f'"{quote["text"]}"'
                if quote.get('timestamp'):
                    quotes_content += f" [{quote['timestamp']}]"
                quotes_content += "\n\n"
            
            if len(self.encoding.encode(quotes_content)) < self.notes_chunk_size * 2:
                chunk = self._create_chunk(
                    content=quotes_content,
                    metadata={**base_metadata, 'section': 'quotes'},
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type='quotes'
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        # Chunk 5: Main content (if not already covered)
        if doc.content and len(doc.content) > 500:
            content_chunks = self._chunk_by_tokens(
                text=doc.content,
                metadata={**base_metadata, 'section': 'content'},
                doc_idx=doc_idx,
                start_counter=chunk_counter
            )
            chunks.extend(content_chunks)
        
        return chunks
    
    def _chunk_transcript(self, doc, doc_idx: int) -> List[Chunk]:
        """Chunk a video transcript - handles various formats"""
        chunks = []
        chunk_counter = 0
        
        base_metadata = {
            'title': doc.title,
            'file_path': doc.file_path,
            'doc_type': 'transcript',
            'video_id': doc.video_id,
            'url': doc.url,
            'creator': doc.creator
        }
        
        # Check if this is a single-line massive transcript
        content_lines = doc.content.split('\n')
        is_single_line = len(content_lines) <= 2 and len(doc.content) > 1000
        
        if is_single_line:
            # Handle massive single-line transcripts by chunking on sentence boundaries
            chunks = self._chunk_single_line_transcript(
                doc.content,
                base_metadata,
                doc_idx
            )
        elif doc.timestamps and len(doc.timestamps) > 5:
            # We have good timestamp data
            chunks = self._chunk_by_timestamps(
                doc.timestamps,
                base_metadata,
                doc_idx,
                self.transcript_minutes
            )
        else:
            # Regular token-based chunking
            chunks = self._chunk_by_tokens(
                doc.content,
                base_metadata,
                doc_idx,
                0
            )
        
        return chunks
    
    def _chunk_by_timestamps(self, timestamps: List[Dict], metadata: Dict, 
                            doc_idx: int, minutes_per_chunk: int) -> List[Chunk]:
        """Chunk transcript by timestamp segments"""
        chunks = []
        chunk_counter = 0
        
        # Group timestamps into time windows
        current_chunk_text = ""
        current_start_time = None
        current_end_time = None
        
        for ts in timestamps:
            time_str = ts['time']
            text = ts['text']
            
            # Parse time to seconds
            time_parts = time_str.split(':')
            if len(time_parts) == 2:
                seconds = int(time_parts[0]) * 60 + int(time_parts[1])
            else:
                seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            
            # Start new chunk if needed
            if current_start_time is None:
                current_start_time = time_str
                current_chunk_text = f"[{time_str}] {text}"
            elif seconds // 60 >= (minutes_per_chunk * (chunk_counter + 1)):
                # Save current chunk
                chunk = self._create_chunk(
                    content=current_chunk_text,
                    metadata={
                        **metadata,
                        'start_time': current_start_time,
                        'end_time': current_end_time,
                        'section': 'transcript_segment'
                    },
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type='transcript_segment'
                )
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk with overlap
                current_start_time = time_str
                current_chunk_text = f"[{time_str}] {text}"
            else:
                current_chunk_text += f"\n[{time_str}] {text}"
                current_end_time = time_str
        
        # Add final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text,
                metadata={
                    **metadata,
                    'start_time': current_start_time,
                    'end_time': current_end_time,
                    'section': 'transcript_segment'
                },
                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                chunk_type='transcript_segment'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_tokens(self, text: str, metadata: Dict, doc_idx: int, 
                        start_counter: int = 0) -> List[Chunk]:
        """Chunk text by token count"""
        chunks = []
        chunk_counter = start_counter
        
        # Split into paragraphs first (if they exist)
        paragraphs = text.split('\n\n')
        
        # If no real paragraphs, try single newlines
        if len(paragraphs) == 1 and len(text) > 2000:
            paragraphs = text.split('\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            para_tokens = len(self.encoding.encode(para))
            
            # If paragraph is too large, split it
            if para_tokens > self.notes_chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        metadata=metadata,
                        chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                        chunk_type='text'
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    sent_tokens = len(self.encoding.encode(sent))
                    if current_tokens + sent_tokens > self.notes_chunk_size:
                        if current_chunk:
                            chunk = self._create_chunk(
                                content=current_chunk,
                                metadata=metadata,
                                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                                chunk_type='text'
                            )
                            chunks.append(chunk)
                            chunk_counter += 1
                        current_chunk = sent
                        current_tokens = sent_tokens
                    else:
                        if current_chunk:
                            current_chunk += " " + sent
                        else:
                            current_chunk = sent
                        current_tokens += sent_tokens
            
            # Add paragraph to current chunk
            elif current_tokens + para_tokens > self.notes_chunk_size:
                # Save current chunk
                chunk = self._create_chunk(
                    content=current_chunk,
                    metadata=metadata,
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type='text'
                )
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + para if overlap_text else para
                current_tokens = len(self.encoding.encode(current_chunk))
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                metadata=metadata,
                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                chunk_type='text'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_single_line_transcript(self, text: str, metadata: Dict, doc_idx: int) -> List[Chunk]:
        """Chunk a massive single-line transcript intelligently"""
        chunks = []
        chunk_counter = 0
        
        # Split by sentence boundaries more intelligently
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        target_size = self.notes_chunk_size
        
        for i, sent in enumerate(sentences):
            sent_tokens = len(self.encoding.encode(sent))
            
            # Check if adding this sentence exceeds our target
            if current_tokens + sent_tokens > target_size and current_chunk:
                # Create chunk with current content
                chunk = self._create_chunk(
                    content=current_chunk,
                    metadata={**metadata, 'chunk_index': chunk_counter},
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type='transcript_segment'
                )
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk with overlap (last 2-3 sentences)
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sent if overlap_text else sent
                current_tokens = len(self.encoding.encode(current_chunk))
            else:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
                current_tokens += sent_tokens
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                metadata={**metadata, 'chunk_index': chunk_counter},
                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                chunk_type='transcript_segment'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences more intelligently"""
        # Handle common sentence endings
        # Simple regex that handles most cases
        import re
        
        # Replace common abbreviations to avoid false splits
        text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\. ', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|Co)\. ', r'\1<DOT> ', text)
        text = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\. ', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_tokens: int = 200) -> str:
        """Get the last portion of text for overlap"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return ""
        
        # Take last 2-3 sentences or last 200 tokens worth
        overlap = ""
        for sent in reversed(sentences[-3:]):
            test_overlap = sent + " " + overlap if overlap else sent
            if len(self.encoding.encode(test_overlap)) <= overlap_tokens:
                overlap = test_overlap
            else:
                break
        
        return overlap.strip()
    
    def _chunk_list_content(self, items: List[str], title: str, metadata: Dict,
                          doc_idx: int, chunk_counter: int, chunk_type: str) -> List[Chunk]:
        """Chunk list content (lessons, action items, etc.)"""
        chunks = []
        current_content = f"## {title}\n\n"
        current_tokens = len(self.encoding.encode(current_content))
        
        for item in items:
            item_text = f"• {item}\n\n"
            item_tokens = len(self.encoding.encode(item_text))
            
            if current_tokens + item_tokens > self.notes_chunk_size and current_content != f"## {title}\n\n":
                # Create chunk
                chunk = self._create_chunk(
                    content=current_content,
                    metadata=metadata,
                    chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                    chunk_type=chunk_type
                )
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk
                current_content = f"## {title} (continued)\n\n{item_text}"
                current_tokens = len(self.encoding.encode(current_content))
            else:
                current_content += item_text
                current_tokens += item_tokens
        
        # Add final chunk
        if current_content != f"## {title}\n\n":
            chunk = self._create_chunk(
                content=current_content,
                metadata=metadata,
                chunk_id=f"doc{doc_idx:03d}_chunk{chunk_counter:03d}",
                chunk_type=chunk_type
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_generic(self, doc, doc_idx: int) -> List[Chunk]:
        """Generic chunking for unknown document types"""
        return self._chunk_by_tokens(
            doc.content,
            {'title': doc.title, 'file_path': doc.file_path},
            doc_idx
        )
    
    def _create_chunk(self, content: str, metadata: Dict, chunk_id: str, chunk_type: str) -> Chunk:
        """Create a chunk with metadata"""
        token_count = len(self.encoding.encode(content))
        
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            token_count=token_count,
            chunk_type=chunk_type
        )


if __name__ == "__main__":
    # Test the chunker
    from lib.processor import DocumentProcessor
    
    processor = DocumentProcessor()
    documents = processor.process_all()
    
    if documents:
        chunker = SmartChunker()
        chunks = chunker.chunk_documents(documents[:5])  # Test with first 5
        
        print(f"\n📊 Chunking Statistics:")
        print(f"✅ Documents processed: {min(5, len(documents))}")
        print(f"📦 Total chunks created: {len(chunks)}")
        
        if chunks:
            avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
            print(f"📏 Average chunk size: {avg_tokens:.0f} tokens")
            
            # Show chunk types
            types = {}
            for chunk in chunks:
                types[chunk.chunk_type] = types.get(chunk.chunk_type, 0) + 1
            
            print("\n📑 Chunk Types:")
            for typ, count in types.items():
                print(f"  - {typ}: {count} chunks")
