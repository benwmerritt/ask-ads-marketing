"""
Universal Document Processor
Works with any YouTube content - source notes or transcripts
"""

import os
import yaml
import frontmatter
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Universal document structure"""
    # Metadata
    title: str
    file_path: str
    doc_type: str  # 'source_note' or 'transcript'
    
    # Optional metadata
    video_id: Optional[str] = None
    url: Optional[str] = None
    creator: Optional[str] = None
    date: Optional[str] = None
    
    # Content sections (may be empty depending on type)
    summary: str = ""
    content: str = ""
    concepts: List[Dict] = field(default_factory=list)
    quotes: List[Dict] = field(default_factory=list) 
    lessons: List[str] = field(default_factory=list)
    timestamps: List[Dict] = field(default_factory=list)
    

class DocumentProcessor:
    """Process any markdown documents into structured format"""
    
    def __init__(self, config_path: str = "config.yaml"):
        import os as _os
        config_path = _os.getenv('WIKI_VAULT_CONFIG', config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.source_dir = Path(self.config['data_sources']['source_notes_dir'])
        self.transcript_dir = Path(self.config['data_sources'].get('transcripts_dir', 'md-transcripts'))
        self.documents: List[Document] = []
        
    def process_all(self) -> List[Document]:
        """Process all available documents"""
        self.documents = []
        
        # Process source notes if enabled
        if self.config['data_sources']['use_source_notes'] and self.source_dir.exists():
            notes = self._process_source_notes()
            self.documents.extend(notes)
            logger.info(f"Processed {len(notes)} source notes")
        
        # Process transcripts if enabled  
        if self.config['data_sources']['use_transcripts'] and self.transcript_dir.exists():
            transcripts = self._process_transcripts()
            self.documents.extend(transcripts)
            logger.info(f"Processed {len(transcripts)} transcripts")
            
        logger.info(f"Total documents: {len(self.documents)}")
        return self.documents
    
    def _process_source_notes(self) -> List[Document]:
        """Process structured source notes - recursively scan all subdirectories"""
        documents = []
        md_files = list(self.source_dir.rglob("*.md"))
        
        for file_path in tqdm(md_files, desc="Processing source notes"):
            try:
                doc = self._parse_source_note(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        return documents
    
    def _parse_source_note(self, file_path: Path) -> Optional[Document]:
        """Parse a single source note - flexible format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            # Handle flexible metadata fields
            metadata = post.metadata
            
            # Prefer explicit title; fallback to filename stem if missing/None
            title_val = metadata.get('title') or file_path.stem
            doc = Document(
                title=title_val,
                file_path=str(file_path),
                doc_type='source_note',
                video_id=metadata.get('video_id'),
                url=metadata.get('url') or metadata.get('video_url'),
                creator=metadata.get('channel') or metadata.get('creator', self.config['knowledge_base']['creator']),
                date=str(metadata.get('date_processed') or metadata.get('source_note_date', ''))
            )
            
            content = post.content
            
            # Try to extract sections but don't fail if they don't exist
            doc.summary = self._extract_section(content, ["Executive Summary", "Summary", "Overview", "Description"])
            doc.content = content  # Store full content regardless
            
            # Extract structured data if present (won't fail if not)
            doc.concepts = self._extract_concepts(content)
            doc.quotes = self._extract_quotes(content)
            doc.lessons = self._extract_list_section(content, ["Key Lessons", "Takeaways", "Key Points", "Notes"])
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def _process_transcripts(self) -> List[Document]:
        """Process video transcripts (.md and .txt supported), ignoring non-content dirs like venv."""
        documents = []
        # Support both Markdown and plain text transcripts
        paths = []
        try:
            paths.extend(self.transcript_dir.rglob("*.md"))
            paths.extend(self.transcript_dir.rglob("*.txt"))
        except Exception:
            pass
        # Ignore files inside these directories
        ignore_dirs = {"venv", ".venv", "site-packages", "__pycache__", "node_modules", "chroma_db", ".git", ".hg", ".idea"}
        filtered_paths = []
        for p in paths:
            parts = set(part.lower() for part in p.parts)
            if parts & {d.lower() for d in ignore_dirs}:
                continue
            filtered_paths.append(p)
        md_files = list(filtered_paths)
        
        for file_path in tqdm(md_files, desc="Processing transcripts"):
            try:
                doc = self._parse_transcript(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        return documents
    
    def _parse_transcript(self, file_path: Path) -> Optional[Document]:
        """Parse a video transcript - handles multiple formats"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            # Handle flexible metadata fields
            metadata = post.metadata
            
            # Prefer explicit title; fallback to filename stem if missing/None
            title_val = metadata.get('title') or file_path.stem
            doc = Document(
                title=title_val,
                file_path=str(file_path),
                doc_type='transcript',
                video_id=metadata.get('video_id'),
                url=metadata.get('url') or metadata.get('video_url'),
                creator=metadata.get('channel') or metadata.get('creator', self.config['knowledge_base']['creator']),
                date=str(metadata.get('date_processed') or metadata.get('transcript_date', ''))
            )
            
            # Store description from metadata if available
            if metadata.get('description'):
                doc.summary = metadata.get('description')[:500]  # Limit length
            
            content = post.content
            
            # Try to extract transcript section, but use full content if not found
            transcript_section = self._extract_section(content, ["Full Transcript", "Transcript"])
            if transcript_section:
                # Handle single-line massive transcripts (common format)
                if '\n' not in transcript_section.strip():
                    # It's one massive line - we'll handle this in chunking
                    doc.content = transcript_section
                else:
                    doc.content = transcript_section
                    doc.timestamps = self._extract_timestamps(transcript_section)
            else:
                # No transcript section header - assume entire content is transcript
                doc.content = content
                doc.timestamps = self._extract_timestamps(content)
            
            # Try to extract quotes but don't fail
            doc.quotes = self._extract_inline_quotes(doc.content)
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def _extract_section(self, content: str, section_names: List[str]) -> str:
        """Extract a section by trying multiple possible names"""
        for name in section_names:
            pattern = rf"#+ {name}\n+(.*?)(?=\n#+ |\Z)"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_list_section(self, content: str, section_names: List[str]) -> List[str]:
        """Extract a list section"""
        section_text = self._extract_section(content, section_names)
        if not section_text:
            return []
        
        # Extract list items (numbered or bulleted)
        items = re.findall(
            r'^\d+\.\s+(.+?)(?=^\d+\.|\n\n|^[\*\-]|$)|^[\*\-]\s+(.+?)(?=^[\*\-]|\n\n|^\d+\.|$)',
            section_text, 
            re.MULTILINE | re.DOTALL
        )
        
        cleaned_items = []
        for item in items:
            text = item[0] if item[0] else item[1]
            if text:
                text = ' '.join(text.split())[:500]  # Clean and limit length
                cleaned_items.append(text.strip())
        
        return cleaned_items
    
    def _extract_concepts(self, content: str) -> List[Dict]:
        """Extract concepts/frameworks from content"""
        concepts = []
        
        # Pattern 1: Marked concepts sections
        concept_section = self._extract_section(content, ["Core Concepts", "Concepts", "Key Concepts", "Frameworks"])
        if concept_section:
            # Look for concept definitions
            pattern = r"(?:###\s+)?(?:Concept \d+:\s+)?([^:\n]+)(?:\n+\*\*Definition\*\*:\s+(.+?)(?=\n\n|\*\*|###|$))?}"
            matches = re.finditer(pattern, concept_section, re.DOTALL)
            
            for match in matches:
                name = match.group(1).strip()
                definition = match.group(2).strip() if match.group(2) else ""
                if name:
                    concepts.append({
                        'name': name,
                        'definition': definition
                    })
        
        # Pattern 2: Wiki-style links (common in notes)
        wiki_links = re.findall(r'\[\[([^\]]+)\]\]', content)
        for link in wiki_links:
            if link and link not in [c['name'] for c in concepts]:
                concepts.append({'name': link, 'definition': ''})
        
        return concepts[:20]  # Limit to top 20
    
    def _extract_quotes(self, content: str) -> List[Dict]:
        """Extract quotes from content"""
        quotes = []
        
        # Pattern 1: Markdown blockquotes
        pattern1 = r'>\s*"([^"]+)"(?:\s*\[([^\]]+)\])?'
        matches1 = re.findall(pattern1, content)
        for text, timestamp in matches1:
            quotes.append({
                'text': text.strip(),
                'timestamp': timestamp.strip() if timestamp else None
            })
        
        # Pattern 2: Quoted text in quotes section
        quotes_section = self._extract_section(content, ["Notable Quotes", "Quotes", "Key Quotes"])
        if quotes_section:
            pattern2 = r'"([^"]+)"'
            matches2 = re.findall(pattern2, quotes_section)
            for text in matches2:
                if text and len(text) > 20:  # Filter short matches
                    quotes.append({'text': text.strip(), 'timestamp': None})
        
        return quotes[:30]  # Limit to top 30
    
    def _extract_timestamps(self, content: str) -> List[Dict]:
        """Extract timestamp markers from transcript"""
        timestamps = []
        
        # Multiple timestamp patterns to handle different formats
        patterns = [
            r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)(?=\[|$)',  # [MM:SS] or [HH:MM:SS]
            r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(.+?)(?=\d{1,2}:\d{2}|$)',  # MM:SS - text
            r'\((\d{1,2}:\d{2}(?::\d{2})?)\)\s*(.+?)(?=\(|$)',  # (MM:SS) text
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                for timestamp, text in matches:
                    timestamps.append({
                        'time': timestamp,
                        'text': text.strip()[:200]  # First 200 chars
                    })
                break  # Use first matching pattern
        
        return timestamps
    
    def _extract_inline_quotes(self, content: str) -> List[Dict]:
        """Extract notable quotes from transcript text"""
        quotes = []
        
        # Look for quoted speech in transcripts
        # Pattern: "quote text" possibly followed by timestamp
        pattern = r'"([^"]{30,500})"(?:\s*\[([^\]]+)\])?'
        matches = re.findall(pattern, content)
        
        for text, timestamp in matches[:20]:  # Limit to 20 quotes
            # Filter out likely false positives
            if not any(skip in text.lower() for skip in ['http', 'www', 'click', 'subscribe']):
                quotes.append({
                    'text': text.strip(),
                    'timestamp': timestamp.strip() if timestamp else None
                })
        
        return quotes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not self.documents:
            return {}
        
        source_notes = [d for d in self.documents if d.doc_type == 'source_note']
        transcripts = [d for d in self.documents if d.doc_type == 'transcript']
        
        # Aggregate concepts
        all_concepts = {}
        for doc in self.documents:
            for concept in doc.concepts:
                name = concept.get('name', '')
                if name:
                    all_concepts[name] = all_concepts.get(name, 0) + 1
        
        top_concepts = sorted(all_concepts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_documents': len(self.documents),
            'source_notes': len(source_notes),
            'transcripts': len(transcripts),
            'total_concepts': len(all_concepts),
            'total_quotes': sum(len(d.quotes) for d in self.documents),
            'top_concepts': top_concepts
        }


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    documents = processor.process_all()
    
    if documents:
        stats = processor.get_statistics()
        print("\n📊 Processing Statistics:")
        print(f"✅ Total documents: {stats['total_documents']}")
        print(f"📝 Source notes: {stats['source_notes']}")
        print(f"📹 Transcripts: {stats['transcripts']}")
        print(f"💡 Unique concepts: {stats['total_concepts']}")
        print(f"💬 Total quotes: {stats['total_quotes']}")
        
        if stats['top_concepts']:
            print("\n🔝 Top Concepts:")
            for concept, count in stats['top_concepts']:
                print(f"  - {concept}: {count} mentions")
