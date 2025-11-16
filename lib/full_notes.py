#!/usr/bin/env python3
"""
Full Notes Reader
Search for and read complete source notes or transcripts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
import frontmatter

# Silence stdout logging when running under MCP by redirecting rich to stderr
SILENT = os.getenv('WIKI_VAULT_SILENT') == '1'
console = Console(file=sys.stderr) if SILENT else Console()


class FullNotesReader:
    """Read complete source notes or transcripts"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        config_path = os.getenv('WIKI_VAULT_CONFIG', config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get directories (be tolerant if transcripts are not configured)
        data_sources = self.config.get('data_sources', {})
        # Notes directory with a sensible default
        self.notes_dir = Path(data_sources.get('source_notes_dir', 'md-vault'))
        # Transcripts are optional; only set if provided
        transcripts_dir_cfg = data_sources.get('transcripts_dir')
        self.transcripts_dir = Path(transcripts_dir_cfg) if transcripts_dir_cfg else None
        
        # Index all documents
        self.documents = self._index_documents()
        
        console.print(f"[cyan]📚 Found {len(self.documents)} documents[/cyan]")
    
    def _index_documents(self):
        """Index all markdown files"""
        documents = []
        
        # Index source notes
        if self.notes_dir.exists():
            ignore_dirs = {"venv", ".venv", "site-packages", "__pycache__", "node_modules", "chroma_db", ".git", ".hg", ".idea"}
            note_paths = list(self.notes_dir.rglob("*.md")) + list(self.notes_dir.rglob("*.txt"))
            for file_path in note_paths:
                parts = set(part.lower() for part in file_path.parts)
                if parts & {d.lower() for d in ignore_dirs}:
                    continue
                doc = self._parse_document(file_path, 'source_note')
                if doc:
                    documents.append(doc)
        
        # Index transcripts (if configured)
        if self.transcripts_dir and self.transcripts_dir.exists():
            ignore_dirs = {"venv", ".venv", "site-packages", "__pycache__", "node_modules", "chroma_db", ".git", ".hg", ".idea"}
            tx_paths = list(self.transcripts_dir.rglob("*.md")) + list(self.transcripts_dir.rglob("*.txt"))
            for file_path in tx_paths:
                parts = set(part.lower() for part in file_path.parts)
                if parts & {d.lower() for d in ignore_dirs}:
                    continue
                doc = self._parse_document(file_path, 'transcript')
                if doc:
                    documents.append(doc)
        
        # Sort safely even if some titles are missing/None
        return sorted(documents, key=lambda x: (x.get('title') or '').lower())
    
    def _parse_document(self, file_path: Path, doc_type: str):
        """Parse a markdown document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            title = post.metadata.get('title') or file_path.stem
            url = post.metadata.get('url') or ''
            video_id = post.metadata.get('video_id') or ''
            content_preview = post.content[:200] if post.content else ''
            
            return {
                'title': str(title),
                'file_path': str(file_path),
                'doc_type': doc_type,
                'url': str(url),
                'video_id': str(video_id),
                'content_preview': str(content_preview)
            }
        except:
            return None
    
    def search_documents(self, query: str):
        """Search for documents by title"""
        query_lower = query.lower()
        matches = []
        
        for doc in self.documents:
            if query_lower in doc['title'].lower():
                matches.append(doc)
        
        return matches
    
    def list_all(self, doc_type: str = None):
        """List all documents"""
        table = Table(title="Available Documents", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Title", style="white")
        table.add_column("Type", style="blue", width=12)
        table.add_column("Preview", style="dim")
        
        docs = self.documents
        if doc_type:
            docs = [d for d in docs if d['doc_type'] == doc_type]
        
        for i, doc in enumerate(docs, 1):
            preview = doc['content_preview'].replace('\n', ' ')[:50] + "..."
            table.add_row(
                str(i),
                doc['title'][:50],
                doc['doc_type'],
                preview
            )
        
        console.print(table)
        return docs
    
    def read_document(self, index: int = None, title: str = None):
        """Read a complete document"""
        
        # Find document
        if index is not None and 0 < index <= len(self.documents):
            doc = self.documents[index - 1]
        elif title:
            matches = self.search_documents(title)
            if not matches:
                console.print(f"[red]No document found matching '{title}'[/red]")
                return
            doc = matches[0]
        else:
            console.print("[red]Please specify a document index or title[/red]")
            return
        
        # Read full content
        try:
            with open(doc['file_path'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Display
            console.print(Panel(
                f"[bold]{doc['title']}[/bold]\n"
                f"Type: {doc['doc_type']} | File: {Path(doc['file_path']).name}",
                title="📄 Document Info",
                expand=False
            ))
            
            if doc.get('url'):
                console.print(f"[blue]URL: {doc['url']}[/blue]")
            
            console.print("\n" + "="*80 + "\n")
            
            # Display content as markdown
            md = Markdown(content)
            console.print(md)
            
            console.print("\n" + "="*80)
            console.print(f"[dim]End of document: {doc['title']}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error reading document: {e}[/red]")


def interactive_mode():
    """Interactive document reader"""
    
    reader = FullNotesReader()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    kb_name = config['knowledge_base']['name']
    
    console.print(Panel(
        f"[bold cyan]{kb_name}[/bold cyan]\n"
        f"Read complete source documents",
        title="📖 Full Notes Reader",
        expand=False
    ))
    
    console.print("\n[dim]Commands:[/dim]")
    console.print("  • [cyan]list[/cyan] - Show all documents")
    console.print("  • [cyan]search [query][/cyan] - Search by title")
    console.print("  • [cyan]read [number][/cyan] - Read document by number")
    console.print("  • [cyan]read [title][/cyan] - Read document by title")
    console.print("  • [cyan]notes[/cyan] - List only source notes")
    if reader.transcripts_dir:
        console.print("  • [cyan]transcripts[/cyan] - List only transcripts")
    console.print("  • [cyan]quit[/cyan] - Exit")
    console.print()
    
    current_list = None
    
    while True:
        try:
            command = console.input("[bold]> [/bold]").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit']:
                console.print("[green]Goodbye![/green]")
                break
            
            elif command.lower() == 'list':
                current_list = reader.list_all()
            
            elif command.lower() == 'notes':
                current_list = reader.list_all('source_note')
            
            elif command.lower() == 'transcripts':
                if reader.transcripts_dir:
                    current_list = reader.list_all('transcript')
                else:
                    console.print("[yellow]Transcripts are not configured in config.yaml[/yellow]")
            
            elif command.lower().startswith('search '):
                query = command[7:]
                matches = reader.search_documents(query)
                
                if matches:
                    table = Table(title=f"Search Results for '{query}'")
                    table.add_column("#", style="cyan")
                    table.add_column("Title", style="white")
                    table.add_column("Type", style="blue")
                    
                    for i, doc in enumerate(matches, 1):
                        table.add_row(str(i), doc['title'], doc['doc_type'])
                    
                    console.print(table)
                    current_list = matches
                else:
                    console.print(f"[yellow]No matches found for '{query}'[/yellow]")
            
            elif command.lower().startswith('read '):
                arg = command[5:].strip()
                
                # Try as number first
                try:
                    index = int(arg)
                    if current_list:
                        # Read from current list
                        if 0 < index <= len(current_list):
                            doc = current_list[index - 1]
                            # Find in full list
                            full_index = reader.documents.index(doc) + 1
                            reader.read_document(index=full_index)
                        else:
                            console.print("[red]Invalid document number[/red]")
                    else:
                        # Read from full list
                        reader.read_document(index=index)
                except ValueError:
                    # Try as title
                    reader.read_document(title=arg)
            
            else:
                console.print(f"[yellow]Unknown command: {command}[/yellow]")
                console.print("[dim]Type 'help' for commands[/dim]")
            
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Direct mode - search or read
        arg = " ".join(sys.argv[1:])
        reader = FullNotesReader()
        
        # Try as number
        try:
            index = int(arg)
            reader.read_document(index=index)
        except ValueError:
            # Search by title
            matches = reader.search_documents(arg)
            
            if len(matches) == 1:
                # Read single match
                reader.read_document(title=matches[0]['title'])
            elif len(matches) > 1:
                # Show matches
                console.print(f"[yellow]Multiple matches for '{arg}':[/yellow]")
                for i, doc in enumerate(matches, 1):
                    console.print(f"  {i}. {doc['title']} ({doc['doc_type']})")
                console.print("\n[dim]Run 'python scripts/full_notes.py' for interactive mode[/dim]")
            else:
                console.print(f"[red]No document found matching '{arg}'[/red]")
    else:
        # Interactive mode
        interactive_mode()
