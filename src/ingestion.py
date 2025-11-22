"""Document ingestion and chunking module."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import tiktoken
import pypdf


class RawDocument:
    """Represents a raw document loaded from file."""
    
    def __init__(self, doc_id: str, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text


class Chunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(
        self,
        doc_id: str,
        title: str,
        chunk_index: int,
        text: str,
        start_char: int = 0,
        end_char: int = 0
    ):
        self.doc_id = doc_id
        self.title = title
        self.chunk_index = chunk_index
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0)
        )


def load_raw_documents(raw_dir: Path) -> List[RawDocument]:
    """
    Load raw text and PDF documents from directory.
    
    Args:
        raw_dir: Path to directory containing .txt and .pdf files
        
    Returns:
        List of RawDocument objects
    """
    raw_docs = []
    raw_dir = Path(raw_dir)
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")
    
    files = list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No .txt or .pdf files found in {raw_dir}")
    
    for file_path in sorted(files):
        doc_id = file_path.stem
        title = doc_id.replace("_", " ").title()
        text = ""
        
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file_path.suffix.lower() == '.pdf':
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            if text.strip():
                raw_docs.append(RawDocument(doc_id=doc_id, title=title, text=text))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return raw_docs


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_documents(
    raw_docs: List[RawDocument],
    max_tokens: int,
    overlap: int
) -> List[Chunk]:
    """
    Split documents into overlapping chunks.
    
    Args:
        raw_docs: List of RawDocument objects
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of Chunk objects
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4
    chunks = []
    
    for doc in raw_docs:
        cleaned_text = clean_text(doc.text)
        
        if not cleaned_text:
            continue
        
        # Tokenize the entire document
        tokens = encoding.encode(cleaned_text)
        
        if len(tokens) <= max_tokens:
            # Document fits in one chunk
            chunk = Chunk(
                doc_id=doc.doc_id,
                title=doc.title,
                chunk_index=0,
                text=cleaned_text,
                start_char=0,
                end_char=len(cleaned_text)
            )
            chunks.append(chunk)
            continue
        
        # Split into overlapping chunks
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = encoding.decode(chunk_tokens)
            
            # Find character positions in original text
            # Approximate by finding the substring
            start_char = cleaned_text.find(chunk_text[:50]) if len(chunk_text) > 50 else 0
            end_char = start_char + len(chunk_text)
            
            chunk = Chunk(
                doc_id=doc.doc_id,
                title=doc.title,
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=start_char,
                end_char=min(end_char, len(cleaned_text))
            )
            chunks.append(chunk)
            
            # Move start position forward, accounting for overlap
            start_idx = end_idx - overlap
            chunk_index += 1
            
            # Prevent infinite loop
            if start_idx >= len(tokens) - overlap:
                break
    
    return chunks


def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    """
    Save chunks to JSONL file.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk.to_dict(), f, ensure_ascii=False)
            f.write("\n")

