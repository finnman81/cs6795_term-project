"""Embeddings and vector store management module."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import faiss
from openai import OpenAI

from src.config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY
from src.ingestion import Chunk


class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""
    
    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score


def load_chunks(processed_path: Path) -> List[Chunk]:
    """
    Load chunks from processed JSONL file.
    
    Args:
        processed_path: Path to processed_docs.jsonl
        
    Returns:
        List of Chunk objects
    """
    chunks = []
    processed_path = Path(processed_path)
    
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed documents not found: {processed_path}")
    
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunks.append(Chunk.from_dict(data))
    
    return chunks


def embed_chunks(chunks: List[Chunk], config) -> np.ndarray:
    """
    Generate embeddings for chunks using OpenAI API.
    
    Args:
        chunks: List of Chunk objects
        config: Configuration object with OPENAI_API_KEY and EMBEDDING_MODEL_NAME
        
    Returns:
        numpy array of embeddings (n_chunks, embedding_dim)
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    texts = [chunk.text for chunk in chunks]
    embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=batch_texts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings, dtype=np.float32)


def build_vector_store(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    config
) -> None:
    """
    Build FAISS vector store and persist to disk.
    
    Args:
        chunks: List of Chunk objects
        embeddings: numpy array of embeddings
        config: Configuration object with VECTOR_STORE_PATH
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")
    
    embedding_dim = embeddings.shape[1]
    
    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    
    # Save index
    vector_store_path = Path(config.VECTOR_STORE_PATH)
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vector_store_path))
    
    # Save chunk metadata mapping (index -> chunk)
    metadata_path = vector_store_path.parent / (vector_store_path.name + '.metadata.json')
    metadata = [
        {
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text
        }
        for chunk in chunks
    ]
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_vector_store(config) -> Tuple[faiss.Index, List[dict]]:
    """
    Load FAISS vector store and metadata from disk.
    
    Args:
        config: Configuration object with VECTOR_STORE_PATH
        
    Returns:
        Tuple of (FAISS index, chunk metadata list)
    """
    vector_store_path = Path(config.VECTOR_STORE_PATH)
    
    if not vector_store_path.exists():
        raise FileNotFoundError(
            f"Vector store not found: {vector_store_path}. "
            "Please run scripts/build_index.py first."
        )
    
    # Load index
    index = faiss.read_index(str(vector_store_path))
    
    # Load metadata
    metadata_path = vector_store_path.parent / (vector_store_path.name + '.metadata.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return index, metadata


def retrieve(
    query: str,
    vector_store: Tuple[faiss.Index, List[dict]],
    config,
    top_k: Optional[int] = None
) -> List[RetrievedChunk]:
    """
    Retrieve top-k most similar chunks for a query.
    
    Args:
        query: Query string
        vector_store: Tuple of (FAISS index, chunk metadata)
        config: Configuration object with OPENAI_API_KEY and EMBEDDING_MODEL_NAME
        top_k: Number of chunks to retrieve (defaults to config.TOP_K_RETRIEVAL)
        
    Returns:
        List of RetrievedChunk objects sorted by similarity (highest first)
    """
    if top_k is None:
        top_k = config.TOP_K_RETRIEVAL
    
    index, metadata = vector_store
    
    # Embed query
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=[query]
    )
    query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
    
    # Search
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, k)
    
    # Build results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata):
            chunk_data = metadata[idx]
            chunk = Chunk(
                doc_id=chunk_data["doc_id"],
                title=chunk_data["title"],
                chunk_index=chunk_data["chunk_index"],
                text=chunk_data["text"]
            )
            # Convert L2 distance to similarity score (lower distance = higher similarity)
            score = 1.0 / (1.0 + distance)
            results.append(RetrievedChunk(chunk=chunk, score=score))
    
    return results

