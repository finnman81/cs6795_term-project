"""Tests for embeddings and retrieval."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src import embeddings
from src import ingestion
from src import config


def test_load_chunks():
    """Test loading chunks from JSONL file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = Path(f.name)
    
    try:
        # Create test JSONL file
        chunks_data = [
            {
                "doc_id": "test1",
                "title": "Test 1",
                "chunk_index": 0,
                "text": "This is about notifications and phone settings."
            },
            {
                "doc_id": "test2",
                "title": "Test 2",
                "chunk_index": 0,
                "text": "This is about sleep routines and bedtime habits."
            }
        ]
        
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for chunk_data in chunks_data:
                json.dump(chunk_data, f, ensure_ascii=False)
                f.write('\n')
        
        # Load chunks
        chunks = embeddings.load_chunks(tmp_path)
        
        assert len(chunks) == 2
        assert chunks[0].text == "This is about notifications and phone settings."
        assert chunks[1].text == "This is about sleep routines and bedtime habits."
    finally:
        tmp_path.unlink()


@patch('src.embeddings.OpenAI')
def test_embed_chunks(mock_openai):
    """Test embedding generation."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Mock embedding response
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),  # text-embedding-3-small dimension
        Mock(embedding=[0.2] * 1536)
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    # Create test chunks
    chunks = [
        ingestion.Chunk("test1", "Test 1", 0, "Test text 1"),
        ingestion.Chunk("test2", "Test 2", 0, "Test text 2")
    ]
    
    # Generate embeddings
    embeddings_array = embeddings.embed_chunks(chunks, config)
    
    assert embeddings_array.shape == (2, 1536)
    assert np.allclose(embeddings_array[0], [0.1] * 1536)
    assert np.allclose(embeddings_array[1], [0.2] * 1536)


def test_build_and_load_vector_store():
    """Test building and loading vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        vector_store_path = tmp_path / "test_index.faiss"
        
        # Create test chunks and embeddings
        chunks = [
            ingestion.Chunk("test1", "Test 1", 0, "Notifications and phone settings"),
            ingestion.Chunk("test2", "Test 2", 0, "Sleep routines and bedtime")
        ]
        
        # Create dummy embeddings (small dimension for testing)
        embeddings_array = np.random.rand(2, 10).astype(np.float32)
        
        # Mock config
        mock_config = Mock()
        mock_config.VECTOR_STORE_PATH = vector_store_path
        
        # Build vector store
        embeddings.build_vector_store(chunks, embeddings_array, mock_config)
        
        # Verify files exist
        assert vector_store_path.exists()
        metadata_path = vector_store_path.with_suffix('.metadata.json')
        assert metadata_path.exists()
        
        # Load vector store
        index, metadata = embeddings.load_vector_store(mock_config)
        
        assert index.ntotal == 2
        assert len(metadata) == 2
        assert metadata[0]["text"] == "Notifications and phone settings"


@patch('src.embeddings.OpenAI')
def test_retrieve(mock_openai):
    """Test retrieval functionality."""
    # Create test vector store
    import faiss
    
    embedding_dim = 10
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add test embeddings
    test_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "notifications"
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "sleep"
    ], dtype=np.float32)
    index.add(test_embeddings)
    
    metadata = [
        {
            "doc_id": "test1",
            "title": "Notifications Doc",
            "chunk_index": 0,
            "text": "This is about notifications and phone settings."
        },
        {
            "doc_id": "test2",
            "title": "Sleep Doc",
            "chunk_index": 0,
            "text": "This is about sleep routines and bedtime habits."
        }
    ]
    
    vector_store = (index, metadata)
    
    # Mock OpenAI for query embedding
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Mock query embedding (closer to first document)
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    mock_client.embeddings.create.return_value = mock_response
    
    # Mock config
    mock_config = Mock()
    mock_config.TOP_K_RETRIEVAL = 1
    
    # Retrieve
    results = embeddings.retrieve("notifications", vector_store, mock_config)
    
    assert len(results) == 1
    assert "notifications" in results[0].chunk.text.lower()
    assert results[0].score > 0

