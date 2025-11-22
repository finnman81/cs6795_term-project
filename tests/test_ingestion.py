"""Tests for document ingestion and chunking."""

import pytest
import tempfile
import json
from pathlib import Path
from src import ingestion
from src import config


def test_load_raw_documents():
    """Test loading raw documents from directory."""
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test files
        test_file1 = tmp_path / "test_doc1.txt"
        test_file1.write_text("This is a test document about digital wellbeing.")
        
        test_file2 = tmp_path / "test_doc2.txt"
        test_file2.write_text("Another document about focus and productivity.")
        
        # Load documents
        docs = ingestion.load_raw_documents(tmp_path)
        
        assert len(docs) == 2
        assert docs[0].doc_id == "test_doc1"
        assert docs[1].doc_id == "test_doc2"
        assert "digital wellbeing" in docs[0].text


def test_clean_text():
    """Test text cleaning function."""
    dirty_text = "This   has   multiple   spaces\n\n\nand   weird   characters."
    cleaned = ingestion.clean_text(dirty_text)
    
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.strip() == cleaned


def test_chunk_documents():
    """Test document chunking."""
    # Create test document
    doc = ingestion.RawDocument(
        doc_id="test",
        title="Test Document",
        text="This is a test document. " * 100  # Make it long enough to chunk
    )
    
    # Chunk with small max_tokens to force chunking
    chunks = ingestion.chunk_documents([doc], max_tokens=50, overlap=10)
    
    assert len(chunks) > 1
    assert all(chunk.doc_id == "test" for chunk in chunks)
    assert all(chunk.chunk_index >= 0 for chunk in chunks)
    
    # Check chunk indices are sequential
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == sorted(indices)


def test_save_and_load_chunks():
    """Test saving and loading chunks."""
    # Create test chunks
    chunks = [
        ingestion.Chunk(
            doc_id="test1",
            title="Test 1",
            chunk_index=0,
            text="Test chunk 1"
        ),
        ingestion.Chunk(
            doc_id="test2",
            title="Test 2",
            chunk_index=0,
            text="Test chunk 2"
        )
    ]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = Path(f.name)
    
    try:
        ingestion.save_chunks(chunks, tmp_path)
        
        # Load chunks
        loaded_chunks = []
        with open(tmp_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    loaded_chunks.append(ingestion.Chunk.from_dict(data))
        
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0].text == "Test chunk 1"
        assert loaded_chunks[1].text == "Test chunk 2"
    finally:
        tmp_path.unlink()


def test_chunk_metadata():
    """Test that chunk metadata is correct."""
    doc = ingestion.RawDocument(
        doc_id="metadata_test",
        title="Metadata Test",
        text="Short text."
    )
    
    chunks = ingestion.chunk_documents([doc], max_tokens=100, overlap=0)
    
    assert len(chunks) == 1
    assert chunks[0].doc_id == "metadata_test"
    assert chunks[0].title == "Metadata Test"
    assert chunks[0].chunk_index == 0

