"""Integration tests for the full RAG pipeline."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
import tempfile
import json
import faiss

from src import ingestion
from src import embeddings
from src import rag_pipeline
from src import config

@pytest.fixture
def mock_openai():
    with patch('src.embeddings.OpenAI') as mock_emb, \
         patch('src.rag_pipeline.OpenAI') as mock_llm:
        
        # Mock Embedding Client
        mock_emb_client = Mock()
        mock_emb.return_value = mock_emb_client
        
        def create_embedding(model, input):
            # Return random embeddings
            n_inputs = len(input)
            return Mock(data=[Mock(embedding=np.random.rand(1536).tolist()) for _ in range(n_inputs)])
            
        mock_emb_client.embeddings.create.side_effect = create_embedding

        # Mock LLM Client
        mock_llm_client = Mock()
        mock_llm.return_value = mock_llm_client
        
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(message=Mock(content="This is a generated answer based on the context."))
        ]
        mock_llm_client.chat.completions.create.return_value = mock_completion
        
        yield mock_emb, mock_llm

def test_full_pipeline_flow(mock_openai):
    """Test the complete flow from ingestion to RAG response."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. Setup Mock Data
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        # Create a dummy PDF-like text file (since we tested PDF loading separately)
        with open(raw_dir / "doc1.txt", "w", encoding="utf-8") as f:
            f.write("This is a document about cognitive load theory.\nIt mentions intrinsic and extraneous load.")
            
        config.RAW_DOCS_DIR = raw_dir
        config.PROCESSED_DOCS_PATH = tmp_path / "processed.jsonl"
        config.VECTOR_STORE_PATH = tmp_path / "vector_store.faiss"
        
        # 2. Ingestion
        raw_docs = ingestion.load_raw_documents(config.RAW_DOCS_DIR)
        chunks = ingestion.chunk_documents(raw_docs, max_tokens=100, overlap=10)
        ingestion.save_chunks(chunks, config.PROCESSED_DOCS_PATH)
        
        assert len(chunks) > 0
        
        # 3. Indexing
        loaded_chunks = embeddings.load_chunks(config.PROCESSED_DOCS_PATH)
        embeddings_array = embeddings.embed_chunks(loaded_chunks, config)
        embeddings.build_vector_store(loaded_chunks, embeddings_array, config)
        
        assert config.VECTOR_STORE_PATH.exists()
        
        # 4. RAG Retrieval & Generation
        vector_store = embeddings.load_vector_store(config)
        
        # Test Baseline Condition
        result_baseline = rag_pipeline.answer_query(
            query="What is cognitive load?",
            condition="baseline",
            vector_store=vector_store,
            config=config
        )
        
        assert result_baseline["answer"] == "This is a generated answer based on the context."
        assert len(result_baseline["retrieved_chunks"]) > 0
        assert result_baseline["condition"] == "baseline"
        
        # Test CL-Aware Condition
        result_cl = rag_pipeline.answer_query(
            query="What is cognitive load?",
            condition="cl_aware",
            vector_store=vector_store,
            config=config
        )
        
        assert result_cl["answer"] == "This is a generated answer based on the context."
        assert result_cl["condition"] == "cl_aware"


