"""Tests for RAG pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src import rag_pipeline
from src import ingestion
from src import embeddings


def test_format_context():
    """Test context formatting."""
    # Create mock retrieved chunks
    chunk1 = ingestion.Chunk("doc1", "Doc 1", 0, "First chunk text")
    chunk2 = ingestion.Chunk("doc2", "Doc 2", 0, "Second chunk text")
    
    retrieved_chunks = [
        embeddings.RetrievedChunk(chunk1, 0.9),
        embeddings.RetrievedChunk(chunk2, 0.8)
    ]
    
    context = rag_pipeline.format_context(retrieved_chunks)
    
    assert "First chunk text" in context
    assert "Second chunk text" in context
    assert "Doc 1" in context
    assert "Doc 2" in context


def test_build_prompt_baseline():
    """Test baseline prompt building."""
    query = "How to reduce phone use?"
    context = "Context about phone use reduction."
    
    mock_config = Mock()
    prompt = rag_pipeline.build_prompt(query, context, "baseline", mock_config)
    
    assert query in prompt
    assert context in prompt
    assert "comprehensive answer" in prompt.lower()


def test_build_prompt_cl_aware():
    """Test cognitive-load-aware prompt building."""
    query = "How to reduce phone use?"
    context = "Context about phone use reduction."
    
    mock_config = Mock()
    prompt = rag_pipeline.build_prompt(query, context, "cl_aware", mock_config)
    
    assert query in prompt
    assert context in prompt
    assert "Key Strategies" in prompt
    assert "Try This Today" in prompt


def test_build_prompt_invalid_condition():
    """Test that invalid condition raises error."""
    mock_config = Mock()
    
    with pytest.raises(ValueError, match="Unknown condition"):
        rag_pipeline.build_prompt("query", "context", "invalid", mock_config)


@patch('src.rag_pipeline.OpenAI')
def test_call_llm(mock_openai):
    """Test LLM API call."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test answer."
    mock_client.chat.completions.create.return_value = mock_response
    
    # Mock config
    mock_config = Mock()
    mock_config.MODEL_NAME = "gpt-4-turbo-preview"
    mock_config.OPENAI_API_KEY = "test-key"
    
    # Call LLM
    response = rag_pipeline.call_llm("Test prompt", mock_config)
    
    assert response == "This is a test answer."
    mock_client.chat.completions.create.assert_called_once()


@patch('src.rag_pipeline.call_llm')
@patch('src.rag_pipeline.retrieve')
def test_answer_query(mock_retrieve, mock_call_llm):
    """Test end-to-end answer_query function."""
    # Mock retrieval
    chunk1 = ingestion.Chunk("doc1", "Doc 1", 0, "Test chunk")
    retrieved_chunks = [embeddings.RetrievedChunk(chunk1, 0.9)]
    mock_retrieve.return_value = retrieved_chunks
    
    # Mock LLM call
    mock_call_llm.return_value = "This is the answer."
    
    # Mock vector store and config
    mock_vector_store = ("index", "metadata")
    mock_config = Mock()
    
    # Test baseline condition
    result = rag_pipeline.answer_query(
        query="Test query",
        condition="baseline",
        vector_store=mock_vector_store,
        config=mock_config
    )
    
    assert result["answer"] == "This is the answer."
    assert result["query"] == "Test query"
    assert result["condition"] == "baseline"
    assert len(result["retrieved_chunks"]) == 1
    assert mock_retrieve.called
    assert mock_call_llm.called
    
    # Test cl_aware condition
    result = rag_pipeline.answer_query(
        query="Test query",
        condition="cl_aware",
        vector_store=mock_vector_store,
        config=mock_config
    )
    
    assert result["condition"] == "cl_aware"

