"""Core RAG pipeline module."""

from typing import Dict, Any, List
from openai import OpenAI

from src.config import MODEL_NAME, OPENAI_API_KEY
from src.embeddings import retrieve, RetrievedChunk
from src.prompts import get_baseline_prompt, get_cl_aware_prompt


def format_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a single context string.
    
    Args:
        retrieved_chunks: List of RetrievedChunk objects
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, retrieved in enumerate(retrieved_chunks, 1):
        chunk = retrieved.chunk
        context_parts.append(
            f"[Document {i}: {chunk.title}]\n{chunk.text}\n"
        )
    
    return "\n".join(context_parts)


def build_prompt(
    query: str,
    context: str,
    condition: str,
    config
) -> str:
    """
    Build prompt based on condition (baseline or cl_aware).
    
    Args:
        query: User query
        context: Formatted context string
        condition: "baseline" or "cl_aware"
        config: Configuration object (unused but kept for consistency)
        
    Returns:
        Formatted prompt string
    """
    if condition == "baseline":
        return get_baseline_prompt(query, context)
    elif condition == "cl_aware":
        return get_cl_aware_prompt(query, context)
    else:
        raise ValueError(f"Unknown condition: {condition}. Must be 'baseline' or 'cl_aware'.")


def call_llm(prompt: str, config) -> str:
    """
    Call OpenAI LLM API.
    
    Args:
        prompt: Formatted prompt string
        config: Configuration object with MODEL_NAME and OPENAI_API_KEY
        
    Returns:
        LLM response text
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise RuntimeError(f"Error calling LLM API: {str(e)}") from e


def answer_query(
    query: str,
    condition: str,
    vector_store,
    config
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline: retrieve context, build prompt, call LLM.
    
    Args:
        query: User query
        condition: "baseline" or "cl_aware"
        vector_store: Vector store tuple from embeddings.load_vector_store
        config: Configuration object
        
    Returns:
        Dictionary with:
            - "answer": LLM response
            - "retrieved_chunks": List of RetrievedChunk objects
            - "condition": Condition used
            - "query": Original query
    """
    # Step 1: Retrieve context
    retrieved_chunks = retrieve(query, vector_store, config)
    
    # Step 2: Format context
    context = format_context(retrieved_chunks)
    
    # Step 3: Build prompt for given condition
    prompt = build_prompt(query, context, condition, config)
    
    # Step 4: Call LLM
    answer = call_llm(prompt, config)
    
    # Step 5: Return results
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "condition": condition,
        "query": query
    }

