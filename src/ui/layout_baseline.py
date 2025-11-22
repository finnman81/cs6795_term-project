"""Baseline interface layout (higher cognitive load)."""

import streamlit as st
from typing import List
from src.embeddings import RetrievedChunk


def render_baseline_view(
    query: str,
    answer: str,
    retrieved_chunks: List[RetrievedChunk]
) -> None:
    """
    Render baseline view with dense paragraph and evidence block.
    
    Args:
        query: User query
        answer: LLM answer
        retrieved_chunks: List of retrieved chunks
    """
    st.subheader("Answer")
    
    # Display answer as long paragraph
    st.write(answer)
    
    # Display retrieved evidence in dense block
    if retrieved_chunks:
        st.subheader("Retrieved Evidence")
        with st.expander("View source documents", expanded=False):
            for i, retrieved in enumerate(retrieved_chunks, 1):
                chunk = retrieved.chunk
                st.markdown(f"**Document {i}: {chunk.title}** (Score: {retrieved.score:.3f})")
                st.text(chunk.text)
                st.markdown("---")

