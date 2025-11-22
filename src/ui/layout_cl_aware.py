"""Cognitive-load-aware interface layout (structured, lower load)."""

import streamlit as st
from typing import List
from src.embeddings import RetrievedChunk


def render_cl_aware_view(
    query: str,
    answer: str,
    retrieved_chunks: List[RetrievedChunk]
) -> None:
    """
    Render cognitive-load-aware view with structured sections.
    
    Args:
        query: User query
        answer: LLM answer (may contain markdown structure)
        retrieved_chunks: List of retrieved chunks
    """
    st.subheader("Answer")
    
    # Display answer (may contain markdown structure from prompt)
    st.markdown(answer)
    
    # Display retrieved evidence in collapsible section with better formatting
    if retrieved_chunks:
        with st.expander("📚 Source Information", expanded=False):
            st.markdown("### Documents Used")
            for i, retrieved in enumerate(retrieved_chunks, 1):
                chunk = retrieved.chunk
                st.markdown(f"**{i}. {chunk.title}**")
                st.caption(f"Relevance: {retrieved.score:.1%}")
                st.markdown(f"*{chunk.text[:200]}...*" if len(chunk.text) > 200 else f"*{chunk.text}*")
                if i < len(retrieved_chunks):
                    st.markdown("---")

