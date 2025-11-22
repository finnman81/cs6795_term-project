"""Configuration module for the RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model configuration
MODEL_NAME = "gpt-4-turbo-preview"  # OpenAI model for generation
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # OpenAI embedding model

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR
PROCESSED_DOCS_PATH = DATA_DIR / "processed_docs.jsonl"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.faiss"

# Chunking parameters
MAX_CHUNK_TOKENS = 600
CHUNK_OVERLAP = 100

# Retrieval parameters
TOP_K_RETRIEVAL = 5

# Logging
LOG_FILE_PATH = PROJECT_ROOT / "experiments" / "results_raw.csv"

# API configuration
# Try Streamlit secrets first (for cloud deployment), then environment variable (for local)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except (ImportError, AttributeError):
    # Not in Streamlit context or secrets not available, use environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "For local: set it in your .env file or environment variables. "
        "For Streamlit Cloud: set it in the app's secrets (Settings > Secrets)."
    )

