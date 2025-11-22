"""Script to build vector index from raw documents."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src import ingestion
from src import embeddings


def main():
    """Build processed documents and vector index."""
    print("Building vector index...")
    print(f"Raw documents directory: {config.RAW_DOCS_DIR}")
    print(f"Processed documents path: {config.PROCESSED_DOCS_PATH}")
    print(f"Vector store path: {config.VECTOR_STORE_PATH}")
    print()
    
    # Step 1: Load raw documents
    print("Step 1: Loading raw documents...")
    try:
        raw_docs = ingestion.load_raw_documents(config.RAW_DOCS_DIR)
        print(f"Loaded {len(raw_docs)} documents")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please add .txt files to {config.RAW_DOCS_DIR}")
        return 1
    
    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    chunks = ingestion.chunk_documents(
        raw_docs,
        max_tokens=config.MAX_CHUNK_TOKENS,
        overlap=config.CHUNK_OVERLAP
    )
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Save chunks
    print("Step 3: Saving processed chunks...")
    ingestion.save_chunks(chunks, config.PROCESSED_DOCS_PATH)
    print(f"Saved to {config.PROCESSED_DOCS_PATH}")
    
    # Step 4: Load chunks and generate embeddings
    print("Step 4: Generating embeddings...")
    chunks = embeddings.load_chunks(config.PROCESSED_DOCS_PATH)
    embeddings_array = embeddings.embed_chunks(chunks, config)
    print(f"Generated embeddings: shape {embeddings_array.shape}")
    
    # Step 5: Build vector store
    print("Step 5: Building vector store...")
    embeddings.build_vector_store(chunks, embeddings_array, config)
    print(f"Saved vector store to {config.VECTOR_STORE_PATH}")
    
    print()
    print("Index building complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

