"""Script to test the RAG pipeline with a real query."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src import embeddings
from src import rag_pipeline

QUERIES = [
    "What is the impact of climate change on consumer perceptions?",
    "How can cognitive load theory guide instructional design in medical education?",
    "What strategies improve global rehabilitation data infrastructure?",
    "How should culturally responsive AI chatbots be designed?"
]


def run_query(query: str, vector_store):
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # Test Baseline
    print("Testing Baseline Condition...")
    result_baseline = rag_pipeline.answer_query(
        query=query,
        condition="baseline",
        vector_store=vector_store,
        config=config
    )
    print("\nResponse (Baseline):")
    print(result_baseline["answer"])
    print("\nRetrieved Contexts:")
    for chunk in result_baseline["retrieved_chunks"]:
        print(f"- {chunk.chunk.title} (Score: {chunk.score:.4f})")
        
    print("\n" + "=" * 50 + "\n")
    
    # Test CL-Aware (if different prompt logic exists)
    print("Testing CL-Aware Condition...")
    result_cl = rag_pipeline.answer_query(
        query=query,
        condition="cl_aware",
        vector_store=vector_store,
        config=config
    )
    print("\nResponse (CL-Aware):")
    print(result_cl["answer"])
    
    print("\n" + "#" * 70 + "\n")


def main():
    print("Loading vector store...")
    vector_store = embeddings.load_vector_store(config)
    
    for query in QUERIES:
        run_query(query, vector_store)

if __name__ == "__main__":
    main()

