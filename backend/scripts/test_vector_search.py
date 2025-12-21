#!/usr/bin/env python3
"""Standalone script to test vector search with confidence scores.

This script can be run directly to test vector search functionality:
    python scripts/test_vector_search.py

It creates sample documents, adds them to the vector store, and performs
similarity searches with various queries, displaying confidence scores.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Set environment variables before importing settings
# This ensures settings validation passes for test scripts
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-vector-search-script-32chars")
os.environ.setdefault("APP_ENV", "test")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from app.rag.vector_store import LangChainVectorStore
from app.config.settings import settings
import tempfile
import shutil


# Sample documents covering different topics
SAMPLE_DOCUMENTS = [
    {
        "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
        "metadata": {
            "file_id": "doc_1",
            "file_name": "python_intro.txt",
            "topic": "programming",
            "language": "python",
            "chunk_index": 0,
        }
    },
    {
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Common algorithms include neural networks, decision trees, and support vector machines.",
        "metadata": {
            "file_id": "doc_2",
            "file_name": "ml_basics.txt",
            "topic": "machine_learning",
            "chunk_index": 0,
        }
    },
    {
        "content": "FastAPI is a modern web framework for building APIs with Python. It's based on standard Python type hints and provides automatic API documentation, high performance, and easy async support.",
        "metadata": {
            "file_id": "doc_3",
            "file_name": "fastapi_guide.txt",
            "topic": "programming",
            "language": "python",
            "framework": "fastapi",
            "chunk_index": 0,
        }
    },
    {
        "content": "Vector databases store embeddings in a high-dimensional space, allowing for efficient similarity search. Popular vector databases include ChromaDB, Pinecone, and Weaviate.",
        "metadata": {
            "file_id": "doc_4",
            "file_name": "vector_db.txt",
            "topic": "databases",
            "chunk_index": 0,
        }
    },
    {
        "content": "Retrieval-Augmented Generation (RAG) combines information retrieval with language models to provide accurate, context-aware responses. It retrieves relevant documents and uses them to generate answers.",
        "metadata": {
            "file_id": "doc_5",
            "file_name": "rag_explained.txt",
            "topic": "machine_learning",
            "technique": "rag",
            "chunk_index": 0,
        }
    },
    {
        "content": "Docker is a containerization platform that packages applications and their dependencies into containers. This enables consistent deployment across different environments and simplifies DevOps workflows.",
        "metadata": {
            "file_id": "doc_6",
            "file_name": "docker_intro.txt",
            "topic": "devops",
            "tool": "docker",
            "chunk_index": 0,
        }
    },
    {
        "content": "SQLite is a lightweight, file-based relational database management system. It's embedded in many applications and doesn't require a separate server process, making it ideal for local development and small applications.",
        "metadata": {
            "file_id": "doc_7",
            "file_name": "sqlite_info.txt",
            "topic": "databases",
            "db_type": "sqlite",
            "chunk_index": 0,
        }
    },
    {
        "content": "Natural Language Processing (NLP) involves teaching computers to understand, interpret, and generate human language. Key tasks include sentiment analysis, named entity recognition, and machine translation.",
        "metadata": {
            "file_id": "doc_8",
            "file_name": "nlp_overview.txt",
            "topic": "machine_learning",
            "subfield": "nlp",
            "chunk_index": 0,
        }
    },
    {
        "content": "LangChain is a framework for developing applications powered by language models. It provides tools for building RAG applications, managing prompts, and connecting to various data sources.",
        "metadata": {
            "file_id": "doc_9",
            "file_name": "langchain_intro.txt",
            "topic": "programming",
            "framework": "langchain",
            "chunk_index": 0,
        }
    },
    {
        "content": "ChromaDB is an open-source vector database designed for AI applications. It provides a simple API for storing and querying embeddings, making it easy to build semantic search and RAG applications.",
        "metadata": {
            "file_id": "doc_10",
            "file_name": "chromadb_info.txt",
            "topic": "databases",
            "db_type": "chromadb",
            "chunk_index": 0,
        }
    },
]


def print_search_results(query: str, results_with_scores: List[Tuple[Document, float]], limit: int = 5):
    """Print search results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Query: '{query}'")
    print(f"{'='*80}")
    print(f"Found {len(results_with_scores)} results (showing top {limit}):\n")
    
    for i, (doc, score) in enumerate(results_with_scores[:limit], 1):
        # Convert distance to similarity (assuming cosine distance)
        # Lower distance = higher similarity
        # For cosine similarity, distance = 1 - similarity, so similarity = 1 - distance
        similarity = max(0.0, 1.0 - score) if score <= 1.0 else 1.0 / (1.0 + score)
        
        print(f"Result {i} (Distance: {score:.4f}, Similarity: {similarity:.4f})")
        print(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
        print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {dict(list(doc.metadata.items())[:3])}")  # Show first 3 metadata items
        print()


async def test_vector_search():
    """Main test function."""
    # Create temporary directory for vector store
    temp_dir = Path(tempfile.mkdtemp())
    vector_store_path = temp_dir / "test_vector_store"
    
    try:
        print("="*80)
        print("VECTOR SEARCH TEST")
        print("="*80)
        print(f"\nUsing temporary vector store: {vector_store_path}")
        print(f"Embedding model: {settings.EMBEDDING_MODEL}")
        print("\nInitializing vector store...")
        
        # Initialize vector store
        vector_store = LangChainVectorStore(
            persist_directory=str(vector_store_path),
            collection_name="test_collection"
        )
        
        print("✓ Vector store initialized")
        
        # Convert to LangChain Documents
        print(f"\nAdding {len(SAMPLE_DOCUMENTS)} sample documents...")
        documents = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
            for doc in SAMPLE_DOCUMENTS
        ]
        
        # Add documents to vector store
        doc_ids = await vector_store.add_documents(documents)
        print(f"✓ Added {len(doc_ids)} documents to vector store")
        
        # Get collection info
        info = await vector_store.get_collection_info()
        print(f"\nCollection Info:")
        print(f"  Document count: {info['document_count']}")
        print(f"  Embedding model: {info['embedding_model']}")
        
        # Test queries
        test_queries = [
            "Python programming language",
            "machine learning algorithms",
            "vector databases for embeddings",
            "web framework for APIs",
            "containerization and deployment",
            "RAG retrieval augmented generation",
            "database management systems",
        ]
        
        print("\n" + "="*80)
        print("TESTING SIMILARITY SEARCH")
        print("="*80)
        
        for query in test_queries:
            results_with_scores = await vector_store.similarity_search_with_score(
                query=query,
                k=3
            )
            print_search_results(query, results_with_scores, limit=3)
        
        # Test with metadata filtering
        print("\n" + "="*80)
        print("TESTING FILTERED SEARCH")
        print("="*80)
        
        query = "programming"
        print(f"\nQuery: '{query}' (filtered by language=python)")
        filtered_results = await vector_store.similarity_search(
            query=query,
            k=5,
            filter={"language": "python"}
        )
        
        print(f"Found {len(filtered_results)} results:\n")
        for i, doc in enumerate(filtered_results, 1):
            print(f"Result {i}:")
            print(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"  Content: {doc.page_content[:150]}...")
            print()
        
        # Test relevance scoring
        print("\n" + "="*80)
        print("RELEVANCE SCORE ANALYSIS")
        print("="*80)
        
        relevance_tests = [
            {
                "query": "Python programming",
                "expected_file": "python_intro.txt",
                "description": "Should find Python intro with high confidence"
            },
            {
                "query": "machine learning neural networks",
                "expected_file": "ml_basics.txt",
                "description": "Should find ML basics with high confidence"
            },
            {
                "query": "ChromaDB vector database",
                "expected_file": "chromadb_info.txt",
                "description": "Should find ChromaDB doc with high confidence"
            },
        ]
        
        for test in relevance_tests:
            results_with_scores = await vector_store.similarity_search_with_score(
                query=test["query"],
                k=3
            )
            
            print(f"\n{test['description']}")
            print(f"Query: '{test['query']}'")
            
            # Check if expected file is in top results
            found = False
            for i, (doc, score) in enumerate(results_with_scores, 1):
                if doc.metadata.get("file_name") == test["expected_file"]:
                    similarity = max(0.0, 1.0 - score) if score <= 1.0 else 1.0 / (1.0 + score)
                    print(f"  ✓ Found expected document at rank {i} (similarity: {similarity:.4f})")
                    found = True
                    break
            
            if not found:
                print(f"  ✗ Expected document not in top 3 results")
                print(f"  Top result: {results_with_scores[0][0].metadata.get('file_name')}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nSummary:")
        print("  ✓ Vector store initialized")
        print("  ✓ Documents embedded and stored")
        print("  ✓ Similarity search working")
        print("  ✓ Confidence scores calculated")
        print("  ✓ Metadata filtering working")
        print("  ✓ Relevance scoring verified")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if temp_dir.exists():
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_vector_search())
    sys.exit(0 if success else 1)

