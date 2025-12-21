"""Test vector search functionality with confidence scores."""

import pytest
import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document

from app.rag.vector_store import LangChainVectorStore
from app.core.exceptions import VectorStoreError


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
]


@pytest.fixture
def vector_store(temp_vector_store_dir):
    """Create vector store for testing."""
    return LangChainVectorStore(persist_directory=str(temp_vector_store_dir))


@pytest.fixture
async def populated_vector_store(vector_store):
    """Create and populate vector store with sample documents."""
    # Convert to LangChain Documents
    documents = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in SAMPLE_DOCUMENTS
    ]
    
    # Add documents to vector store
    await vector_store.add_documents(documents)
    
    return vector_store


@pytest.mark.asyncio
async def test_vector_search_basic(populated_vector_store):
    """Test basic vector search functionality."""
    # Test query related to Python programming
    query = "Python programming language features"
    
    results = await populated_vector_store.similarity_search(
        query=query,
        k=3
    )
    
    assert len(results) > 0, "Should return at least one result"
    assert len(results) <= 3, "Should return at most k results"
    
    # Check that results contain Python-related content
    python_docs = [r for r in results if "python" in r.page_content.lower() or 
                   r.metadata.get("language") == "python"]
    assert len(python_docs) > 0, "Should return Python-related documents for Python query"


@pytest.mark.asyncio
async def test_vector_search_with_scores(populated_vector_store):
    """Test vector search with confidence scores."""
    query = "machine learning algorithms"
    
    results_with_scores = await populated_vector_store.similarity_search_with_score(
        query=query,
        k=5
    )
    
    assert len(results_with_scores) > 0, "Should return results with scores"
    
    # Verify scores are in descending order (higher similarity = lower distance)
    scores = [score for _, score in results_with_scores]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    
    # Check that top result is relevant
    top_doc, top_score = results_with_scores[0]
    assert "machine learning" in top_doc.page_content.lower() or \
           top_doc.metadata.get("topic") == "machine_learning", \
           "Top result should be relevant to machine learning"
    
    # Scores should be reasonable (typically between 0 and 1 for cosine similarity)
    # Note: ChromaDB uses distance, so lower is better
    assert all(0 <= score <= 2 for _, score in results_with_scores), \
           "Scores should be in reasonable range"


@pytest.mark.asyncio
async def test_vector_search_relevance(populated_vector_store):
    """Test that search results are relevant to queries."""
    test_cases = [
        {
            "query": "Python web framework",
            "expected_topics": ["programming", "python"],
            "expected_keywords": ["fastapi", "python", "framework"]
        },
        {
            "query": "vector database storage",
            "expected_topics": ["databases"],
            "expected_keywords": ["vector", "database", "chromadb", "embedding"]
        },
        {
            "query": "containerization and deployment",
            "expected_topics": ["devops"],
            "expected_keywords": ["docker", "container"]
        },
        {
            "query": "RAG retrieval augmented generation",
            "expected_topics": ["machine_learning"],
            "expected_keywords": ["rag", "retrieval", "generation"]
        },
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        results = await populated_vector_store.similarity_search(
            query=query,
            k=3
        )
        
        assert len(results) > 0, f"Should return results for query: {query}"
        
        # Check that at least one result matches expected topics or keywords
        top_result = results[0]
        content_lower = top_result.page_content.lower()
        metadata = top_result.metadata
        
        topic_match = any(
            metadata.get("topic") == topic 
            for topic in test_case["expected_topics"]
        )
        keyword_match = any(
            keyword in content_lower 
            for keyword in test_case["expected_keywords"]
        )
        
        assert topic_match or keyword_match, \
            f"Top result should be relevant to query '{query}'. Got: {top_result.page_content[:100]}"


@pytest.mark.asyncio
async def test_vector_search_filtering(populated_vector_store):
    """Test vector search with metadata filtering."""
    query = "programming"
    
    # Search without filter
    all_results = await populated_vector_store.similarity_search(
        query=query,
        k=10
    )
    
    # Search with filter for Python documents only
    filtered_results = await populated_vector_store.similarity_search(
        query=query,
        k=10,
        filter={"language": "python"}
    )
    
    assert len(filtered_results) <= len(all_results), \
           "Filtered results should be subset of all results"
    
    # All filtered results should have language=python
    for result in filtered_results:
        assert result.metadata.get("language") == "python", \
               "All filtered results should have language=python"


@pytest.mark.asyncio
async def test_vector_search_confidence_scores_detailed(populated_vector_store):
    """Detailed test of confidence scores for different query types."""
    queries = [
        ("Python programming", "Should find Python docs with high confidence"),
        ("machine learning neural networks", "Should find ML docs with high confidence"),
        ("database storage", "Should find database docs with high confidence"),
        ("unrelated topic xyz abc", "Should still return results but with lower confidence"),
    ]
    
    for query, description in queries:
        results_with_scores = await populated_vector_store.similarity_search_with_score(
            query=query,
            k=3
        )
        
        assert len(results_with_scores) > 0, f"{description}: Should return results"
        
        # Print scores for debugging
        print(f"\nQuery: '{query}'")
        for i, (doc, score) in enumerate(results_with_scores):
            print(f"  Result {i+1} (score: {score:.4f}): {doc.page_content[:80]}...")
            print(f"    Metadata: {doc.metadata}")
        
        # Top result should have reasonable score
        top_score = results_with_scores[0][1]
        assert top_score is not None, f"{description}: Top result should have a score"


@pytest.mark.asyncio
async def test_vector_search_empty_query(populated_vector_store):
    """Test behavior with empty or very short queries."""
    # Empty query should still work (returns all documents)
    results = await populated_vector_store.similarity_search(
        query="",
        k=5
    )
    
    assert len(results) > 0, "Empty query should return some results"


@pytest.mark.asyncio
async def test_vector_search_k_parameter(populated_vector_store):
    """Test that k parameter controls number of results."""
    query = "programming"
    
    for k in [1, 3, 5, 10]:
        results = await populated_vector_store.similarity_search(
            query=query,
            k=k
        )
        
        assert len(results) <= k, f"Should return at most {k} results for k={k}"


def print_search_results(query: str, results_with_scores: List[tuple], limit: int = 5):
    """Helper function to print search results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Query: '{query}'")
    print(f"{'='*80}")
    print(f"Found {len(results_with_scores)} results (showing top {limit}):\n")
    
    for i, (doc, score) in enumerate(results_with_scores[:limit], 1):
        # Convert distance to similarity (assuming cosine distance)
        # Lower distance = higher similarity
        similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
        
        print(f"Result {i} (Distance: {score:.4f}, Similarity: {similarity:.4f})")
        print(f"  Content: {doc.page_content[:150]}...")
        print(f"  Metadata: {doc.metadata}")
        print()


@pytest.mark.asyncio
async def test_vector_search_demo(populated_vector_store):
    """Demo test that prints detailed results for manual inspection."""
    print("\n" + "="*80)
    print("VECTOR SEARCH DEMONSTRATION")
    print("="*80)
    
    demo_queries = [
        "Python programming language",
        "machine learning and AI",
        "vector databases and embeddings",
        "web development frameworks",
        "containerization tools",
    ]
    
    for query in demo_queries:
        results_with_scores = await populated_vector_store.similarity_search_with_score(
            query=query,
            k=3
        )
        print_search_results(query, results_with_scores, limit=3)
    
    # Test with metadata filter
    print("\n" + "="*80)
    print("FILTERED SEARCH DEMONSTRATION")
    print("="*80)
    
    query = "programming"
    filtered_results = await populated_vector_store.similarity_search(
        query=query,
        k=5,
        filter={"language": "python"}
    )
    
    print(f"\nQuery: '{query}' (filtered by language=python)")
    print(f"Found {len(filtered_results)} results:\n")
    for i, doc in enumerate(filtered_results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:150]}...")
        print(f"  Metadata: {doc.metadata}")
        print()

