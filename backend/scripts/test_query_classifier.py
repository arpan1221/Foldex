"""Simple test script for query classifier (can run directly without pytest)."""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.query_classifier import QueryClassifier, QueryType

# Sample files for file reference detection
available_files = [
    {"file_name": "README.md"},
    {"file_name": "LICENSE.txt"},
    {"file_name": "cs2.wav"},
    {"file_name": "Screenshot 2025-12-10 at 5.48.41 PM.png"},
    {"file_name": "Reconciliation.docx"},
    {"file_name": "tax_delinquent.csv"},
]

classifier = QueryClassifier(available_files=available_files, use_llm=True)

# Test cases with expected types
test_cases = [
    # FACTUAL_SPECIFIC
    ("What does the audio file say?", QueryType.FACTUAL_SPECIFIC, "audio"),
    ("What is in the screenshot?", QueryType.FACTUAL_SPECIFIC, "image"),
    ("What does LICENSE say?", QueryType.FACTUAL_SPECIFIC, None),
    ("Describe the PDF", QueryType.FACTUAL_SPECIFIC, "document"),
    ("Show me the CSV data", QueryType.FACTUAL_SPECIFIC, "text"),
    
    # FACTUAL_GENERAL
    ("What is this folder about?", QueryType.FACTUAL_GENERAL, None),
    ("Summarize the contents", QueryType.FACTUAL_GENERAL, None),
    ("Give me an overview", QueryType.FACTUAL_GENERAL, None),
    
    # RELATIONSHIP
    ("What are common themes?", QueryType.RELATIONSHIP, None),
    ("Find patterns across files", QueryType.RELATIONSHIP, None),
    ("How do these documents relate?", QueryType.RELATIONSHIP, None),
    
    # COMPARISON
    ("Compare README and LICENSE", QueryType.COMPARISON, None),
    ("What is the difference between X and Y?", QueryType.COMPARISON, None),
    ("README versus LICENSE", QueryType.COMPARISON, None),
    
    # ENTITY_SEARCH
    ("Find all mentions of Whisper", QueryType.ENTITY_SEARCH, None),
    ("Where is Y discussed?", QueryType.ENTITY_SEARCH, None),
    ("Every reference to Z", QueryType.ENTITY_SEARCH, None),
    
    # TEMPORAL
    ("What changed recently?", QueryType.TEMPORAL, None),
    ("Show me the latest updates", QueryType.TEMPORAL, None),
    ("What changed yesterday?", QueryType.TEMPORAL, None),
]

async def run_tests():
    """Run query classification tests."""
    print("=" * 80)
    print("QUERY CLASSIFIER TEST RESULTS")
    print("=" * 80)
    print()
    
    # Check if LLM is available
    if classifier.llm:
        print("✅ Using LLM-based classification (llama3.2:1b)")
    else:
        print("⚠️  LLM not available, using pattern-based fallback")
    print()
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_type, expected_content_type in test_cases:
        try:
            # Try async LLM classification first, fallback to sync pattern matching
            if classifier.llm:
                result = await classifier.classify(query)
            else:
                result = classifier.classify_sync(query)
        except Exception as e:
            print(f"❌ Error classifying query: {query}")
            print(f"   Error: {str(e)}")
            print()
            continue
        
        type_match = result.query_type == expected_type
        content_match = expected_content_type is None or result.content_type == expected_content_type
        
        status = "✅" if (type_match and content_match) else "❌"
        
        if type_match and content_match:
            correct += 1
        
        print(f"{status} Query: \"{query}\"")
        print(f"   Expected: {expected_type.value}", end="")
        if expected_content_type:
            print(f" ({expected_content_type})", end="")
        print()
        print(f"   Got:      {result.query_type.value}", end="")
        if result.content_type:
            print(f" ({result.content_type})", end="")
        print(f" (confidence: {result.confidence:.2f})")
        if result.file_references:
            print(f"   Files:    {result.file_references}")
        if result.entities:
            print(f"   Entities: {result.entities[:3]}")  # Show first 3
        print(f"   {result.explanation}")
        print()
    
    accuracy = (correct / total) * 100
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 80)
    
    if accuracy >= 80:
        print("✅ PASSED: Classification accuracy meets 80% requirement")
        return 0
    else:
        print("❌ FAILED: Classification accuracy below 80% requirement")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)

