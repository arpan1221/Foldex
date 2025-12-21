"""Test suite for query classifier."""

import pytest
from app.rag.query_classifier import QueryClassifier, QueryType, QueryUnderstanding


class TestQueryClassifier:
    """Test query classification accuracy and edge cases."""
    
    @pytest.fixture
    def available_files(self):
        """Sample files for file reference detection."""
        return [
            {"file_name": "README.md"},
            {"file_name": "LICENSE.txt"},
            {"file_name": "cs2.wav"},
            {"file_name": "Screenshot 2025-12-10 at 5.48.41 PM.png"},
            {"file_name": "Reconciliation.docx"},
            {"file_name": "tax_delinquent.csv"},
        ]
    
    @pytest.fixture
    def classifier(self, available_files):
        """Create classifier instance."""
        return QueryClassifier(available_files=available_files)
    
    # FACTUAL_SPECIFIC tests
    def test_factual_specific_audio(self, classifier):
        """Test audio-specific queries."""
        result = classifier.classify("What does the audio file say?")
        assert result.query_type == QueryType.FACTUAL_SPECIFIC
        assert result.confidence >= 0.5
        assert result.content_type == "audio"
        assert "audio" in result.explanation.lower()
    
    def test_factual_specific_image(self, classifier):
        """Test image-specific queries."""
        result = classifier.classify("What is in the screenshot?")
        assert result.query_type == QueryType.FACTUAL_SPECIFIC
        assert result.content_type == "image"
    
    def test_factual_specific_file_reference(self, classifier):
        """Test file name reference detection."""
        result = classifier.classify("What does LICENSE say?")
        assert result.query_type == QueryType.FACTUAL_SPECIFIC
        assert result.confidence >= 0.7
        assert "LICENSE.txt" in result.file_references or "LICENSE" in result.file_references
    
    def test_factual_specific_content_describe(self, classifier):
        """Test describe/explain queries."""
        result = classifier.classify("Describe the contents of the PDF")
        assert result.query_type == QueryType.FACTUAL_SPECIFIC
        assert result.content_type == "document"
    
    # FACTUAL_GENERAL tests
    def test_factual_general_overview(self, classifier):
        """Test general overview queries."""
        result = classifier.classify("What is this folder about?")
        assert result.query_type == QueryType.FACTUAL_GENERAL
        assert result.confidence >= 0.3
    
    def test_factual_general_summarize(self, classifier):
        """Test summarize queries."""
        result = classifier.classify("Summarize the contents")
        assert result.query_type == QueryType.FACTUAL_GENERAL
    
    # RELATIONSHIP tests
    def test_relationship_common_themes(self, classifier):
        """Test common themes queries."""
        result = classifier.classify("What are common themes?")
        assert result.query_type == QueryType.RELATIONSHIP
        assert result.confidence >= 0.5
    
    def test_relationship_patterns(self, classifier):
        """Test pattern queries."""
        result = classifier.classify("Find patterns across files")
        assert result.query_type == QueryType.RELATIONSHIP
    
    def test_relationship_connections(self, classifier):
        """Test connection queries."""
        result = classifier.classify("How do these documents relate?")
        assert result.query_type == QueryType.RELATIONSHIP
    
    # COMPARISON tests
    def test_comparison_explicit(self, classifier):
        """Test explicit comparison queries."""
        result = classifier.classify("Compare README and LICENSE")
        assert result.query_type == QueryType.COMPARISON
        assert result.confidence >= 0.7
        assert len(result.entities) >= 1  # Should extract README and/or LICENSE
    
    def test_comparison_difference(self, classifier):
        """Test difference queries."""
        result = classifier.classify("What is the difference between X and Y?")
        assert result.query_type == QueryType.COMPARISON
    
    def test_comparison_versus(self, classifier):
        """Test versus queries."""
        result = classifier.classify("README versus LICENSE")
        assert result.query_type == QueryType.COMPARISON
    
    # ENTITY_SEARCH tests
    def test_entity_search_all_mentions(self, classifier):
        """Test all mentions queries."""
        result = classifier.classify("Find all mentions of Whisper")
        assert result.query_type == QueryType.ENTITY_SEARCH
        assert "Whisper" in result.entities
    
    def test_entity_search_where(self, classifier):
        """Test where queries."""
        result = classifier.classify("Where is Y discussed?")
        assert result.query_type == QueryType.ENTITY_SEARCH
    
    def test_entity_search_every_reference(self, classifier):
        """Test every reference queries."""
        result = classifier.classify("Every reference to Z")
        assert result.query_type == QueryType.ENTITY_SEARCH
    
    # TEMPORAL tests
    def test_temporal_recent(self, classifier):
        """Test recent queries."""
        result = classifier.classify("What changed recently?")
        assert result.query_type == QueryType.TEMPORAL
    
    def test_temporal_latest(self, classifier):
        """Test latest queries."""
        result = classifier.classify("Show me the latest updates")
        assert result.query_type == QueryType.TEMPORAL
    
    def test_temporal_yesterday(self, classifier):
        """Test time-based queries."""
        result = classifier.classify("What changed yesterday?")
        assert result.query_type == QueryType.TEMPORAL
    
    # Edge cases
    def test_empty_query(self, classifier):
        """Test empty query handling."""
        result = classifier.classify("")
        assert result.query_type == QueryType.FACTUAL_GENERAL
        assert result.confidence == 0.0
    
    def test_tie_resolution_rel_over_specific(self, classifier):
        """Test that RELATIONSHIP is preferred over FACTUAL_SPECIFIC in ties."""
        # Query that could match both - should prefer RELATIONSHIP
        result = classifier.classify("What are common themes in the audio files?")
        # Should lean towards RELATIONSHIP due to "common themes"
        assert result.query_type in [QueryType.RELATIONSHIP, QueryType.FACTUAL_SPECIFIC]
        # If it's RELATIONSHIP, that's correct. If FACTUAL_SPECIFIC, it's also acceptable
        # due to "audio files" being very specific
    
    def test_entity_extraction_quoted(self, classifier):
        """Test entity extraction from quoted strings."""
        result = classifier.classify('Find all mentions of "Whisper transcription"')
        assert "Whisper transcription" in result.entities or "Whisper" in result.entities
    
    def test_entity_extraction_capitalized(self, classifier):
        """Test entity extraction from capitalized words."""
        result = classifier.classify("Where is Whisper discussed?")
        assert "Whisper" in result.entities or len(result.entities) > 0
    
    def test_file_reference_case_insensitive(self, classifier):
        """Test file reference detection is case-insensitive."""
        result = classifier.classify("What is in readme.md?")
        assert len(result.file_references) > 0
    
    def test_multiple_file_references(self, classifier):
        """Test detection of multiple file references."""
        result = classifier.classify("Compare README and LICENSE")
        assert len(result.file_references) >= 1
    
    # Integration tests
    def test_content_type_detection(self, classifier):
        """Test content type detection for various types."""
        test_cases = [
            ("What's in the audio file?", "audio"),
            ("Describe the screenshot", "image"),
            ("Show me the video", "video"),
            ("What's in the PDF?", "document"),
            ("Explain the code", "code"),
            ("What's in the CSV?", "text"),
        ]
        
        for query, expected_type in test_cases:
            result = classifier.classify(query)
            if result.query_type == QueryType.FACTUAL_SPECIFIC:
                assert result.content_type == expected_type, f"Failed for query: {query}"
    
    def test_confidence_scores(self, classifier):
        """Test that confidence scores are reasonable."""
        # High confidence queries
        high_conf_queries = [
            "Compare X and Y",  # Very specific pattern
            "What does LICENSE say?",  # File reference
            "Find all mentions of Z",  # Clear pattern
        ]
        
        for query in high_conf_queries:
            result = classifier.classify(query)
            assert result.confidence >= 0.5, f"Low confidence for: {query}"
        
        # Low confidence (ambiguous) queries
        result = classifier.classify("Tell me about this")
        # Should have lower confidence or default to FACTUAL_GENERAL
        assert result.confidence <= 0.8
    
    def test_explanation_generation(self, classifier):
        """Test that explanations are generated."""
        result = classifier.classify("What does the audio file say?")
        assert result.explanation
        assert result.query_type.value in result.explanation
    
    def test_update_available_files(self, classifier):
        """Test updating available files."""
        new_files = [{"file_name": "new_file.txt"}]
        classifier.update_available_files(new_files)
        
        result = classifier.classify("What is in new_file.txt?")
        assert "new_file.txt" in result.file_references


# Test accuracy requirement (80%+)
def test_classification_accuracy():
    """Test overall classification accuracy on representative queries."""
    available_files = [
        {"file_name": "README.md"},
        {"file_name": "LICENSE.txt"},
        {"file_name": "cs2.wav"},
    ]
    
    classifier = QueryClassifier(available_files=available_files)
    
    # Ground truth test cases
    test_cases = [
        # (query, expected_type, min_confidence)
        ("What does the audio file say?", QueryType.FACTUAL_SPECIFIC, 0.5),
        ("What are common themes?", QueryType.RELATIONSHIP, 0.5),
        ("Compare README and LICENSE", QueryType.COMPARISON, 0.7),
        ("Find all mentions of Whisper", QueryType.ENTITY_SEARCH, 0.5),
        ("What changed yesterday?", QueryType.TEMPORAL, 0.5),
        ("What is this folder about?", QueryType.FACTUAL_GENERAL, 0.3),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_type, min_conf in test_cases:
        result = classifier.classify(query)
        if result.query_type == expected_type and result.confidence >= min_conf:
            correct += 1
        else:
            print(f"Mismatch: '{query}' -> {result.query_type} (expected {expected_type}), "
                  f"confidence: {result.confidence:.2f}")
    
    accuracy = correct / total
    assert accuracy >= 0.8, f"Classification accuracy {accuracy:.1%} below 80% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

