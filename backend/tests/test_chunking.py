"""
Test script for new intelligent chunking system.

Tests chunking with the two PDF papers in the repository.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.chunking import FoldexChunker


def print_chunk_sample(chunks, num_samples=3):
    """Print sample chunks with metadata."""
    print(f"\n{'='*80}")
    print(f"CHUNKING RESULTS: {len(chunks)} total chunks")
    print(f"{'='*80}\n")

    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\n{'─'*80}")
        print(f"CHUNK #{i + 1}")
        print(f"{'─'*80}")

        # Print metadata
        print("\nMETADATA:")
        metadata = chunk.metadata
        for key, value in sorted(metadata.items()):
            if key in ["prev_context", "next_context"]:
                # Show truncated context
                if value:
                    print(f"  {key}: '{value[:50]}...' ({len(value)} chars)")
                else:
                    print(f"  {key}: (empty)")
            elif isinstance(value, list):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        # Print content
        print("\nCONTENT:")
        content = chunk.page_content
        print(f"  Length: {len(content)} characters (~{len(content)//4} tokens)")
        print(f"  Preview: {content[:300]}...")
        if len(content) > 300:
            print(f"  ... (truncated, {len(content) - 300} more chars)")

    print(f"\n{'='*80}")
    print(f"Showing {min(num_samples, len(chunks))} of {len(chunks)} chunks")
    print(f"{'='*80}\n")


def main():
    """Test chunking with PDF files."""
    print("Testing Foldex Intelligent Chunking System")
    print("=" * 80)

    # Initialize chunker (optimized for Llama 3.2:3b)
    chunker = FoldexChunker(
        chunk_size=800,  # ~200 tokens (optimized for Llama 3.2:3b)
        chunk_overlap=150,
        context_window_size=150,
    )

    # Test files
    test_files = [
        {
            "path": "Deep_Reinforcement_Learning_based_Intelligent_Traffic_Control.pdf",
            "name": "Traffic Control Paper",
        },
        {
            "path": "Deep_Reinforcement_Learning_for_Automated_Stock_Trading_Inclusion_of_Short_Selling (1).pdf",
            "name": "Stock Trading Paper",
        },
    ]

    for test_file in test_files:
        file_path = test_file["path"]
        file_name = test_file["name"]

        print(f"\n\n{'#'*80}")
        print(f"# Testing: {file_name}")
        print(f"# File: {file_path}")
        print(f"{'#'*80}\n")

        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            continue

        # Prepare file metadata
        file_metadata = {
            "file_id": f"test_{file_name.replace(' ', '_').lower()}",
            "file_name": file_name,
            "drive_url": f"https://drive.google.com/file/test_{file_name}",
            "folder_id": "test_folder",
            "user_id": "test_user",
        }

        try:
            # Chunk PDF
            print(f"⏳ Chunking PDF...")
            chunks = chunker.chunk_pdf(file_path, file_metadata)

            # Print results
            print(f"✅ Successfully chunked PDF")
            print_chunk_sample(chunks, num_samples=3)

            # Statistics
            print("\nSTATISTICS:")
            print(f"  Total chunks: {len(chunks)}")
            chunk_sizes = [len(c.page_content) for c in chunks]
            print(f"  Avg chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars (~{sum(chunk_sizes) / len(chunk_sizes) / 4:.0f} tokens)")
            print(f"  Min chunk size: {min(chunk_sizes)} chars")
            print(f"  Max chunk size: {max(chunk_sizes)} chars")

            # Check metadata completeness
            print("\nMETADATA COMPLETENESS:")
            first_chunk = chunks[0].metadata
            required_fields = [
                "chunk_id", "file_id", "file_name", "page_number",
                "document_title", "authors", "chunk_index", "total_chunks",
                "prev_context", "next_context"
            ]
            for field in required_fields:
                has_field = field in first_chunk
                value = first_chunk.get(field)
                status = "✅" if has_field and value not in [None, "", []] else "⚠️"
                print(f"  {status} {field}: {type(value).__name__}")

        except Exception as e:
            print(f"❌ Error chunking PDF: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

