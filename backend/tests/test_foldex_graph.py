"""
Test script for FoldexGraph multi-document synthesis workflow.

This script demonstrates the stateful multi-step reasoning pipeline:
1. Retrieve chunks with hybrid retriever
2. Extract entities from each document
3. Synthesize across documents
4. Generate citations

Run with:
    python backend/test_foldex_graph.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.langgraph.foldex_graph import create_foldex_graph
from app.services.rag_service import get_rag_service
from app.rag.vector_store import LangChainVectorStore


async def test_foldex_graph(folder_id: str = None):
    """Test the FoldexGraph workflow with a sample query.
    
    Args:
        folder_id: Optional folder_id to test. If not provided, will try to find one.
    """
    
    print("=" * 80)
    print("FoldexGraph Multi-Document Synthesis Test")
    print("=" * 80)
    print()
    
    # Get folder_id from command line or use default
    if not folder_id:
        if len(sys.argv) > 1:
            folder_id = sys.argv[1]
        else:
            # Try to find an existing folder_id from vector_db
            vector_db_path = Path(__file__).parent.parent / "data" / "vector_db"
            if vector_db_path.exists():
                folders = [d.name for d in vector_db_path.iterdir() if d.is_dir() and d.name != "__pycache__"]
                if folders:
                    folder_id = folders[0]
                    print(f"Using folder_id from vector_db: {folder_id}")
                else:
                    print("No folder_id provided and none found in vector_db")
                    print("\nUsage: python test_foldex_graph.py <folder_id>")
                    print("\nTo test:")
                    print("1. Process a folder with multiple documents")
                    print("2. Run: python test_foldex_graph.py <folder_id>")
                    return
            else:
                print("No folder_id provided")
                print("\nUsage: python test_foldex_graph.py <folder_id>")
                return
    
    if not folder_id:
        print("Skipping test - no folder_id provided")
        return
    
    # Test query
    query = "What are common entities between files in this folder?"
    print(f"Query: {query}")
    print()
    
    try:
        # Initialize RAG service
        print("Initializing RAG service...")
        vector_store = LangChainVectorStore()
        rag_service = get_rag_service(vector_store=vector_store)
        
        # Ensure folder is initialized
        print(f"Loading documents for folder: {folder_id}")
        await rag_service.initialize_for_folder(folder_id)
        
        # Get retriever and LLM
        retriever = rag_service.retrievers[folder_id]
        llm = rag_service.llm
        
        # Create FoldexGraph
        print("Creating FoldexGraph workflow...")
        foldex_graph = create_foldex_graph(retriever, llm)
        
        # Execute graph
        print("\n" + "=" * 80)
        print("Executing Graph Workflow")
        print("=" * 80)
        print()
        
        print("Step 1: Retrieving chunks...")
        graph_result = foldex_graph.invoke(
            query=query,
            folder_id=folder_id,
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print()
        
        if graph_result.get("error"):
            print(f"❌ Error: {graph_result['error']}")
            return
        
        # Show unique files
        unique_files = graph_result.get("unique_files", [])
        print(f"📁 Files Analyzed: {len(unique_files)}")
        for i, file_name in enumerate(unique_files, 1):
            print(f"   {i}. {file_name}")
        print()
        
        # Show entities per file
        entities_per_file = graph_result.get("entities_per_file", {})
        print("🔍 Extracted Entities:")
        for file_name, entities in entities_per_file.items():
            print(f"\n   {file_name}:")
            if isinstance(entities, dict) and "error" not in entities:
                for key, value in entities.items():
                    if isinstance(value, list):
                        print(f"      {key}: {', '.join(map(str, value))}")
                    else:
                        print(f"      {key}: {value}")
            else:
                print(f"      Error: {entities.get('error', 'Unknown error')}")
        print()
        
        # Show synthesis
        synthesis = graph_result.get("synthesis", "")
        print("📝 Cross-Document Synthesis:")
        print("-" * 80)
        print(synthesis)
        print("-" * 80)
        print()
        
        # Show citations
        citations = graph_result.get("citations", [])
        print(f"📚 Citations: {len(citations)}")
        for citation in citations:
            print(f"   [{citation.get('citation_number', '?')}] {citation.get('file_name', 'Unknown')}")
            if citation.get("page_number"):
                print(f"       Page: {citation['page_number']}")
            if citation.get("google_drive_url"):
                print(f"       URL: {citation['google_drive_url']}")
        print()
        
        # Summary
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"✅ Workflow completed successfully")
        print(f"   - Files analyzed: {len(unique_files)}")
        print(f"   - Total chunks: {graph_result.get('total_chunks', 0)}")
        print(f"   - Synthesis length: {len(synthesis)} characters")
        print(f"   - Citations: {len(citations)}")
        print()
        
        # Show expected output structure
        print("Expected API Response Structure:")
        print("-" * 80)
        import json
        api_response = {
            "answer": synthesis,
            "citations": citations,
            "entities_found": entities_per_file
        }
        print(json.dumps(api_response, indent=2, default=str))
        print()
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Get folder_id from command line if provided
    folder_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_foldex_graph(folder_id=folder_id))

