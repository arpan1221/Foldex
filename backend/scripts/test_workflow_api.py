"""API-based workflow test script - requires backend server to be running."""

import requests
import json
import sys
import time
from typing import Dict, Any, Optional


BASE_URL = "http://localhost:8000"


def test_workflow_api(folder_name: str = "Test Folder", access_token: Optional[str] = None):
    """Test complete workflow via API.
    
    Args:
        folder_name: Name of folder to test
        access_token: JWT access token (if None, will need to authenticate)
    """
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    
    print(f"\n{'='*60}")
    print("🚀 FOLDEX WORKFLOW TEST (API)")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Folder: {folder_name}\n")
    
    # Step 1: Check folder status
    print("="*60)
    print("STEP 1: Checking Folder Status")
    print("="*60)
    
    # First, get user folders to find the folder_id
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/folders",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 401:
            print("❌ Authentication required. Please provide access_token or authenticate first.")
            print("\nTo get a token:")
            print("1. Sign in via the frontend at http://localhost:3000")
            print("2. Get the token from browser localStorage: localStorage.getItem('access_token')")
            print("3. Run: python test_workflow_api.py <folder_name> <access_token>")
            return
        
        response.raise_for_status()
        folders = response.json()
        
        folder = None
        for f in folders:
            if f.get("folder_name") == folder_name:
                folder = f
                break
        
        if not folder:
            print(f"❌ Folder '{folder_name}' not found")
            print(f"\nAvailable folders:")
            for f in folders:
                print(f"   - {f.get('folder_name')} ({f.get('folder_id')})")
            return
        
        folder_id = folder["folder_id"]
        print(f"✅ Found folder: {folder['folder_name']}")
        print(f"   ID: {folder_id}")
        print(f"   Status: {folder.get('status')}")
        print(f"   File Count: {folder.get('file_count')}\n")
        
        # Get folder status details
        status_response = requests.get(
            f"{BASE_URL}/api/v1/folders/{folder_id}/status",
            headers=headers,
            timeout=10
        )
        status_response.raise_for_status()
        status = status_response.json()
        
        print(f"   Processing Status: {status.get('status')}")
        if status.get('error'):
            print(f"   Error: {status.get('error')}")
        print()
        
        # Get files in folder
        files_response = requests.get(
            f"{BASE_URL}/api/v1/folders/{folder_id}/files",
            headers=headers,
            timeout=10
        )
        files_response.raise_for_status()
        files = files_response.json()
        
        print(f"📄 Files in folder ({len(files)}):")
        for file in files:
            print(f"   - {file.get('file_name')} ({file.get('mime_type')})")
        print()
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print("   Make sure the backend server is running:")
        print("   cd backend && uvicorn app.main:app --reload")
        return
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        return
    
    # Step 2: Test queries
    test_queries = [
        {
            "name": "Simple Query",
            "query": "What is the main topic of this folder?",
        },
        {
            "name": "File Summary Query",
            "query": "Summarize the README file.",
        },
        {
            "name": "Cross-Document Query",
            "query": "What are the common themes across all documents in this folder?",
        },
        {
            "name": "Specific Information Query",
            "query": "What information is in the LICENSE file?",
        },
    ]
    
    conversation_id = None
    
    for test in test_queries:
        print("="*60)
        print(f"STEP: {test['name']}")
        print("="*60)
        print(f"Query: {test['query']}\n")
        
        try:
            query_data = {
                "query": test["query"],
                "folder_id": folder_id,
            }
            
            if conversation_id:
                query_data["conversation_id"] = conversation_id
            
            response = requests.post(
                f"{BASE_URL}/api/v1/chat/query",
                headers={**headers, "Content-Type": "application/json"},
                json=query_data,
                timeout=120  # Longer timeout for LLM queries
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result.get("response", "")
            citations = result.get("citations", [])
            conversation_id = result.get("conversation_id")
            
            print(f"✅ Response received ({len(answer)} characters)")
            print(f"\n📝 Answer:")
            print("-" * 60)
            # Show full answer, but truncate if very long
            if len(answer) > 800:
                print(answer[:800])
                print("\n... (truncated)")
            else:
                print(answer)
            print("-" * 60)
            
            if citations:
                print(f"\n📚 Citations ({len(citations)}):")
                # Group by file
                file_groups = {}
                for citation in citations:
                    file_name = citation.get("file_name", "Unknown")
                    if file_name not in file_groups:
                        file_groups[file_name] = []
                    file_groups[file_name].append(citation)
                
                for file_name, file_citations in file_groups.items():
                    print(f"   📄 {file_name} ({len(file_citations)} chunks)")
                    # Show first citation from each file
                    if file_citations:
                        first_cit = file_citations[0]
                        chunk_preview = first_cit.get("chunk_content", "")[:100]
                        if chunk_preview:
                            print(f"      Preview: {chunk_preview}...")
            else:
                print("\n⚠️  No citations found")
            
            print()
            time.sleep(2)  # Brief pause between queries
            
        except requests.exceptions.Timeout:
            print("❌ Query timed out (took longer than 120 seconds)")
            print()
        except requests.exceptions.RequestException as e:
            print(f"❌ Query failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error detail: {error_detail}")
                except:
                    print(f"   Status: {e.response.status_code}")
            print()
    
    # Summary
    print("="*60)
    print("✅ Workflow Test Complete")
    print("="*60)
    print(f"\nTested folder: {folder_name}")
    print(f"Folder ID: {folder_id}")
    print(f"Total queries tested: {len(test_queries)}")
    print(f"Conversation ID: {conversation_id}\n")


if __name__ == "__main__":
    folder_name = "Test Folder"
    access_token = None
    
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    if len(sys.argv) > 2:
        access_token = sys.argv[2]
    
    test_workflow_api(folder_name, access_token)

