"""
Knowledge graph API endpoints for Foldex.

Provides endpoints to query and visualize the knowledge graph
built from folder documents.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
import structlog

from app.api.deps import get_current_user
from app.database.sqlite_manager import SQLiteManager
from app.knowledge_graph.graph_builder import FoldexKnowledgeGraph

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/graph/{folder_id}")
async def get_knowledge_graph(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get knowledge graph for a folder.
    
    Args:
        folder_id: Folder identifier
        current_user: Current authenticated user
        
    Returns:
        Graph data in JSON format for D3.js visualization
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        # Get all chunks for this folder
        db = SQLiteManager()
        chunks_data = await db.get_chunks_by_folder(folder_id)
        
        if not chunks_data:
            return {
                "nodes": [],
                "links": [],
                "stats": {
                    "node_count": 0,
                    "link_count": 0,
                    "document_count": 0,
                },
                "message": "No documents found in folder",
            }
        
        # Convert to LangChain Documents
        from langchain_core.documents import Document
        chunks = []
        for chunk_data in chunks_data:
            # Handle both dict and DocumentChunk objects
            if isinstance(chunk_data, dict):
                content = chunk_data.get("content", "")
                metadata = chunk_data.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.get("chunk_id")
                file_id = chunk_data.get("file_id")
            else:
                # DocumentChunk object
                content = chunk_data.content
                metadata = chunk_data.metadata or {}
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.chunk_id
                file_id = chunk_data.file_id
            
            chunks.append(Document(
                page_content=content,
                metadata={
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                },
            ))
        
        # Build knowledge graph
        kg = FoldexKnowledgeGraph()
        kg.build_from_documents(chunks)
        
        # Export to JSON
        graph_json = kg.to_json()
        
        logger.info(
            "Knowledge graph retrieved",
            folder_id=folder_id,
            node_count=graph_json["stats"]["node_count"],
            link_count=graph_json["stats"]["link_count"],
        )
        
        return graph_json
        
    except Exception as e:
        logger.error(
            "Failed to get knowledge graph",
            folder_id=folder_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build knowledge graph: {str(e)}",
        )


@router.get("/common-entities/{folder_id}")
async def get_common_entities(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Find entities shared across multiple documents in a folder.
    
    Args:
        folder_id: Folder identifier
        current_user: Current authenticated user
        
    Returns:
        List of common entities with their connected documents
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        # Get all chunks for this folder
        db = SQLiteManager()
        chunks_data = await db.get_chunks_by_folder(folder_id)
        
        if not chunks_data:
            return {
                "common_entities": [],
                "message": "No documents found in folder",
            }
        
        # Convert to LangChain Documents
        from langchain_core.documents import Document
        chunks = []
        for chunk_data in chunks_data:
            # Handle both dict and DocumentChunk objects
            if isinstance(chunk_data, dict):
                content = chunk_data.get("content", "")
                metadata = chunk_data.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.get("chunk_id")
                file_id = chunk_data.get("file_id")
            else:
                # DocumentChunk object
                content = chunk_data.content
                metadata = chunk_data.metadata or {}
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.chunk_id
                file_id = chunk_data.file_id
            
            chunks.append(Document(
                page_content=content,
                metadata={
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                },
            ))
        
        # Build knowledge graph
        kg = FoldexKnowledgeGraph()
        kg.build_from_documents(chunks)
        
        # Find common entities
        common_entities = kg.find_common_entities()
        
        logger.info(
            "Common entities retrieved",
            folder_id=folder_id,
            entity_count=len(common_entities),
        )
        
        return {
            "common_entities": common_entities,
            "total_shared": len(common_entities),
        }
        
    except Exception as e:
        logger.error(
            "Failed to get common entities",
            folder_id=folder_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find common entities: {str(e)}",
        )


@router.get("/entity-subgraph/{folder_id}")
async def get_entity_subgraph(
    folder_id: str,
    entity_name: str,
    depth: int = 2,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get subgraph centered on a specific entity.
    
    Args:
        folder_id: Folder identifier
        entity_name: Name of the entity to center on
        depth: Maximum distance from entity (default: 2)
        current_user: Current authenticated user
        
    Returns:
        Subgraph data in JSON format
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        # Get all chunks for this folder
        db = SQLiteManager()
        chunks_data = await db.get_chunks_by_folder(folder_id)
        
        if not chunks_data:
            return {
                "nodes": [],
                "links": [],
                "message": "No documents found in folder",
            }
        
        # Convert to LangChain Documents
        from langchain_core.documents import Document
        chunks = []
        for chunk_data in chunks_data:
            # Handle both dict and DocumentChunk objects
            if isinstance(chunk_data, dict):
                content = chunk_data.get("content", "")
                metadata = chunk_data.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.get("chunk_id")
                file_id = chunk_data.get("file_id")
            else:
                # DocumentChunk object
                content = chunk_data.content
                metadata = chunk_data.metadata or {}
                file_name = metadata.get("file_name", "unknown")
                chunk_id = chunk_data.chunk_id
                file_id = chunk_data.file_id
            
            chunks.append(Document(
                page_content=content,
                metadata={
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                },
            ))
        
        # Build knowledge graph
        kg = FoldexKnowledgeGraph()
        kg.build_from_documents(chunks)
        
        # Get subgraph
        subgraph = kg.get_entity_subgraph(entity_name, depth=depth)
        
        logger.info(
            "Entity subgraph retrieved",
            folder_id=folder_id,
            entity_name=entity_name,
            node_count=len(subgraph.get("nodes", [])),
        )
        
        return subgraph
        
    except Exception as e:
        logger.error(
            "Failed to get entity subgraph",
            folder_id=folder_id,
            entity_name=entity_name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entity subgraph: {str(e)}",
        )

