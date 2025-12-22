"""
Knowledge graph API endpoints for Foldex.

Provides endpoints to query and visualize the knowledge graph
built from folder documents.
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from typing import Dict, Any
import structlog
from sqlalchemy import select

from app.api.deps import get_current_user
from app.api.v1.websocket import manager
from app.database.sqlite_manager import SQLiteManager
from app.services.langgraph_knowledge_service import get_knowledge_service
from app.knowledge_graph.graph_builder import FoldexKnowledgeGraph  # For query endpoints
from app.models.database import FolderRecord
from app.utils.datetime_utils import get_eastern_time
from sqlalchemy import update

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
        
        # Get stored graph data from database
        db = SQLiteManager()
        graph_data_bytes = await db.get_knowledge_graph(folder_id)
        
        if graph_data_bytes:
            # Decode stored JSON graph data
            import json
            try:
                graph_json = json.loads(graph_data_bytes.decode('utf-8'))
                logger.info(
                    "Knowledge graph retrieved from database",
                    folder_id=folder_id,
                    node_count=graph_json.get("stats", {}).get("node_count", 0),
                    link_count=graph_json.get("stats", {}).get("link_count", 0),
                )
                return graph_json
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(
                    "Failed to decode stored graph data, will rebuild",
                    folder_id=folder_id,
                    error=str(e),
                )
                # Fall through to rebuild if decoding fails
        
        # Graph not yet built or failed to decode - return empty graph
        # Frontend can show a message that graph is being built
        return {
            "nodes": [],
            "links": [],
            "stats": {
                "node_count": 0,
                "link_count": 0,
                "document_count": 0,
            },
            "message": "Knowledge graph is being built. Please check back in a moment.",
            "building": True,
        }
        
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


@router.post("/{folder_id}/build")
async def build_knowledge_graph(
    folder_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Build knowledge graph for a folder (user-initiated).
    
    This endpoint triggers knowledge graph building which may take over 1 minute
    for large folders. The graph will be built in the background and progress
    will be sent via WebSocket.
    
    Args:
        folder_id: Folder identifier
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        
    Returns:
        Status message
        
    Raises:
        HTTPException: If folder not found or build fails
    """
    try:
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        # Verify folder exists
        db = SQLiteManager()
        async with db._get_db_manager().get_session() as session:
            folder_stmt = select(FolderRecord).where(FolderRecord.folder_id == folder_id)
            folder_result = await session.execute(folder_stmt)
            folder = folder_result.scalar_one_or_none()
            
            if not folder:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Folder not found",
                )
        
        # Build knowledge graph in background
        async def build_task():
            """Background task to build knowledge graph."""
            try:
                # Send building status
                await manager.send_message(
                    folder_id,
                    {
                        "type": "building_graph",
                        "folder_id": folder_id,
                        "message": "Building knowledge graph...",
                    },
                )
                
                # Get chunks (already returns DocumentChunk objects)
                chunks = await db.get_chunks_by_folder(folder_id, include_subfolders=True)
                
                if not chunks:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "graph_error",
                            "folder_id": folder_id,
                            "error": "No documents found in folder",
                        },
                    )
                    return
                
                # Build knowledge graph using LangGraph workflow
                # IMPORTANT: LangGraph workflow already extracts entities and relationships
                # We don't need to do entity extraction again with FoldexKnowledgeGraph
                logger.info(
                    "Starting knowledge graph build via LangGraph service",
                    folder_id=folder_id,
                    chunk_count=len(chunks),
                )
                
                knowledge_service = get_knowledge_service()
                graph = await knowledge_service.build_knowledge_graph(
                    documents=chunks,
                    folder_id=folder_id,
                    resume_from_checkpoint=False,
                )
                
                logger.info(
                    "Knowledge graph build completed",
                    folder_id=folder_id,
                    node_count=graph.number_of_nodes(),
                    edge_count=graph.number_of_edges(),
                )

                # Convert graph to JSON format for storage
                import json
                # Use to_json method if available, otherwise build manually
                if hasattr(graph, 'to_json'):
                    graph_json = graph.to_json()
                else:
                    # Fallback: build JSON from NetworkX graph
                    graph_json = {
                        "nodes": [
                            {
                                "id": node,
                                "label": graph.nodes[node].get("label", node),
                                "type": graph.nodes[node].get("type", "entity"),
                                "node_type": graph.nodes[node].get("node_type", "entity"),
                            }
                            for node in graph.nodes()
                        ],
                        "links": [
                            {
                                "source": u,
                                "target": v,
                                "relation": graph.edges[u, v].get("relation", "related"),
                            }
                            for u, v in graph.edges()
                        ],
                        "stats": {
                            "node_count": graph.number_of_nodes(),
                            "edge_count": graph.number_of_edges(),
                            "relationship_types": len(set(
                                graph.edges[u, v].get("relation", "related")
                                for u, v in graph.edges()
                            )),
                        },
                    }
                
                graph_stats = graph_json.get("stats", {})
                
                # Store graph in database
                graph_data_json = json.dumps(graph_json).encode('utf-8')
                await db.store_knowledge_graph(folder_id, graph_data_json)
                
                # Update folder summary's graph_statistics field
                # This ensures the frontend can detect that the graph exists
                try:
                    # Update graph_statistics while preserving other fields
                    async with db._get_db_manager().get_session() as session:
                        stmt = update(FolderRecord).where(
                            FolderRecord.folder_id == folder_id
                        ).values(
                            graph_statistics=graph_stats,
                            updated_at=get_eastern_time()
                        )
                        await session.execute(stmt)
                        await session.commit()
                        logger.info("Updated folder graph_statistics", folder_id=folder_id)
                except Exception as e:
                    logger.warning(
                        "Failed to update folder graph_statistics",
                        folder_id=folder_id,
                        error=str(e)
                    )
                    # Don't fail the whole operation if updating stats fails
                
                # Send completion status
                await manager.send_message(
                    folder_id,
                    {
                        "type": "graph_complete",
                        "folder_id": folder_id,
                        "message": "Knowledge graph built successfully",
                        "graph_stats": graph_stats,
                    },
                )
            except Exception as e:
                logger.error(
                    "Background knowledge graph build failed",
                    folder_id=folder_id,
                    error=str(e),
                )
                await manager.send_message(
                    folder_id,
                    {
                        "type": "graph_error",
                        "folder_id": folder_id,
                        "error": str(e),
                    },
                )
        
        background_tasks.add_task(build_task)
        
        return {
            "message": "Knowledge graph build started",
            "folder_id": folder_id,
            "status": "building",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start knowledge graph build",
            folder_id=folder_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start knowledge graph build: {str(e)}",
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
        
        # Build knowledge graph (async)
        kg = FoldexKnowledgeGraph()
        await kg.build_from_documents(chunks)
        
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
        
        # Build knowledge graph (async)
        kg = FoldexKnowledgeGraph()
        await kg.build_from_documents(chunks)
        
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

