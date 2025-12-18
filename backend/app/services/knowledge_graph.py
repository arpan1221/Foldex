"""Knowledge graph construction service."""

from typing import List, Dict
import structlog

from app.knowledge_graph.graph_builder import GraphBuilder
from app.knowledge_graph.relationship_detector import RelationshipDetector
from app.database.sqlite_manager import SQLiteManager

logger = structlog.get_logger(__name__)


class KnowledgeGraphService:
    """Service for building and managing knowledge graphs."""

    def __init__(
        self,
        graph_builder: GraphBuilder | None = None,
        relationship_detector: RelationshipDetector | None = None,
        db: SQLiteManager | None = None,
    ):
        """Initialize knowledge graph service.

        Args:
            graph_builder: Graph builder instance
            relationship_detector: Relationship detector instance
            db: Database manager
        """
        self.graph_builder = graph_builder or GraphBuilder()
        self.relationship_detector = relationship_detector or RelationshipDetector()
        self.db = db or SQLiteManager()

    async def build_graph(self, folder_id: str) -> None:
        """Build knowledge graph for a folder.

        Args:
            folder_id: Google Drive folder ID
        """
        try:
            logger.info("Building knowledge graph", folder_id=folder_id)

            # Get all chunks for folder
            chunks = await self.db.get_chunks_by_folder(folder_id)

            # Detect relationships
            relationships = await self.relationship_detector.detect_relationships(
                chunks
            )

            # Build graph
            graph = await self.graph_builder.build_graph(chunks, relationships)

            # Store graph
            await self.db.store_knowledge_graph(folder_id, graph)

            logger.info(
                "Knowledge graph built",
                folder_id=folder_id,
                relationship_count=len(relationships),
            )
        except Exception as e:
            logger.error(
                "Knowledge graph construction failed",
                folder_id=folder_id,
                error=str(e),
            )
            raise

