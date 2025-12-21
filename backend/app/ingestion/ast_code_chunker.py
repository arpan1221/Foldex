"""
AST-based code chunker for intelligent code file processing.

Uses Abstract Syntax Tree parsing to chunk code files by functions, classes, and methods,
preserving code structure and enabling better retrieval for code-specific queries.

Supports:
- Python: Uses built-in `ast` module
- JavaScript/TypeScript: Uses tree-sitter (if available)
- Other languages: Falls back to text-based chunking
"""

from typing import List, Dict, Any, Optional, Tuple
import ast
import re
import structlog
from pathlib import Path

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None

# Try to import tree-sitter for multi-language support
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None

logger = structlog.get_logger(__name__)


class ASTCodeChunker:
    """Chunk code files using AST parsing to preserve structure.
    
    Supports multiple languages:
    - Python: Uses built-in `ast` module
    - JavaScript/TypeScript: Uses tree-sitter (if available)
    - Other languages: Falls back to text-based chunking
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """Initialize AST code chunker.

        Args:
            max_chunk_size: Maximum characters per chunk (functions/classes larger than this will be split)
            min_chunk_size: Minimum characters per chunk (small functions will be combined)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize tree-sitter parsers if available
        self.tree_sitter_parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._initialize_tree_sitter_parsers()

    def _initialize_tree_sitter_parsers(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        try:
            # Note: tree-sitter requires compiled language grammars
            # For now, we'll try to use them if available, but they need to be built separately
            # This is a placeholder - in production, you'd need to build the language grammars
            logger.debug("Tree-sitter available, but language grammars must be built separately")
        except Exception as e:
            logger.warning("Failed to initialize tree-sitter parsers", error=str(e))

    def chunk_code(
        self,
        content: str,
        file_metadata: Dict[str, Any],
        language: str,
    ) -> List[Document]:
        """Chunk code file using AST parsing for supported languages.

        Args:
            content: Source code content
            file_metadata: File metadata dictionary
            language: Programming language (python, javascript, typescript, etc.)

        Returns:
            List of Document objects with AST-based chunks
        """
        language_lower = language.lower()
        
        # Python: Use built-in AST module
        if language_lower == "python":
            return self.chunk_python(content, file_metadata)
        
        # JavaScript/TypeScript: Use tree-sitter if available
        elif language_lower in ["javascript", "typescript", "js", "ts"]:
            if TREE_SITTER_AVAILABLE:
                try:
                    return self.chunk_javascript_typescript(content, file_metadata, language_lower)
                except Exception as e:
                    logger.warning(
                        "Tree-sitter chunking failed, falling back to text-based",
                        language=language,
                        error=str(e),
                    )
            # Fall back to text-based chunking
            return self._fallback_text_chunking(content, file_metadata)
        
        # Other languages: Fall back to text-based chunking
        else:
            logger.debug(
                "Language not yet supported for AST chunking, using text-based",
                language=language,
            )
            return self._fallback_text_chunking(content, file_metadata)

    def chunk_python(self, content: str, file_metadata: Dict[str, Any]) -> List[Document]:
        """Chunk Python code using AST parsing.

        Args:
            content: Python source code content
            file_metadata: File metadata dictionary

        Returns:
            List of Document objects with AST-based chunks
        """
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Extract code units (functions, classes, top-level code)
            code_units = self._extract_code_units(tree, content)
            
            # Group small units and split large ones
            chunks = self._create_chunks_from_units(code_units, content, file_metadata)
            
            logger.info(
                "AST chunking completed",
                file_path=file_metadata.get("file_name", "unknown"),
                code_units=len(code_units),
                chunks=len(chunks),
            )
            
            return chunks
            
        except SyntaxError as e:
            logger.warning(
                "Python AST parsing failed (syntax error), falling back to text-based chunking",
                error=str(e),
                file_path=file_metadata.get("file_name", "unknown"),
            )
            return self._fallback_text_chunking(content, file_metadata)
        except Exception as e:
            logger.warning(
                "Python AST parsing failed, falling back to text-based chunking",
                error=str(e),
                file_path=file_metadata.get("file_name", "unknown"),
            )
            return self._fallback_text_chunking(content, file_metadata)

    def _extract_code_units(
        self, tree: ast.AST, content: str
    ) -> List[Dict[str, Any]]:
        """Extract code units (functions, classes, top-level code) from AST.

        Args:
            tree: Parsed AST tree
            content: Original source code

        Returns:
            List of code unit dictionaries with type, name, start_line, end_line
        """
        code_units = []
        lines = content.splitlines(keepends=True)
        
        # Track top-level code (code not inside functions/classes)
        top_level_start = 1
        
        # Process only top-level nodes (not nested)
        for node in ast.iter_child_nodes(tree):
            # Only process top-level nodes (functions, classes)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            
            # Get node location
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            
            # If end_lineno is not available, estimate from node structure
            if not end_line:
                # Walk the node to find the last line
                last_line = start_line
                for child in ast.walk(node):
                    if hasattr(child, 'lineno') and child.lineno:
                        last_line = max(last_line, child.lineno)
                end_line = last_line
            
            # Extract code text
            if end_line and start_line <= len(lines):
                code_text = ''.join(lines[start_line - 1:end_line])
            else:
                # Fallback: use ast.get_source_segment if available (Python 3.8+)
                try:
                    code_text = ast.get_source_segment(content, node) or ""
                except AttributeError:
                    # Python < 3.8: estimate from line numbers
                    code_text = '\n'.join(lines[start_line - 1:end_line or start_line])
            
            # Determine unit type and name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                unit_type = "function"
                unit_name = node.name
                # Check if it's a method (has decorators or is inside a class context)
                if node.decorator_list:
                    # Could be a method, but we'll handle classes separately
                    pass
            elif isinstance(node, ast.ClassDef):
                unit_type = "class"
                unit_name = node.name
            else:
                continue
            
            code_units.append({
                "type": unit_type,
                "name": unit_name,
                "start_line": start_line,
                "end_line": end_line,
                "code": code_text,
                "node": node,
            })
            
            # Update top-level start for next iteration
            top_level_start = max(top_level_start, end_line + 1 if end_line else start_line + 1)
        
        # Sort by start line
        code_units.sort(key=lambda x: x["start_line"])
        
        # Add top-level code (imports, module-level code) as first unit if any
        if code_units and code_units[0]["start_line"] > 1:
            top_level_end = code_units[0]["start_line"] - 1
            if top_level_end >= 1:
                top_level_code = ''.join(lines[0:top_level_end])
                if top_level_code.strip():
                    code_units.insert(0, {
                        "type": "module",
                        "name": "__init__",
                        "start_line": 1,
                        "end_line": top_level_end,
                        "code": top_level_code,
                        "node": None,
                    })
        
        return code_units


    def _create_chunks_from_units(
        self,
        code_units: List[Dict[str, Any]],
        content: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Create chunks from code units, grouping small ones and splitting large ones.

        Args:
            code_units: List of code unit dictionaries
            content: Original source code
            file_metadata: File metadata

        Returns:
            List of Document objects
        """
        chunks = []
        current_chunk_units = []
        current_chunk_size = 0
        
        for unit in code_units:
            unit_size = len(unit["code"])
            
            # If unit is too large, split it
            if unit_size > self.max_chunk_size:
                # Flush current chunk if any
                if current_chunk_units:
                    chunks.append(self._create_chunk_from_units(
                        current_chunk_units, file_metadata
                    ))
                    current_chunk_units = []
                    current_chunk_size = 0
                
                # Split large unit
                sub_chunks = self._split_large_unit(unit, file_metadata)
                chunks.extend(sub_chunks)
                continue
            
            # Check if we should start a new chunk
            if (current_chunk_size + unit_size > self.max_chunk_size and
                current_chunk_units):
                # Current chunk is full, flush it
                chunks.append(self._create_chunk_from_units(
                    current_chunk_units, file_metadata
                ))
                current_chunk_units = []
                current_chunk_size = 0
            
            # Add unit to current chunk
            current_chunk_units.append(unit)
            current_chunk_size += unit_size
        
        # Flush remaining units
        if current_chunk_units:
            chunks.append(self._create_chunk_from_units(
                current_chunk_units, file_metadata
            ))
        
        return chunks

    def _create_chunk_from_units(
        self,
        units: List[Dict[str, Any]],
        file_metadata: Dict[str, Any],
    ) -> Document:
        """Create a Document chunk from code units.

        Args:
            units: List of code unit dictionaries
            file_metadata: File metadata

        Returns:
            Document object
        """
        # Combine code from all units
        chunk_code = '\n\n'.join(unit["code"] for unit in units)
        
        # Determine chunk metadata
        if len(units) == 1:
            unit = units[0]
            chunk_type = unit["type"]
            primary_name = unit["name"]
            start_line = unit["start_line"]
            end_line = unit["end_line"]
        else:
            # Multiple units in one chunk
            chunk_type = "mixed"
            primary_name = f"{units[0]['name']}_and_{len(units) - 1}_more"
            start_line = units[0]["start_line"]
            end_line = units[-1]["end_line"]
        
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(chunk_code, file_metadata.get("file_id", "unknown"))
        
        # Build metadata
        chunk_metadata = {
            "chunk_id": chunk_id,
            "file_id": file_metadata.get("file_id"),
            "file_name": file_metadata.get("file_name"),
            "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
            "language": "python",
            "chunk_type": chunk_type,
            "function_name": primary_name if chunk_type == "function" else None,
            "class_name": primary_name if chunk_type == "class" else None,
            "line_start": start_line,
            "line_end": end_line,
            "mime_type": file_metadata.get("mime_type", "text/x-python"),
            "source": file_metadata.get("source", ""),
        }
        
        # Add unit names for mixed chunks
        if chunk_type == "mixed":
            chunk_metadata["unit_names"] = [u["name"] for u in units]
            chunk_metadata["unit_types"] = [u["type"] for u in units]
        
        # Add any additional file metadata
        for key in ["folder_id", "user_id", "created_at", "modified_at"]:
            if key in file_metadata:
                chunk_metadata[key] = file_metadata[key]
        
        return Document(
            page_content=chunk_code,
            metadata=chunk_metadata,
        )

    def _split_large_unit(
        self,
        unit: Dict[str, Any],
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Split a large code unit into smaller chunks.

        Args:
            unit: Code unit dictionary
            file_metadata: File metadata

        Returns:
            List of Document objects
        """
        chunks = []
        code = unit["code"]
        lines = code.splitlines(keepends=True)
        
        # For functions/classes, try to split by logical sections
        # Otherwise, split by line count
        chunk_size_lines = max(20, self.max_chunk_size // 80)  # Approximate lines per chunk
        
        current_chunk_lines = []
        current_line_start = unit["start_line"]
        chunk_index = 0
        
        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            
            if len(current_chunk_lines) >= chunk_size_lines:
                # Create chunk
                chunk_code = ''.join(current_chunk_lines)
                chunk_id = self._generate_chunk_id(
                    chunk_code,
                    file_metadata.get("file_id", "unknown"),
                    suffix=f"_{chunk_index}",
                )
                
                chunk_end_line = current_line_start + len(current_chunk_lines) - 1
                
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "file_id": file_metadata.get("file_id"),
                    "file_name": file_metadata.get("file_name"),
                    "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                    "language": "python",
                    "chunk_type": f"{unit['type']}_part",
                    "function_name": unit["name"] if unit["type"] == "function" else None,
                    "class_name": unit["name"] if unit["type"] == "class" else None,
                    "line_start": current_line_start,
                    "line_end": chunk_end_line,
                    "part_index": chunk_index,
                    "mime_type": file_metadata.get("mime_type", "text/x-python"),
                    "source": file_metadata.get("source", ""),
                }
                
                chunks.append(Document(
                    page_content=chunk_code,
                    metadata=chunk_metadata,
                ))
                
                # Update line start for next chunk
                current_line_start = chunk_end_line + 1
                current_chunk_lines = []
                chunk_index += 1
        
        # Add remaining lines
        if current_chunk_lines:
            chunk_code = ''.join(current_chunk_lines)
            chunk_id = self._generate_chunk_id(
                chunk_code,
                file_metadata.get("file_id", "unknown"),
                suffix=f"_{chunk_index}",
            )
            
            chunk_metadata = {
                "chunk_id": chunk_id,
                "file_id": file_metadata.get("file_id"),
                "file_name": file_metadata.get("file_name"),
                "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                "language": "python",
                "chunk_type": f"{unit['type']}_part",
                "function_name": unit["name"] if unit["type"] == "function" else None,
                "class_name": unit["name"] if unit["type"] == "class" else None,
                "line_start": current_line_start,
                "line_end": unit["end_line"],
                "part_index": chunk_index,
                "mime_type": file_metadata.get("mime_type", "text/x-python"),
                "source": file_metadata.get("source", ""),
            }
            
            chunks.append(Document(
                page_content=chunk_code,
                metadata=chunk_metadata,
            ))
        
        return chunks

    def chunk_javascript_typescript(
        self,
        content: str,
        file_metadata: Dict[str, Any],
        language: str,
    ) -> List[Document]:
        """Chunk JavaScript/TypeScript code using tree-sitter.

        Args:
            content: JavaScript/TypeScript source code content
            file_metadata: File metadata dictionary
            language: Language identifier (javascript, typescript, js, ts)

        Returns:
            List of Document objects with AST-based chunks
        """
        # Note: This requires tree-sitter language grammars to be built
        # For now, fall back to text-based chunking
        # In a full implementation, you would:
        # 1. Load the appropriate tree-sitter grammar
        # 2. Parse the code into an AST
        # 3. Extract functions, classes, methods
        # 4. Create chunks preserving structure
        
        logger.debug(
            "Tree-sitter chunking not yet fully implemented, using text-based fallback",
            language=language,
        )
        return self._fallback_text_chunking(content, file_metadata)

    def _fallback_text_chunking(
        self,
        content: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Fallback to simple text-based chunking if AST parsing fails.

        Args:
            content: Source code content
            file_metadata: File metadata

        Returns:
            List of Document objects
        """
        # Simple line-based chunking
        lines = content.splitlines(keepends=True)
        chunks = []
        chunk_size_lines = max(50, self.max_chunk_size // 80)
        
        for i in range(0, len(lines), chunk_size_lines):
            chunk_lines = lines[i:i + chunk_size_lines]
            chunk_code = ''.join(chunk_lines)
            
            chunk_id = self._generate_chunk_id(
                chunk_code,
                file_metadata.get("file_id", "unknown"),
                suffix=f"_fallback_{i // chunk_size_lines}",
            )
            
            chunk_metadata = {
                "chunk_id": chunk_id,
                "file_id": file_metadata.get("file_id"),
                "file_name": file_metadata.get("file_name"),
                "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                "language": "python",
                "chunk_type": "module",
                "line_start": i + 1,
                "line_end": min(i + chunk_size_lines, len(lines)),
                "mime_type": file_metadata.get("mime_type", "text/x-python"),
                "source": file_metadata.get("source", ""),
            }
            
            chunks.append(Document(
                page_content=chunk_code,
                metadata=chunk_metadata,
            ))
        
        return chunks

    def _generate_chunk_id(self, content: str, file_id: str, suffix: str = "") -> str:
        """Generate a unique chunk ID from content.

        Args:
            content: Chunk content
            file_id: File identifier
            suffix: Optional suffix for uniqueness

        Returns:
            Chunk ID string
        """
        import hashlib
        
        # Create hash from content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"{file_id}_chunk_{content_hash}{suffix}"

