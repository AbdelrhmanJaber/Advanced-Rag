from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.preprocessing.tree_buillder import NaryTree, Node


class DocumentChunker:
    """
    Transforms an N-ary Document Tree into flat, embedding-ready chunks
    with full contextual metadata (breadcrumbs, parent lineage, sequential indexing,
    and type flags) to solve orphan chunking and support Parent-Child RAG.
    """

    def __init__(self, doc_name: str) -> None:
        self.doc_name = doc_name

    def flatten_tree(self, tree: NaryTree) -> List[Dict[str, Any]]:
        """Flatten an NaryTree instance into a list of chunk dictionaries."""
        chunks: List[Dict[str, Any]] = []
        counter = [0]

        def _walk(node: Node, ancestors: List[str], active_section: str) -> None:
            # When encountering a section title, track it in the active path
            if node.node_type == "Title":
                active_section = node.text
                current_ancestors = ancestors + [node.text]
            else:
                current_ancestors = ancestors

            # Content leaf elements become searchable/embeddable chunks
            if node.node_type in ("Paragraph", "ListItem", "Table", "Image"):
                breadcrumb_str = " > ".join([self.doc_name] + current_ancestors)
                meta = node.metadata or {}

                # Text prepended with breadcrumb to eliminate orphan-chunk ambiguity
                embed_text = f"[{breadcrumb_str}]\n{node.text}"

                chunk_dict: Dict[str, Any] = {
                    "chunk_id": f"{self.doc_name}_{node.node_id}",
                    "chunk_index": counter[0],
                    "text": node.text,
                    "embed_text": embed_text,
                    "metadata": {
                        "doc_name": self.doc_name,
                        "page": meta.get("page", 1),
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "parent_title": active_section,
                        "breadcrumb": breadcrumb_str,
                        "level": node.level,
                        "node_type": node.node_type,
                        "char_count": len(node.text),
                        "token_count_est": max(1, len(node.text.split()) * 4 // 3),
                        "has_table": node.node_type == "Table",
                        "has_image": node.node_type == "Image",
                        "table_markdown": meta.get("table_markdown"),
                        "table_html": meta.get("table_html"),
                        "image_path": meta.get("image_path"),
                        "caption": meta.get("caption"),
                    },
                }
                chunks.append(chunk_dict)
                counter[0] += 1

            for child in node.children:
                _walk(child, current_ancestors, active_section)

        _walk(tree.root, [], "DocumentRoot")
        return chunks

    def flatten_dict_tree(self, tree_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten a serialized tree dictionary (from _tree.json) into chunks."""
        chunks: List[Dict[str, Any]] = []
        counter = [0]

        def _walk(node_dict: Dict[str, Any], ancestors: List[str], active_section: str) -> None:
            ntype = node_dict.get("node_type", "")
            text = node_dict.get("text", "")
            node_id = node_dict.get("node_id", "")
            parent_id = node_dict.get("parent_id")
            level = node_dict.get("level", 4)
            meta = node_dict.get("metadata", {})

            if ntype == "Title":
                active_section = text
                current_ancestors = ancestors + [text]
            else:
                current_ancestors = ancestors

            if ntype in ("Paragraph", "ListItem", "Table", "Image"):
                breadcrumb_str = " > ".join([self.doc_name] + current_ancestors)
                embed_text = f"[{breadcrumb_str}]\n{text}"

                chunk_dict: Dict[str, Any] = {
                    "chunk_id": f"{self.doc_name}_{node_id}",
                    "chunk_index": counter[0],
                    "text": text,
                    "embed_text": embed_text,
                    "metadata": {
                        "doc_name": self.doc_name,
                        "page": meta.get("page", 1),
                        "node_id": node_id,
                        "parent_id": parent_id,
                        "parent_title": active_section,
                        "breadcrumb": breadcrumb_str,
                        "level": level,
                        "node_type": ntype,
                        "char_count": len(text),
                        "token_count_est": max(1, len(text.split()) * 4 // 3),
                        "has_table": ntype == "Table",
                        "has_image": ntype == "Image",
                        "table_markdown": meta.get("table_markdown"),
                        "table_html": meta.get("table_html"),
                        "image_path": meta.get("image_path"),
                        "caption": meta.get("caption"),
                    },
                }
                chunks.append(chunk_dict)
                counter[0] += 1

            for child in node_dict.get("children", []):
                _walk(child, current_ancestors, active_section)

        _walk(tree_dict, [], "DocumentRoot")
        return chunks

    def save_chunks_json(
        self,
        chunks: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> str:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(out_path.resolve())


def chunk_document_tree(
    tree_or_dict: Union[NaryTree, Dict[str, Any]],
    doc_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Helper function to chunk a tree (or tree dict) and optionally save to disk."""
    chunker = DocumentChunker(doc_name=doc_name)
    if isinstance(tree_or_dict, NaryTree):
        chunks = chunker.flatten_tree(tree_or_dict)
    else:
        chunks = chunker.flatten_dict_tree(tree_or_dict)

    if output_path is not None:
        chunker.save_chunks_json(chunks, output_path)

    return chunks
