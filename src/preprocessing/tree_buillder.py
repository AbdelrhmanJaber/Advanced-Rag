from __future__ import annotations
import json
from typing import Any, Dict, List, Optional


class Node:
    def __init__(
        self,
        text: str,
        node_type: str,
        level: int,
        parent: Optional[Node] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> None:
        self.node_id: Optional[str] = node_id
        self.text: str = text
        self.node_type: str = node_type
        self.level: int = level
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        # Extra payload: image_path, table_markdown, table_html, …
        self.metadata: Dict[str, Any] = metadata or {}

    @property
    def parent_id(self) -> Optional[str]:
        return self.parent.node_id if self.parent else None

    def add_child(self, child: Node) -> None:
        child.parent = self
        self.children.append(child)


class NaryTree:
    def __init__(self, text: str, node_type: str) -> None:
        self._counter: int = 0
        self.root: Node = Node(text, node_type, level=0, node_id=f"node_{self._counter}")

    def insert(
        self,
        text: str,
        node_type: str,
        level: int,
        parent: Node,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> Node:
        self._counter += 1
        if node_id is None:
            node_id = f"node_{self._counter}"
        node = Node(text, node_type, level, parent, metadata=metadata, node_id=node_id)
        parent.children.append(node)
        return node

    def pre_order_traversal(self, node: Optional[Node] = None) -> None:
        if node is None:
            node = self.root
        print(
            "text:", node.text,
            "| node_type:", node.node_type,
            "| level:", node.level,
            "| metadata:", node.metadata,
        )
        for child in node.children:
            self.pre_order_traversal(child)

    def to_dict(self, node: Optional[Node] = None) -> Dict[str, Any]:
        if node is None:
            node = self.root
        d: Dict[str, Any] = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "text": node.text,
            "node_type": node.node_type,
            "level": node.level,
            "children": [self.to_dict(child) for child in node.children],
        }
        if node.metadata:
            d["metadata"] = node.metadata
        return d

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
