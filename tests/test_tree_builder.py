import sys
from pathlib import Path

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import unittest
from io import StringIO
from unittest.mock import patch

from src.preprocessing.tree_buillder import Node, NaryTree


class TestTreeBuilder(unittest.TestCase):

    def setUp(self):
        # Create a basic tree structure for testing:
        # Document (root)
        #  ├── Section 1 (Header)
        #  │    ├── Paragraph 1.1
        #  │    └── Paragraph 1.2
        #  └── Section 2 (Header)
        #       └── Paragraph 2.1
        self.tree = NaryTree(text="Document Root", node_type="document")
        self.sec1 = self.tree.insert(text="Section 1", node_type="header", level=1, parent=self.tree.root)
        self.p1_1 = self.tree.insert(text="Paragraph 1.1", node_type="paragraph", level=2, parent=self.sec1)
        self.p1_2 = self.tree.insert(text="Paragraph 1.2", node_type="paragraph", level=2, parent=self.sec1)
        self.sec2 = self.tree.insert(text="Section 2", node_type="header", level=1, parent=self.tree.root)
        self.p2_1 = self.tree.insert(text="Paragraph 2.1", node_type="paragraph", level=2, parent=self.sec2)

    def test_root_initialization(self):
        """Test NaryTree root creation."""
        self.assertEqual(self.tree.root.text, "Document Root")
        self.assertEqual(self.tree.root.node_type, "document")
        self.assertEqual(self.tree.root.level, 0)
        self.assertIsNone(self.tree.root.parent)
        self.assertEqual(len(self.tree.root.children), 2)

    def test_node_add_child(self):
        """Test direct node.add_child functionality."""
        standalone_parent = Node(text="Parent", node_type="section", level=1)
        standalone_child = Node(text="Child", node_type="paragraph", level=2)
        standalone_parent.add_child(standalone_child)

        self.assertIn(standalone_child, standalone_parent.children)
        self.assertEqual(standalone_child.parent, standalone_parent)

    def test_tree_structure_and_relationships(self):
        """Test hierarchical parent-child relationships after insertion."""
        self.assertEqual(self.sec1.parent, self.tree.root)
        self.assertEqual(len(self.sec1.children), 2)
        self.assertIn(self.p1_1, self.sec1.children)
        self.assertIn(self.p1_2, self.sec1.children)

        self.assertEqual(self.sec2.parent, self.tree.root)
        self.assertEqual(len(self.sec2.children), 1)
        self.assertIn(self.p2_1, self.sec2.children)

    def test_pre_order_traversal(self):
        """Test pre-order traversal output formatting."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.tree.pre_order_traversal()
            output = fake_out.getvalue()

        lines = output.strip().split('\n')
        self.assertEqual(len(lines), 6)
        self.assertIn("Document Root", lines[0])
        self.assertIn("Section 1", lines[1])
        self.assertIn("Paragraph 1.1", lines[2])

    def test_to_dict_conversion(self):
        """Test dictionary representation of the document tree."""
        tree_dict = self.tree.to_dict()

        self.assertEqual(tree_dict["text"], "Document Root")
        self.assertEqual(tree_dict["node_type"], "document")
        self.assertEqual(tree_dict["level"], 0)
        self.assertEqual(tree_dict["node_id"], "node_0")
        self.assertIsNone(tree_dict["parent_id"])
        self.assertEqual(len(tree_dict["children"]), 2)

        sec1_dict = tree_dict["children"][0]
        self.assertEqual(sec1_dict["text"], "Section 1")
        self.assertEqual(sec1_dict["parent_id"], "node_0")
        self.assertEqual(len(sec1_dict["children"]), 2)
        self.assertEqual(sec1_dict["children"][0]["text"], "Paragraph 1.1")
        self.assertEqual(sec1_dict["children"][0]["parent_id"], sec1_dict["node_id"])

    def test_to_json_serialization(self):
        """Test JSON serialization string format and deserialization."""
        json_str = self.tree.to_json()
        self.assertIsInstance(json_str, str)

        parsed_data = json.loads(json_str)
        self.assertEqual(parsed_data["text"], "Document Root")
        self.assertEqual(parsed_data["node_id"], "node_0")
        self.assertEqual(parsed_data["children"][0]["text"], "Section 1")
        self.assertEqual(parsed_data["children"][0]["parent_id"], "node_0")
        self.assertEqual(len(parsed_data["children"][0]["children"]), 2)


if __name__ == "__main__":
    unittest.main()
