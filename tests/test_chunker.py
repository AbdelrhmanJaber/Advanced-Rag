import json
import unittest
from pathlib import Path

from src.preprocessing.tree_buillder import NaryTree
from src.preprocessing.chunker import DocumentChunker, chunk_document_tree


class TestDocumentChunker(unittest.TestCase):

    def setUp(self):
        self.tree = NaryTree(text="test_doc.pdf", node_type="DocumentRoot")
        # Structure:
        # DocumentRoot (node_0)
        #   ├── Section 1 (node_1, Title, lvl 1)
        #   │     ├── Paragraph 1.1 (node_2, Paragraph, lvl 4)
        #   │     └── Subsection 1.2 (node_3, Title, lvl 2)
        #   │           └── List item 1 (node_4, ListItem, lvl 5)
        #   └── Section 2 (node_5, Title, lvl 1)
        #         └── Table 2.1 (node_6, Table, lvl 4)
        self.sec1 = self.tree.insert("Section 1", "Title", 1, self.tree.root)
        self.p1_1 = self.tree.insert(
            "Paragraph 1.1 text", "Paragraph", 4, self.sec1, metadata={"page": 1}
        )
        self.subsec = self.tree.insert("Subsection 1.2", "Title", 2, self.sec1)
        self.item = self.tree.insert(
            "Item 1 text", "ListItem", 5, self.subsec, metadata={"page": 2}
        )
        self.sec2 = self.tree.insert("Section 2", "Title", 1, self.tree.root)
        self.tbl = self.tree.insert(
            "| A | B |",
            "Table",
            4,
            self.sec2,
            metadata={"page": 3, "table_markdown": "| A | B |\n|---|---|\n| 1 | 2 |"},
        )

    def test_flatten_tree_creates_chunks(self):
        chunker = DocumentChunker("test_doc.pdf")
        chunks = chunker.flatten_tree(self.tree)

        # 3 content nodes: p1_1, item, tbl
        self.assertEqual(len(chunks), 3)

        # First chunk: Paragraph 1.1
        c0 = chunks[0]
        self.assertEqual(c0["chunk_index"], 0)
        self.assertEqual(c0["text"], "Paragraph 1.1 text")
        self.assertIn("test_doc.pdf > Section 1", c0["embed_text"])
        self.assertEqual(c0["metadata"]["page"], 1)
        self.assertEqual(c0["metadata"]["parent_title"], "Section 1")
        self.assertEqual(c0["metadata"]["parent_id"], self.sec1.node_id)
        self.assertEqual(c0["metadata"]["node_type"], "Paragraph")
        self.assertFalse(c0["metadata"]["has_table"])

        # Second chunk: Item 1 under Subsection 1.2
        c1 = chunks[1]
        self.assertEqual(c1["chunk_index"], 1)
        self.assertEqual(c1["metadata"]["parent_title"], "Subsection 1.2")
        self.assertIn("test_doc.pdf > Section 1 > Subsection 1.2", c1["metadata"]["breadcrumb"])
        self.assertEqual(c1["metadata"]["parent_id"], self.subsec.node_id)
        self.assertEqual(c1["metadata"]["page"], 2)

        # Third chunk: Table
        c2 = chunks[2]
        self.assertEqual(c2["chunk_index"], 2)
        self.assertEqual(c2["metadata"]["parent_title"], "Section 2")
        self.assertTrue(c2["metadata"]["has_table"])
        self.assertIsNotNone(c2["metadata"]["table_markdown"])

    def test_flatten_dict_tree(self):
        tree_dict = self.tree.to_dict()
        chunker = DocumentChunker("test_doc.pdf")
        chunks = chunker.flatten_dict_tree(tree_dict)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["metadata"]["parent_title"], "Section 1")
        self.assertEqual(chunks[1]["metadata"]["parent_title"], "Subsection 1.2")
        self.assertEqual(chunks[2]["metadata"]["parent_title"], "Section 2")

    def test_save_chunks_json(self, tmp_path_factory=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_chunks.json"
            chunks = chunk_document_tree(self.tree, "test_doc.pdf", output_path=out_file)
            self.assertTrue(out_file.exists())
            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["chunk_id"], chunks[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
