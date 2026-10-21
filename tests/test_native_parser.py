"""
test_native_parser.py
=====================
Integration tests for the full ingestion pipeline:

    LocalFileLoader  →  validator  →  classifier  →  NativeDocumentParser  →  JSON

Tests use synthetic PDFs built in memory with PyMuPDF so no external
fixture files are required.
"""

import sys
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Make sure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pymupdf  # type: ignore

from src.database.metadata_database import init_database, DB_PATH
from src.ingestion.validator import validator, calculate_hash_file
from src.preprocessing.classifier import classifier
from src.preprocessing.native_parser import NativeDocumentParser, run_native_parser
from src.preprocessing.tree_buillder import NaryTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_native_pdf(path: Path, text: str = "") -> None:
    """Create a one-page text PDF with enough characters to be NATIVE."""
    long_text = text or (
        "Introduction\n\n"
        "This document is a native PDF with selectable text. "
        "It contains more than 50 characters so the classifier marks it as NATIVE. "
        "Section 1: Background\n\n"
        "Background details go here. They form a paragraph of narrative text. "
        "Section 2: Method\n\n"
        "The method is described in this section with further detail."
    )
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((50, 72), long_text, fontsize=11)
    doc.save(str(path))
    doc.close()


def _make_native_pdf_with_table(path: Path) -> None:
    """Create a PDF that PyMuPDF can detect a table in."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Draw a simple grid that PyMuPDF recognises as a table
    for col in (50, 200, 350):
        page.draw_line((col, 100), (col, 160))
    for row in (100, 130, 160):
        page.draw_line((50, row), (350, row))
    # Cell text
    page.insert_text((55, 120), "Header A", fontsize=9)
    page.insert_text((205, 120), "Header B", fontsize=9)
    page.insert_text((355, 120), "Header C", fontsize=9)
    page.insert_text((55, 150), "Row 1 A", fontsize=9)
    page.insert_text((205, 150), "Row 1 B", fontsize=9)
    page.insert_text((355, 150), "Row 1 C", fontsize=9)
    # Add plenty of surrounding text so classifier sees NATIVE
    page.insert_text((50, 200), "A" * 200, fontsize=9)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Unit tests for NativeDocumentParser in isolation
# ---------------------------------------------------------------------------

class TestNativeDocumentParserUnit(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    def test_file_not_found_raises(self) -> None:
        """Constructor must raise FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            NativeDocumentParser("/non/existent/file.pdf")

    # ------------------------------------------------------------------
    def test_parse_returns_nary_tree(self) -> None:
        """parse_to_tree() must return an NaryTree instance."""
        pdf_path = self.tmp_path / "native.pdf"
        _make_native_pdf(pdf_path)

        parser = NativeDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        tree = parser.parse_to_tree()
        self.assertIsInstance(tree, NaryTree)

    # ------------------------------------------------------------------
    def test_root_node_is_document_root(self) -> None:
        """The root of the returned tree must be named after the file."""
        pdf_path = self.tmp_path / "my_doc.pdf"
        _make_native_pdf(pdf_path)

        parser = NativeDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        tree = parser.parse_to_tree()
        self.assertEqual(tree.root.node_type, "DocumentRoot")
        self.assertEqual(tree.root.text, "my_doc.pdf")

    # ------------------------------------------------------------------
    def test_tree_has_children(self) -> None:
        """Parsed tree must contain at least one content node."""
        pdf_path = self.tmp_path / "content.pdf"
        _make_native_pdf(pdf_path)

        parser = NativeDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        tree = parser.parse_to_tree()
        self.assertGreater(len(tree.root.children), 0, "Tree has no children – no content extracted")

    # ------------------------------------------------------------------
    def test_to_json_is_valid(self) -> None:
        """to_json() must produce parseable JSON with expected keys."""
        pdf_path = self.tmp_path / "json_test.pdf"
        _make_native_pdf(pdf_path)

        parser = NativeDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        tree = parser.parse_to_tree()
        json_str = tree.to_json()

        # Must be valid JSON
        data = json.loads(json_str)
        self.assertIn("text", data)
        self.assertIn("node_type", data)
        self.assertIn("level", data)
        self.assertIn("children", data)
        self.assertEqual(data["node_type"], "DocumentRoot")

    # ------------------------------------------------------------------
    def test_parse_and_save_json_creates_file(self) -> None:
        """parse_and_save_json() must write a readable JSON file."""
        pdf_path = self.tmp_path / "save_test.pdf"
        _make_native_pdf(pdf_path)

        out_json = self.tmp_path / "output" / "tree.json"
        parser = NativeDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        returned_path = parser.parse_and_save_json(json_output_path=str(out_json))

        # File exists
        self.assertTrue(Path(returned_path).exists())

        # Content is valid JSON
        content = Path(returned_path).read_text(encoding="utf-8")
        data = json.loads(content)
        self.assertEqual(data["node_type"], "DocumentRoot")

    # ------------------------------------------------------------------
    def test_default_output_dir_created_next_to_file(self) -> None:
        """Without explicit output_dir the parser places assets beside the PDF."""
        pdf_path = self.tmp_path / "auto_dir.pdf"
        _make_native_pdf(pdf_path)

        parser = NativeDocumentParser(str(pdf_path))
        # Expected default: <pdf_dir>/extracted_assets/<pdf_stem>/
        expected = pdf_path.parent / "extracted_assets" / pdf_path.stem
        self.assertEqual(parser.output_dir, expected)

    # ------------------------------------------------------------------
    def test_node_metadata_present_in_json(self) -> None:
        """Nodes that carry metadata (e.g. image_path) must appear in JSON."""
        pdf_path = self.tmp_path / "meta_test.pdf"
        _make_native_pdf(pdf_path)

        assets_dir = self.tmp_path / "assets"
        parser = NativeDocumentParser(str(pdf_path), output_dir=str(assets_dir))
        tree = parser.parse_to_tree()
        json_str = tree.to_json()
        data = json.loads(json_str)

        def _collect_nodes(node_dict: dict) -> list:
            nodes = [node_dict]
            for child in node_dict.get("children", []):
                nodes.extend(_collect_nodes(child))
            return nodes

        all_nodes = _collect_nodes(data)
        # Every node with a "metadata" key must have a non-null value
        for n in all_nodes:
            if "metadata" in n:
                self.assertIsInstance(n["metadata"], dict)


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline(unittest.TestCase):
    """
    End-to-end: validator → classifier → NativeDocumentParser → JSON file.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        init_database()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    def _run_pipeline(self, pdf_path: Path) -> tuple:
        """
        Execute the full pipeline for a single file.

        Returns (classification_meta, json_path_str | None).
        """
        # Step 1 – Validate
        is_valid, val_meta, val_msg = validator(str(pdf_path))
        if not is_valid:
            return None, None

        # Step 2 – Classify
        class_meta = classifier(str(pdf_path))

        # Step 3 – Route to parser (only NATIVE for now)
        if class_meta.get("file_type") != "NATIVE":
            return class_meta, None

        # Step 4 – Parse and save JSON
        assets_dir = self.tmp_path / "assets" / pdf_path.stem
        parser = NativeDocumentParser(str(pdf_path), output_dir=str(assets_dir))
        json_path = parser.parse_and_save_json(
            json_output_path=str(assets_dir / f"{pdf_path.stem}_tree.json")
        )
        return class_meta, json_path

    # ------------------------------------------------------------------
    def test_pipeline_native_pdf_produces_json(self) -> None:
        """Full pipeline for a native text PDF must produce a valid JSON file."""
        pdf_path = self.tmp_path / "pipeline_native.pdf"
        _make_native_pdf(pdf_path)

        class_meta, json_path = self._run_pipeline(pdf_path)

        self.assertIsNotNone(class_meta, "Pipeline returned no classification")
        self.assertIsNotNone(json_path, "Pipeline did not produce a JSON path")
        self.assertTrue(Path(json_path).exists(), "JSON file does not exist on disk")

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.assertEqual(data["node_type"], "DocumentRoot")
        self.assertGreater(len(data["children"]), 0)

    # ------------------------------------------------------------------
    def test_pipeline_json_schema(self) -> None:
        """Every node in the output JSON must have text, node_type, level, children."""
        pdf_path = self.tmp_path / "schema_check.pdf"
        _make_native_pdf(pdf_path)

        _, json_path = self._run_pipeline(pdf_path)
        self.assertIsNotNone(json_path)

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        def _check_schema(node: dict, path: str = "root") -> None:
            for required_key in ("text", "node_type", "level", "children"):
                self.assertIn(
                    required_key, node,
                    f"Missing '{required_key}' in node at {path}",
                )
            for i, child in enumerate(node["children"]):
                _check_schema(child, f"{path}.children[{i}]")

        _check_schema(data)

    # ------------------------------------------------------------------
    def test_pipeline_db_status_is_classified(self) -> None:
        """After running the pipeline the DB row must have status CLASSIFIED."""
        pdf_path = self.tmp_path / "db_check.pdf"
        _make_native_pdf(pdf_path)

        self._run_pipeline(pdf_path)

        file_hash = calculate_hash_file(str(pdf_path))
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM processed_files WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()

        self.assertIsNotNone(row, "No DB record found for the processed file")
        self.assertEqual(row[0], "CLASSIFIED")

    # ------------------------------------------------------------------
    def test_pipeline_duplicate_file_skipped(self) -> None:
        """Second run of the same file must be rejected by the validator."""
        pdf_path = self.tmp_path / "duplicate_pipeline.pdf"
        _make_native_pdf(pdf_path, text="Unique content for duplicate detection test " * 5)

        # First run
        is_valid_1, _, msg_1 = validator(str(pdf_path))
        self.assertTrue(is_valid_1, f"First run failed: {msg_1}")

        # Second run – must be rejected
        is_valid_2, _, msg_2 = validator(str(pdf_path))
        self.assertFalse(is_valid_2)
        self.assertIn("already processed", msg_2)

    # ------------------------------------------------------------------
    def test_pipeline_scanned_pdf_skips_native_parser(self) -> None:
        """A SCANNED doc must be classified correctly but not reach NativeDocumentParser."""
        pdf_path = self.tmp_path / "scanned_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        # < 50 chars → SCANNED
        page.insert_text((50, 50), "Hi")
        doc.save(str(pdf_path))
        doc.close()

        class_meta, json_path = self._run_pipeline(pdf_path)
        self.assertIsNotNone(class_meta)
        self.assertEqual(class_meta["file_type"], "SCANNED")
        self.assertIsNone(json_path, "NativeParser should NOT be called for SCANNED docs")

    # ------------------------------------------------------------------
    def test_pipeline_corrupt_file_rejected(self) -> None:
        """Corrupt files must fail validation before reaching any parser."""
        corrupt_path = self.tmp_path / "corrupt.pdf"
        corrupt_path.write_bytes(b"Not a valid PDF at all!!!")

        is_valid, _, msg = validator(str(corrupt_path))
        self.assertFalse(is_valid)
        self.assertIn("corrupted or invalid", msg)

    # ------------------------------------------------------------------
    def test_run_native_parser_convenience_function(self) -> None:
        """run_native_parser() convenience wrapper must return an NaryTree."""
        pdf_path = self.tmp_path / "convenience.pdf"
        _make_native_pdf(pdf_path)

        # Must be validated first so the DB record exists
        is_valid, _, _ = validator(str(pdf_path))
        self.assertTrue(is_valid)

        tree = run_native_parser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
        )
        self.assertIsInstance(tree, NaryTree)
        json_str = tree.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["node_type"], "DocumentRoot")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
