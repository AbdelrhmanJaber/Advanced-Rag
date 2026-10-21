"""
test_scanned_parser.py
======================
Unit and Integration tests for ScannedDocumentParser:

    LocalFileLoader  →  validator  →  classifier  →  ScannedDocumentParser  →  JSON

Tests use synthetic scanned PDFs created in-memory (by rendering text onto an
image pixmap and embedding only the image into a clean PDF) so no external
fixture files are required.
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pymupdf  # type: ignore

from src.database.metadata_database import DB_PATH, init_database
from src.ingestion.validator import calculate_hash_file, validator
from src.preprocessing.classifier import classifier
from src.preprocessing.scanned_parser import ScannedDocumentParser, run_scanned_parser
from src.preprocessing.tree_buillder import NaryTree


# ---------------------------------------------------------------------------
# Synthetic PDF Generators
# ---------------------------------------------------------------------------


def _make_scanned_pdf(
    path: Path,
    title: str = "SAMPLE SCANNED REPORT",
    body: str = "This is a scanned document rendered as an image layer.",
) -> None:
    """
    Create a genuine scanned PDF (image-only page without native selectable text).

    Strategy:
    1. Render text onto a temporary PyMuPDF page.
    2. Convert page to a raster pixmap image.
    3. Insert the pixmap image onto a fresh page in the output PDF.
    This guarantees page.get_text() returns 0 characters, so classifier sees SCANNED.
    """
    # Step 1: Render temporary page
    temp_doc = pymupdf.open()
    temp_page = temp_doc.new_page(width=595, height=842)  # A4
    temp_page.insert_text((50, 72), title, fontsize=18)
    temp_page.insert_text((50, 110), body, fontsize=11)
    temp_page.insert_text(
        (50, 130),
        "Additional narrative details describing the scanned workflow process.",
        fontsize=11,
    )
    temp_page.insert_text((50, 170), "- Item 1: First item", fontsize=11)
    temp_page.insert_text((50, 190), "- Item 2: Second item", fontsize=11)

    pix = temp_page.get_pixmap(dpi=150)

    # Step 2: Save as pure image PDF
    scanned_doc = pymupdf.open()
    scanned_page = scanned_doc.new_page(width=595, height=842)
    scanned_page.insert_image(scanned_page.rect, pixmap=pix)

    scanned_doc.save(str(path))
    scanned_doc.close()
    temp_doc.close()


def _make_blank_scanned_pdf(path: Path) -> None:
    """Create a scanned PDF page with no text (white image)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = page.get_pixmap(dpi=72)

    scanned_doc = pymupdf.open()
    scanned_page = scanned_doc.new_page(width=595, height=842)
    scanned_page.insert_image(scanned_page.rect, pixmap=pix)

    scanned_doc.save(str(path))
    scanned_doc.close()
    doc.close()


# ---------------------------------------------------------------------------
# Unit Tests for ScannedDocumentParser
# ---------------------------------------------------------------------------


class TestScannedDocumentParserUnit(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_file_not_found_raises(self) -> None:
        """Constructor must raise FileNotFoundError for nonexistent paths."""
        with self.assertRaises(FileNotFoundError):
            ScannedDocumentParser("/non/existent/scanned.pdf")

    def test_parse_returns_nary_tree(self) -> None:
        """parse_to_tree() must return an NaryTree object."""
        pdf_path = self.tmp_path / "scanned_unit.pdf"
        _make_scanned_pdf(pdf_path)

        parser = ScannedDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        tree = parser.parse_to_tree()
        self.assertIsInstance(tree, NaryTree)

    def test_root_node_is_document_root(self) -> None:
        """The root node must have node_type DocumentRoot and text matching file stem/name."""
        pdf_path = self.tmp_path / "report.pdf"
        _make_scanned_pdf(pdf_path)

        parser = ScannedDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        tree = parser.parse_to_tree()
        self.assertEqual(tree.root.node_type, "DocumentRoot")
        self.assertEqual(tree.root.text, "report.pdf")

    def test_to_json_is_valid(self) -> None:
        """to_json() must return valid JSON string matching expected schema."""
        pdf_path = self.tmp_path / "json_check.pdf"
        _make_scanned_pdf(pdf_path)

        parser = ScannedDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        tree = parser.parse_to_tree()
        json_str = tree.to_json()

        data = json.loads(json_str)
        self.assertIn("text", data)
        self.assertIn("node_type", data)
        self.assertIn("level", data)
        self.assertIn("children", data)
        self.assertEqual(data["node_type"], "DocumentRoot")

    def test_parse_and_save_json_creates_file(self) -> None:
        """parse_and_save_json() must write a JSON file to disk."""
        pdf_path = self.tmp_path / "save_scanned.pdf"
        _make_scanned_pdf(pdf_path)

        out_json = self.tmp_path / "outputs" / "scanned_tree.json"
        parser = ScannedDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        saved_path = parser.parse_and_save_json(json_output_path=str(out_json))

        self.assertTrue(Path(saved_path).exists())
        data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
        self.assertEqual(data["node_type"], "DocumentRoot")

    def test_blank_page_handled_gracefully(self) -> None:
        """Blank scanned pages should not cause errors and produce a valid tree."""
        pdf_path = self.tmp_path / "blank.pdf"
        _make_blank_scanned_pdf(pdf_path)

        parser = ScannedDocumentParser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        tree = parser.parse_to_tree()
        self.assertIsInstance(tree, NaryTree)
        self.assertEqual(tree.root.node_type, "DocumentRoot")

    def test_run_scanned_parser_convenience_function(self) -> None:
        """run_scanned_parser() wrapper function must return NaryTree."""
        pdf_path = self.tmp_path / "convenience_scanned.pdf"
        _make_scanned_pdf(pdf_path)

        tree = run_scanned_parser(
            str(pdf_path),
            output_dir=str(self.tmp_path / "assets"),
            run_table_detection=False,
        )
        self.assertIsInstance(tree, NaryTree)


# ---------------------------------------------------------------------------
# Integration Tests with Classifier & Ingestion Pipeline
# ---------------------------------------------------------------------------


class TestScannedPipelineIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        init_database()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_pipeline(self, pdf_path: Path) -> tuple:
        """
        Execute full pipeline:
        validator → classifier → ScannedDocumentParser → JSON file.
        """
        # Step 1: Validate
        is_valid, val_meta, val_msg = validator(str(pdf_path))
        if not is_valid:
            return None, None

        # Step 2: Classify
        class_meta = classifier(str(pdf_path))

        # Step 3: Route to parser based on type
        if class_meta.get("file_type") == "SCANNED":
            assets_dir = self.tmp_path / "assets" / pdf_path.stem
            parser = ScannedDocumentParser(
                str(pdf_path),
                output_dir=str(assets_dir),
                run_table_detection=False,
            )
            json_path = parser.parse_and_save_json(
                json_output_path=str(assets_dir / f"{pdf_path.stem}_tree.json")
            )
            return class_meta, json_path

        return class_meta, None

    def test_classifier_identifies_scanned_pdf(self) -> None:
        """Synthetic image PDF must be correctly classified as SCANNED."""
        pdf_path = self.tmp_path / "scanned_doc.pdf"
        _make_scanned_pdf(pdf_path)

        class_meta, json_path = self._run_pipeline(pdf_path)

        self.assertIsNotNone(class_meta)
        self.assertEqual(class_meta["file_type"], "SCANNED")
        self.assertIsNotNone(json_path)
        self.assertTrue(Path(json_path).exists())

    def test_pipeline_output_json_structure(self) -> None:
        """Output JSON generated from scanned PDF pipeline must match schema."""
        pdf_path = self.tmp_path / "scanned_tree_test.pdf"
        _make_scanned_pdf(pdf_path)

        _, json_path = self._run_pipeline(pdf_path)
        self.assertIsNotNone(json_path)

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        def _check_schema(node: dict, path: str = "root") -> None:
            for key in ("text", "node_type", "level", "children"):
                self.assertIn(key, node, f"Missing key '{key}' at {path}")
            for i, child in enumerate(node.get("children", [])):
                _check_schema(child, f"{path}.children[{i}]")

        _check_schema(data)

    def test_pipeline_db_status(self) -> None:
        """Database must record CLASSIFIED status after pipeline run."""
        pdf_path = self.tmp_path / "scanned_db_test.pdf"
        _make_scanned_pdf(pdf_path)

        self._run_pipeline(pdf_path)

        file_hash = calculate_hash_file(str(pdf_path))
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM processed_files WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "CLASSIFIED")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
