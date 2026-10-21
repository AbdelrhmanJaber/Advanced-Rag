import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
import tempfile
import sqlite3
import json
import pymupdf

from src.ingestion.validator import validator, calculate_hash_file
from src.preprocessing.classifier import classifier, check_file_type, check_has_tables, check_has_images
from src.database.metadata_database import init_database, DB_PATH

class TestClassifierIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        # Ensure database table exists before testing
        init_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_classifier_with_native_text_pdf(self):
        """Test validation + classification pipeline on a native text PDF."""
        pdf_path = self.temp_path / "native_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        # Add text with more than 50 characters
        page.insert_text((50, 50), "This is a native text document containing more than 50 characters to ensure it gets classified as NATIVE.")
        doc.save(str(pdf_path))
        doc.close()

        # Step 1: Validate file
        is_valid, val_meta, val_msg = validator(str(pdf_path))
        self.assertTrue(is_valid, f"Validation failed: {val_msg}")
        self.assertIn("hashed_value", val_meta)

        # Step 2: Classify file
        class_meta = classifier(str(pdf_path))
        self.assertEqual(class_meta["file_type"], "NATIVE")
        self.assertFalse(class_meta["has_table"])
        self.assertFalse(class_meta["has_images"])

        # Step 3: Verify database state
        file_hash = calculate_hash_file(str(pdf_path))
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata_json, status FROM processed_files WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            db_meta = json.loads(row[0])
            self.assertEqual(row[1], "CLASSIFIED")
            self.assertEqual(db_meta["file_type"], "NATIVE")

    def test_classifier_with_scanned_pdf(self):
        """Test validation + classification on a scanned (low text count) PDF."""
        pdf_path = self.temp_path / "scanned_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        # Insert minimal text (< 50 chars) to simulate scanned doc before OCR
        page.insert_text((50, 50), "Short text")
        doc.save(str(pdf_path))
        doc.close()

        is_valid, val_meta, val_msg = validator(str(pdf_path))
        self.assertTrue(is_valid)

        class_meta = classifier(str(pdf_path))
        self.assertEqual(class_meta["file_type"], "SCANNED")
        self.assertTrue(class_meta["has_table"])
        self.assertTrue(class_meta["has_images"])

    def test_validation_failure_prevents_classifier(self):
        """Test that invalid/corrupt files fail validation prior to classification."""
        corrupt_path = self.temp_path / "corrupt.pdf"
        corrupt_path.write_bytes(b"Invalid binary garbage")

        is_valid, val_meta, val_msg = validator(str(corrupt_path))
        self.assertFalse(is_valid)
        self.assertIn("corrupted or invalid", val_msg)

    def test_already_processed_file_validation(self):
        """Test duplicate validation detection."""
        pdf_path = self.temp_path / "duplicate_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Unique content for duplicate file hashing test baseline.")
        doc.save(str(pdf_path))
        doc.close()

        # First run: valid
        is_valid_1, _, _ = validator(str(pdf_path))
        self.assertTrue(is_valid_1)

        # Second run: rejected because hash is already processed
        is_valid_2, val_meta_2, msg_2 = validator(str(pdf_path))
        self.assertFalse(is_valid_2)
        self.assertIn("already processed", msg_2)

if __name__ == "__main__":
    unittest.main()
