import sys
from pathlib import Path

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import json
import unittest
import tempfile
import sqlite3
import pymupdf as fitz

from src.ingestion.validator import validator, calculate_hash_file, is_file_already_processed
from src.database.metadata_database import init_database, save_metadata_in_db, DB_PATH

class TestValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.log_file = Path("validator_issues.log")
        init_database()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_pdf(self):
        """Test passing case with a valid PDF file."""
        pdf_path = self.temp_path / "valid.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        is_valid, meta, msg = validator(str(pdf_path))
        self.assertTrue(is_valid)
        self.assertEqual(msg, "file is verified")
        self.assertIn("hashed_value", meta)

    def test_zero_byte_file(self):
        """Test non-passing case with an empty (0 byte) file."""
        empty_path = self.temp_path / "empty.pdf"
        empty_path.touch()

        is_valid, meta, msg = validator(str(empty_path))
        self.assertFalse(is_valid)
        self.assertIn("File size is 0 Bytes", msg)

    def test_corrupted_pdf_file(self):
        """Test non-passing case with a corrupted file."""
        corrupted_path = self.temp_path / "corrupted.pdf"
        with open(corrupted_path, "wb") as f:
            f.write(b"%PDF-1.4\n%Malformed binary trash content header...")

        is_valid, meta, msg = validator(str(corrupted_path))
        self.assertFalse(is_valid)
        self.assertIn("File is corrupted or invalid", msg)

    def test_non_existent_file(self):
        """Test non-passing case with non-existent file."""
        missing_path = self.temp_path / "does_not_exist.pdf"
        is_valid, meta, msg = validator(str(missing_path))
        self.assertFalse(is_valid)
        self.assertIn("File Path doesn't exist", msg)

    def test_encrypted_pdf_file(self):
        """Test non-passing case with an encrypted PDF."""
        encrypted_path = self.temp_path / "encrypted.pdf"
        doc = fitz.open()
        doc.new_page()
        encrypt_flag = getattr(fitz, "PDF_ENCRYPT_AES_256", 4)
        doc.save(str(encrypted_path), encryption=encrypt_flag, user_pw="secret")
        doc.close()

        is_valid, meta, msg = validator(str(encrypted_path))
        self.assertFalse(is_valid)
        self.assertIn("encrypted", msg.lower())

    def test_hash_and_processed_status(self):
        """Test calculating file hash and saving hash metadata to DB."""
        test_file = self.temp_path / "sample.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(test_file))
        doc.close()

        file_hash = calculate_hash_file(str(test_file))
        self.assertTrue(isinstance(file_hash, str))

        # Check before saving
        self.assertFalse(is_file_already_processed(file_hash))

        # Save new hash into DB
        save_metadata_in_db(file_hash, test_file.name, {"fileName": test_file.name})

        # Check after saving - should return True
        self.assertTrue(is_file_already_processed(file_hash))

if __name__ == "__main__":
    unittest.main()
