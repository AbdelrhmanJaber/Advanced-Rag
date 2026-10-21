import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sqlite3
import pytest
from src.database.metadata_database import DB_PATH, init_database


@pytest.fixture(autouse=True)
def clean_db_for_tests():
    init_database()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM processed_files")
        conn.commit()
    yield
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM processed_files")
        conn.commit()
