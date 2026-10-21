import sqlite3
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"

def init_database():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_hash TEXT PRIMARY KEY,
                file_name TEXT,
                metadata_json TEXT, 
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("[DB] Database and Schema initialized successfully!")



def connect_metadata_database():
    return sqlite3.connect(DB_PATH)

def save_metadata_in_db(hashed_value : str , file_name: str , metadata : dict):
    metadata_json = json.dumps(metadata , ensure_ascii=False)
    with connect_metadata_database() as conn :
        conn.execute("""
            INSERT OR REPLACE INTO processed_files (file_hash, file_name, metadata_json, status)
            VALUES (?, ?, ?, ?)
        """, (hashed_value, file_name, metadata_json, "VALIDATED"))
        conn.commit()


def update_metadata_in_db(hashed_value : str , new_metadata : dict , new_status : str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM processed_files WHERE file_hash = ?", (hashed_value,))
        row = cursor.fetchone()
        
        if row:
            current_metadata = json.loads(row[0])
            
            current_metadata.update(new_metadata)
            
            updated_json_str = json.dumps(current_metadata, ensure_ascii=False)
            
            cursor.execute("""
                UPDATE processed_files 
                SET metadata_json = ?, status = ?
                WHERE file_hash = ?
            """, (updated_json_str, new_status, hashed_value))
            
            conn.commit()    


if __name__ == "__main__":
    init_database()