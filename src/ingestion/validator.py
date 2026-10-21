from builtins import Exception
import logging
import pymupdf 
from pathlib import Path
import hashlib
import sys
from pathlib import Path

from database.metadata_database import connect_metadata_database, save_metadata_in_db

# Configure logger for validator
logger = logging.getLogger("validator")
if not logger.handlers:
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler("validator_issues.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def is_file_already_processed(file_hash : str) -> bool:
    with connect_metadata_database() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_files where file_hash = ?",(file_hash,))
        return cursor.fetchone() is not None




def validator(file_path):
    file_path_obj = Path(file_path)
    # check if path not exist
    if not file_path_obj.exists():
        return False , {} , "File Path doesn't exist"
    # check if the path is really file
    if not file_path_obj.is_file():
        return False , {} , "File directory is not exist"
    try:
        # check file size
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            return False , {} , "File size is 0 Bytes"
        # hash the file
        hashed_value = calculate_hash_file(file_path)
        if is_file_already_processed(hashed_value):
            return False , {"file hash value" : hashed_value} , "File is already processed before"
        #check file pages
        file = pymupdf.open(file_path_obj)
        pages = len(file)
        if pages == 0:
            return False , {"file hash value" : hashed_value} , "File is empty , number of pages = 0"
        # check if file encrypted
        is_encrypted = file.is_encrypted
        if is_encrypted:
            return False , {"file hash value": hashed_value} , "File is encrypted can't be opened"
        metadata = {
            "fileName" : file_path_obj.name , 
            "pages" : pages,
            "is_encrypted" : is_encrypted,
            "hashed_value" : hashed_value 
        }    
        save_metadata_in_db(hashed_value , file_path_obj.name , metadata)
        return True , metadata , "file is verified"
    except Exception as e:
        msg = f"File is corrupted or invalid ({file_path}): {str(e)}"
        logger.error(msg)
        return False, {} , f"File is corrupted or invalid : {str(e)}"


# implement hashing for files and save it in json file 

def calculate_hash_file(file_path, chunk_size = 8192):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()


if __name__ == "__main__":
    import os
    from typing import Generator

    class LocalFileLoader:
        def __init__(self, raw_dir: str):
            self.raw_direct = Path(raw_dir)
            if not self.raw_direct.exists():
                raise FileNotFoundError(f'Path {self.raw_direct} does not exist')
        def scan_files(self) -> Generator[Path, None, None]:
            for file_path in self.raw_direct.rglob("*.pdf"):
                if file_path.is_file():
                    yield file_path
    loader = LocalFileLoader("data")

    for file_path in loader.scan_files():
        is_valid, data_or_meta, message = validator(file_path)
        print(f"File: {file_path.name} | Valid: {is_valid} | Message: {message}")