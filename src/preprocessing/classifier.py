import pymupdf
from src.ingestion.validator import calculate_hash_file
from database.metadata_database import update_metadata_in_db

def check_file_type(doc) -> str:
    total_chars = 0
    for page in doc:
        text = page.get_text()
        total_chars+= len(text.strip())

    if total_chars < 50 :
        return "SCANNED"
    else:
        return "NATIVE"


def check_has_tables(doc) -> bool:
    for page in doc:
        tables_exist = page.find_tables()
        if tables_exist.tables : 
            return True
    return False


def check_has_images(doc) -> bool :
    for page in doc :
        images_exist = page.get_images(full = True)
        if images_exist:
            return True
    return False


def classifier(file_path) :
    file = pymupdf.open(file_path)
    hashed_file = calculate_hash_file(file_path)
    file_type = check_file_type(file)
    has_table = False
    has_images = False
    # init values untill ocr
    if file_type == "SCANNED":
        has_table = True
        has_images = True
    else :
        has_table = check_has_tables(file)
        has_images = check_has_images(file)
    classification_metadata = {
                "file_type" : file_type,
                "has_table" : has_table,
                "has_images" : has_images
            }
    update_metadata_in_db(hashed_file , classification_metadata , "CLASSIFIED")
    return classification_metadata
