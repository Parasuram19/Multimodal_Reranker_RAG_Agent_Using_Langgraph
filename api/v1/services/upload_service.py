import shutil
from pathlib import Path
from fastapi import UploadFile
from ingestion.ingestion1 import run_ingestion_pipeline 

def process_and_ingest_document(file: UploadFile) -> str:
    current_file = Path(__file__).resolve()
    BASE_DIR = current_file.parents[4] 
    DATA_DIR = BASE_DIR / "data"
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_ingestion_pipeline()

    return file.filename