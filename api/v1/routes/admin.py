from fastapi import APIRouter, HTTPException, UploadFile, File
from api.v1.services.upload_service import process_and_ingest_document

router = APIRouter(tags=["Admin"])


@router.post("/admin/upload")
def upload_document(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        saved_filename = process_and_ingest_document(file)
        return {
            "message": "File uploaded and ingested successfully",
            "file": saved_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
