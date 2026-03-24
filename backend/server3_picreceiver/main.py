from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import shutil
import os
import uuid

app = FastAPI(title="Simple Image Saver", version="1.0.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def secure_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename or "")
    if not ext:
        ext = ".jpg"
    return f"{uuid.uuid4().hex}{ext.lower()}"

@app.post("/api/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Receives an image as multipart/form-data with field name 'file'
    and saves it to ./uploads.
    """

    filename = secure_filename(file.filename)
    dest_path = UPLOAD_DIR / filename

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    return {
        "status": "ok",
        "saved_as": filename,
        "path": str(dest_path),
        "size_bytes": dest_path.stat().st_size,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)