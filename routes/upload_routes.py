from fastapi import APIRouter, UploadFile, File
from services.supabase_client import supabase

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_path = f"videos/{file.filename}"

    supabase.storage.from_("video").upload(file_path, file_bytes)
    url = supabase.storage.from_("video").get_public_url(file_path)

    return {"url": url}
