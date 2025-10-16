from fastapi import APIRouter, Depends, HTTPException

from backend.app.db.schema import SessionLocal
from backend.app.models.picture import PictureCreate, PictureRead
from backend.app.services.picture_service import PictureService

router = APIRouter()


def get_picture_service() -> PictureService:
    return PictureService(session=SessionLocal())


@router.get("/pictures", response_model=list[PictureRead])
def get_Pictures(service: PictureService = Depends(get_picture_service)):
    return service.list_Pictures()


@router.post("/pictures", response_model=PictureRead)
def create_Picture(Picture: PictureCreate, service: PictureService = Depends(get_picture_service)):
    return service.create_Picture(Picture.filepath)


@router.get("/pictures/{picture_id}", response_model=PictureRead)
def get_Picture(picture_id: int, service: PictureService = Depends(get_picture_service)):
    picture = service.get_Picture(picture_id)
    if not picture:
        raise HTTPException(status_code=404, detail="Picture not found")
    return picture


@router.put("/pictures/{picture_id}", response_model=PictureRead)
def update_Picture(
    picture_id: int, Picture: PictureCreate, service: PictureService = Depends(get_picture_service)
):
    updated = service.update_Picture(picture_id, Picture.filepath)
    if not updated:
        raise HTTPException(status_code=404, detail="Picture not found")
    return updated


@router.delete("/pictures/{picture_id}")
def delete_Picture(picture_id: int, service: PictureService = Depends(get_picture_service)):
    success = service.delete_Picture(picture_id)
    if not success:
        raise HTTPException(status_code=404, detail="Picture not found")
    return {"success": True}
