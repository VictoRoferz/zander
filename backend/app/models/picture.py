from pydantic import BaseModel


class PictureCreate(BaseModel):
    filepath: str


class PictureRead(BaseModel):
    id: int
    filepath: str
