from sqlalchemy.orm import Session

from backend.app.db.schema import Picture


class PictureService:
    def __init__(self, session: Session):
        self._db = session

    def list_Pictures(self) -> list[Picture]:
        return self._db.query(Picture).all()

    def get_Picture(self, picture_id: int) -> Picture | None:
        return self._db.query(Picture).filter(Picture.id == picture_id).first()

    def create_Picture(self, filepath: str) -> Picture:
        picture = Picture(filepath=filepath)
        self._db.add(picture)
        self._db.commit()
        self._db.refresh(picture)
        return picture

    def update_Picture(self, picture_id: int, filepath: str) -> Picture | None:
        picture = self.get_Picture(picture_id)
        if not picture:
            return None
        picture.filepath = filepath
        self._db.commit()
        self._db.refresh(picture)
        return picture

    def delete_Picture(self, Picture_id: int) -> bool:
        picture = self.get_Picture(Picture_id)
        if not picture:
            return False
        self._db.delete(picture)
        self._db.commit()
        return True
