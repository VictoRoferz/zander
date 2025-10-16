from fastapi import FastAPI

from backend.app.api import picture
from backend.app.core.config import config
from backend.app.core.logging import setup_logging
from backend.app.db.schema import Base, engine

setup_logging()
Base.metadata.create_all(bind=engine)

app = FastAPI(title=config.app_name)


# Register routes
app.include_router(picture.router, prefix="/api")
