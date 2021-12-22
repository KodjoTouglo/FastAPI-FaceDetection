from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api import detection



app = FastAPI(title="Face Detection")

app.include_router(detection.router)


app.mount("/media", StaticFiles(directory="media"), name="media")