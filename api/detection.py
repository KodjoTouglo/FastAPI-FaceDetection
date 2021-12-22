from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from models.func.detect_eyes import detect_eyes
from models.func.detect_face import detect_face
from models.func.detect_smile import detect_smile
from models.func.detect_plate import detect_plate
from models.func.blur_plate import blur_plate
from models.func.blur_face import blur_face
from starlette.responses import StreamingResponse
from PIL import Image

import io
import numpy as np
import shutil
import cv2




router = APIRouter(prefix="/detection", tags=["Detection on images"])


@router.post("/upload")
def upload_file(file: UploadFile=File(...)):
    path = f"media/{file.filename}"
    with open(path, "w+b") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "filename": path,
        "type": file.content_type
    }


@router.get("/download/{name}", response_class=FileResponse)
def download_file(name: str):
    path = f"media/{name}"
    return path



@router.post("/detect_eyes", response_class=FileResponse)
async def eyes_detection(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = detect_eyes(image)
    
    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



@router.post("/detect_face", response_class=FileResponse)
async def face_detection(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = detect_face(image)

    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



@router.post("/detection_smile", response_class=FileResponse)
async def smile_detection(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = detect_smile(image)

    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



@router.post("/plate_detection", response_class=FileResponse)
async def plate_detection(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = detect_plate(image)

    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



@router.post("/blur_plate", response_class=FileResponse)
async def blurred_plate(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = blur_plate(image)

    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



@router.post("/blur_face", response_class=FileResponse)
async def blurred_face(file: UploadFile=File(...)):
    image = np.array(Image.open(file.file))
    rgb = blur_face(image)

    _, im_png = cv2.imencode(".png", rgb)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")