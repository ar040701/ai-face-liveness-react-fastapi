from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from modules.liveness import check_liveness, reset_liveness


app = FastAPI(title="AI Face Liveness Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
        "https://ai-face-liveness-react-fastapi.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "AI Face Liveness Detection API is running"
    }


@app.post("/verify-frame")
async def verify_frame(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "status": "ERROR",
            "message": "Invalid image frame"
        }

    result = check_liveness(frame)
    return result


@app.post("/reset")
def reset():
    reset_liveness()
    return {
        "message": "Verification reset"
    }