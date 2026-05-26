"""
FastAPI application: serves MJPEG stream and identity management endpoints.
"""

import cv2
import numpy as np
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import time

import src.registry as registry
from src.pipeline import Pipeline
import tensorflow as tf
from src.inference import _preprocess, _run_embedding

with open("config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.start()
    yield
    pipeline.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def mjpeg_generator():
    """Yield MJPEG frames for the video stream."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        jpeg = pipeline.get_frame()
        if jpeg:
            yield boundary + jpeg + b"\r\n"


@app.get("/video")
def video_feed():
    """Live annotated MJPEG stream."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class RegisterRequest(BaseModel):
    name: str


@app.post("/register")
def register_face(req: RegisterRequest):
    """Capture a frame from the live pipeline and register the detected face."""
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    crop = None
    for _ in range(20):
        crop = pipeline.get_crop()
        if crop is not None:
            break
        time.sleep(0.1)

    if crop is None:
        raise HTTPException(status_code=400, detail="No face detected — look at the camera")

    batch = np.expand_dims(_preprocess(crop.astype("float32")), axis=0)
    embedding = _run_embedding(tf.constant(batch)).numpy()[0]
    registry.register(req.name.strip(), embedding)
    return {"registered": req.name.strip()}

@app.get("/identities")
def list_identities():
    """Return all registered names."""
    return registry.list_names()


@app.delete("/identities/{name}")
def delete_identity(name: str):
    """Remove a registered identity."""
    if not registry.delete(name):
        raise HTTPException(status_code=404, detail=f"'{name}' not found")
    return {"deleted": name}

@app.get("/status")
def status():
    return pipeline.status()

@app.get("/")
def index():
    """Serve test UI."""
    with open("test.html") as f:
        return HTMLResponse(f.read())