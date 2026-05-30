"""
FastAPI application: serves MJPEG stream, identity management endpoints,
and Grad-CAM spoof explainability endpoint.
"""

import cv2
import numpy as np
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
import time
import os
from datetime import datetime

import tensorflow as tf

import src.registry as registry
from src.pipeline import Pipeline
from src.inference import _preprocess, _run_embedding
from src.spoof_explainability import SpoofExplainability


with open("config.yaml") as f:
    config = yaml.safe_load(f)


pipeline = Pipeline()

# Lasith's innovative feature:
# Explainable Anti-Spoofing Decision Visualisation using Grad-CAM
spoof_explainer = SpoofExplainability(config["models"]["anti_spoofing"])


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
    """Capture multiple frames from the live pipeline and register the averaged face template."""
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    samples = config["registration"]["enrolment_samples"]
    interval = config["registration"]["enrolment_interval"]

    collected = []

    for _ in range(samples):
        crop = None

        for _ in range(20):
            crop = pipeline.get_crop()

            if crop is not None:
                break

            time.sleep(0.05)

        if crop is not None:
            batch = np.expand_dims(_preprocess(crop.astype("float32")), axis=0)
            embedding = _run_embedding(tf.constant(batch)).numpy()[0]
            collected.append(embedding)

        time.sleep(interval)

    if not collected:
        raise HTTPException(
            status_code=400,
            detail="No face detected — look at the camera"
        )

    registry.register_multi(req.name.strip(), collected)

    return {
        "registered": req.name.strip(),
        "samples": len(collected)
    }


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
    """Return current pipeline status for frontend panel."""
    return pipeline.status()


@app.get("/spoof-explainability")
def spoof_explainability():
    """
    Generate Grad-CAM explanation for the latest detected face crop.

    Innovative feature:
    Explainable Anti-Spoofing Decision Visualisation using Grad-CAM heatmaps.
    """

    crop = None

    for _ in range(20):
        crop = pipeline.get_crop()

        if crop is not None:
            break

        time.sleep(0.1)

    if crop is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected — look at the camera before generating explanation"
        )

    try:
        result = spoof_explainer.explain_face(crop)

        label = result["label"]
        score = result["score"]
        confidence = result["confidence"]
        overlay = result["overlay"]

        output_dir = "output/spoof_explainability"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spoof_gradcam_{label}_{timestamp}.jpg"
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, overlay)

        return {
            "prediction": label,
            "spoof_score": score,
            "confidence": confidence,
            "heatmap_url": f"/spoof-explainability/image/{filename}",
            "message": "Grad-CAM spoof explanation generated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate spoof explanation: {str(e)}"
        )


@app.get("/spoof-explainability/image/{filename}")
def get_spoof_explainability_image(filename: str):
    """Serve saved Grad-CAM heatmap image."""

    image_path = os.path.join("output/spoof_explainability", filename)

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Heatmap image not found"
        )

    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/")
def index():
    """Serve test UI."""
    with open("test.html") as f:
        return HTMLResponse(f.read())