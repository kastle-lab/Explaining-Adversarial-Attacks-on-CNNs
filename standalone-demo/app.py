from __future__ import annotations

import base64
import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_resnet50V2_10classes_retest2023June.h5"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Class order recovered from the successful training job's class_indices output.
CLASS_NAMES = [
    "airport_terminal",
    "alley",
    "amusement_park",
    "art_gallery",
    "art_studio",
    "attic",
    "bar",
    "bathroom",
    "beach",
    "bedroom",
    "bridge",
    "building_facade",
    "castle",
    "childs_room",
    "classroom",
    "closet",
    "coast",
    "cockpit",
    "conference_room",
    "corridor",
    "creek",
    "dining_room",
    "dorm_room",
    "game_room",
    "highway",
    "home_office",
    "hotel_room",
    "house",
    "kitchen",
    "lighthouse",
    "living_room",
    "lobby",
    "mountain",
    "mountain_snowy",
    "nursery",
    "office",
    "park",
    "parlor",
    "pasture",
    "playroom",
    "reception",
    "river",
    "roundabout",
    "shoe_shop",
    "skyscraper",
    "staircase",
    "street",
    "waiting_room",
    "window_seat",
    "youth_hostel",
]


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.after_request
def prevent_local_data_caching(response):
    """Keep uploaded images and prediction responses out of browser caches."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError(
        f"Model produces {model.output_shape[-1]} classes, but {len(CLASS_NAMES)} labels are configured."
    )

# Warm the model once so startup verifies inference and the first user request is faster.
warmup = model(np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32), training=False).numpy()
if warmup.shape != (1, len(CLASS_NAMES)) or not np.isfinite(warmup).all():
    raise RuntimeError("The model loaded but failed its startup inference check.")


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


@app.get("/")
def index():
    return render_template("index.html", class_count=len(CLASS_NAMES))


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH.name, classes=len(CLASS_NAMES))


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image before running a prediction."), 400

    raw = upload.read()
    if not raw:
        return jsonify(error="The selected image is empty."), 400

    try:
        source = Image.open(io.BytesIO(raw))
        source.load()
        image = source.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image."), 400

    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    batch = np.asarray(resized, dtype=np.float32) / 255.0
    scores = model.predict(np.expand_dims(batch, axis=0), verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][:5]

    preview = io.BytesIO()
    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    image.save(preview, format="JPEG", quality=88)

    return jsonify(
        prediction=display_name(CLASS_NAMES[int(top_indices[0])]),
        confidence=float(scores[int(top_indices[0])]),
        top=[
            {
                "label": display_name(CLASS_NAMES[int(index)]),
                "confidence": float(scores[int(index)]),
            }
            for index in top_indices
        ],
        preview=f"data:image/jpeg;base64,{base64.b64encode(preview.getvalue()).decode('ascii')}",
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="The image is larger than the 10 MB upload limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
