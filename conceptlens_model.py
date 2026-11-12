"""Central model, label, preprocessing, and layer-selection utilities."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import Model, load_model


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "model_config.json"


@lru_cache(maxsize=1)
def get_model_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_class_records() -> tuple[dict[str, Any], ...]:
    config = get_model_config()
    payload = json.loads((PROJECT_ROOT / config["labels"]).read_text(encoding="utf-8"))
    records = tuple(payload["classes"])
    expected = config["expected_output_classes"]
    if len(records) != expected or [item["index"] for item in records] != list(range(expected)):
        raise ValueError(f"Class mapping must contain contiguous indices 0..{expected - 1}.")
    return records


CLASS_RECORDS = get_class_records()
CLASS_NAMES = tuple(item["class_id"] for item in CLASS_RECORDS)
DISPLAY_NAMES = tuple(item["display_name"] for item in CLASS_RECORDS)
MODEL_PATH = PROJECT_ROOT / get_model_config()["checkpoint"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def layer_shape(layer: Any) -> tuple[Any, ...]:
    return tuple(layer.output.shape)


def validate_model(model: Any) -> None:
    config = get_model_config()
    expected_input = (
        None,
        config["input"]["height"],
        config["input"]["width"],
        config["input"]["channels"],
    )
    expected_output = (None, config["expected_output_classes"])
    if tuple(model.input_shape) != expected_input:
        raise ValueError(f"Expected model input {expected_input}, received {model.input_shape}.")
    if tuple(model.output_shape) != expected_output:
        raise ValueError(f"Expected model output {expected_output}, received {model.output_shape}.")

    activation_layer = model.get_layer(config["activation_layer"])
    if layer_shape(activation_layer) != (None, config["activation_units"]):
        raise ValueError(
            f"Expected {config['activation_units']}-unit activation layer, "
            f"received {layer_shape(activation_layer)}."
        )
    if activation_layer.get_config().get("activation") != "relu":
        raise ValueError("The configured activation layer is not ReLU.")

    gradcam_shape = layer_shape(model.get_layer(config["gradcam_layer"]))
    if len(gradcam_shape) != 4 or gradcam_shape[1] in (None, 1) or gradcam_shape[2] in (None, 1):
        raise ValueError(f"Configured Grad-CAM layer is not spatial: {gradcam_shape}.")


@st.cache_resource
def load_conceptlens_model() -> Any:
    """Load and validate the configured 50-class checkpoint once per process."""
    config = get_model_config()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ConceptLens checkpoint was not found: {MODEL_PATH}")
    actual_hash = file_sha256(MODEL_PATH)
    if actual_hash != config["checkpoint_sha256"].upper():
        raise ValueError(
            "ConceptLens checkpoint hash mismatch. "
            f"Expected {config['checkpoint_sha256']}, received {actual_hash}."
        )
    model = load_model(MODEL_PATH, compile=False)
    validate_model(model)
    return model


def preprocess_pil_image(source: Image.Image) -> np.ndarray:
    """Mirror the existing app's RGB, 224Ã—224, 0â€“1 inference preprocessing."""
    config = get_model_config()
    resized = source.convert("RGB").resize(
        (config["input"]["width"], config["input"]["height"])
    )
    return np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)


def preprocess_image_path(path: Path | str) -> np.ndarray:
    with Image.open(path) as source:
        return preprocess_pil_image(source)


def activation_model_for(model: Any) -> Model:
    layer = model.get_layer(get_model_config()["activation_layer"])
    return Model(inputs=model.input, outputs=layer.output)


def class_record(index: int) -> dict[str, Any]:
    if not 0 <= index < len(CLASS_RECORDS):
        raise IndexError(f"Class index {index} is outside the configured label mapping.")
    return CLASS_RECORDS[index]


def prediction_records(scores: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (len(CLASS_RECORDS),):
        raise ValueError(f"Expected {len(CLASS_RECORDS)} probabilities, received {values.shape}.")
    indices = np.argsort(values)[::-1][:limit]
    return [
        {**class_record(int(index)), "probability": float(values[index])}
        for index in indices
    ]


