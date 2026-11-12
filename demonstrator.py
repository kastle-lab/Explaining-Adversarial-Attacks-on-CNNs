#!/usr/bin/env python
"""Inference, 64-neuron profiling, and Grad-CAM helpers for ConceptLens."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model

from conceptlens_model import (
    PROJECT_ROOT,
    activation_model_for,
    class_record,
    file_sha256,
    get_model_config,
    layer_shape,
    load_conceptlens_model,
    prediction_records,
    preprocess_image_path,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ACTIVATION_COLUMNS = [
    "Neuron ID",
    "Activation Value",
    "Reference Maximum",
    "Reference Percentage",
    "Reference Available",
    "Scene Association",
    "Primary Scene Association ID",
    "Association AUC",
    "Secondary Scene Association ID",
    "Secondary Association AUC",
    "AUC Margin",
    "Evidence Tier",
    "Association Available",
    "Image Name",
]


class GradCAMError(RuntimeError):
    """Raised when a spatial Grad-CAM explanation cannot be generated."""


class ActivationAnalysisError(RuntimeError):
    """Raised when the verified 64-neuron representation is unavailable."""


def figure_to_png_bytes(figure: Any) -> bytes:
    image_bytes = BytesIO()
    figure.savefig(image_bytes, format="png", dpi=150, facecolor="white")
    image_bytes.seek(0)
    return image_bytes.getvalue()


@lru_cache(maxsize=1)
def load_activation_reference() -> pd.DataFrame:
    path = PROJECT_ROOT / get_model_config()["activation_reference"]
    frame = pd.read_csv(path)
    expected_ids = list(range(get_model_config()["activation_units"]))
    if frame["neuron_id"].tolist() != expected_ids:
        raise ActivationAnalysisError("Activation reference must contain neuron IDs 0 through 63.")
    return frame


@lru_cache(maxsize=1)
def load_activation_scene_associations() -> pd.DataFrame:
    config = get_model_config()
    path = PROJECT_ROOT / config["activation_scene_associations"]
    if file_sha256(path) != config["activation_scene_associations_sha256"].upper():
        raise ActivationAnalysisError(f"Activation scene-association hash mismatch: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    expected_ids = list(range(config["activation_units"]))
    if frame["neuron_id"].tolist() != expected_ids:
        raise ActivationAnalysisError("Scene associations must contain neuron IDs 0 through 63.")
    allowed_tiers = {"high", "moderate", "ambiguous", "inactive"}
    if not set(frame["evidence_tier"]).issubset(allowed_tiers):
        raise ActivationAnalysisError("Scene associations contain an unknown evidence tier.")
    return frame


def activation_rows_for(
    activations: np.ndarray, image_name: str = "", *, include_all: bool = True, top_n: int = 10
) -> pd.DataFrame:
    """Describe new-model neurons without assigning legacy semantics.

    Reference percentages compare each value with the maximum observed in the
    supplied 1,921-image Stage 1 activation CSV. High/moderate-evidence scene
    associations come from descriptive one-vs-rest AUC analysis of the same
    reference. Neither value is a probability, error rate, object concept, or
    causal claim about neuron identity.
    """
    values = np.asarray(activations, dtype=np.float64).reshape(-1)
    expected = get_model_config()["activation_units"]
    if values.shape != (expected,):
        raise ActivationAnalysisError(f"Expected {expected} activations, received {values.shape}.")
    if not np.isfinite(values).all():
        raise ActivationAnalysisError("The activation vector contains non-finite values.")

    reference = load_activation_reference()
    associations = load_activation_scene_associations()
    maxima = reference["reference_max"].to_numpy(dtype=np.float64)
    available = maxima > 0
    relative = np.full(expected, np.nan, dtype=np.float64)
    relative[available] = values[available] / maxima[available] * 100.0
    frame = pd.DataFrame(
        {
            "Neuron ID": np.arange(expected, dtype=int),
            "Activation Value": values,
            "Reference Maximum": maxima,
            "Reference Percentage": relative,
            "Reference Available": available,
            "Scene Association": associations["primary_scene_display"],
            "Primary Scene Association ID": associations["primary_scene"],
            "Association AUC": pd.to_numeric(associations["primary_auc"], errors="coerce"),
            "Secondary Scene Association ID": associations["secondary_scene"],
            "Secondary Association AUC": pd.to_numeric(
                associations["secondary_auc"], errors="coerce"
            ),
            "AUC Margin": pd.to_numeric(associations["auc_margin"], errors="coerce"),
            "Evidence Tier": associations["evidence_tier"],
            "Association Available": associations["evidence_tier"].isin({"high", "moderate"}),
            "Image Name": image_name,
        }
    )
    if include_all:
        return frame
    return (
        frame[frame["Reference Available"] & (frame["Activation Value"] > 0)]
        .sort_values(["Reference Percentage", "Activation Value"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_activation_bars(axis: Any, activation_results: pd.DataFrame, top_n: int = 10) -> None:
    filtered = (
        activation_results[
            activation_results["Reference Available"]
            & (activation_results["Activation Value"] > 0)
        ]
        .sort_values(["Reference Percentage", "Activation Value"], ascending=False)
        .head(top_n)
        .sort_values("Reference Percentage", ascending=True)
    )
    if filtered.empty:
        axis.text(0.5, 0.5, "No positive calibrated activations", ha="center", va="center")
    else:
        labels = []
        for _, row in filtered.iterrows():
            label = f"Neuron {int(row['Neuron ID'])}"
            if row["Association Available"] and row["Scene Association"]:
                label += f" - {row['Scene Association']}"
            labels.append(label)
        axis.barh(
            labels,
            filtered["Reference Percentage"],
        )
    axis.set_xlabel("Activation as % of Stage 1 reference maximum")


def render_activation_visualization(
    input_image: Image.Image, image_name: str, activation_results: pd.DataFrame
) -> Any:
    figure = plt.figure(figsize=(15, 8), facecolor="white")
    figure.text(
        0.035, 0.965, "64-Neuron Scene-Association Profile", ha="left", va="top",
        fontsize=19, fontweight="bold", color="#262730"
    )
    figure.text(
        0.035,
        0.925,
        "Supported labels are statistical scene associations; legacy object concepts/error rates remain unavailable.",
        ha="left",
        va="top",
        fontsize=10,
        color="#6c757d",
    )
    grid = figure.add_gridspec(
        1, 2, left=0.035, right=0.97, bottom=0.10, top=0.86, width_ratios=[1, 2], wspace=0.22
    )
    image_axis = figure.add_subplot(grid[0, 0])
    image_axis.imshow(input_image)
    image_axis.set_xticks([])
    image_axis.set_yticks([])
    image_axis.set_xlabel(image_name, fontsize=10, color="#6c757d", labelpad=10)
    for spine in image_axis.spines.values():
        spine.set_visible(False)
    chart_axis = figure.add_subplot(grid[0, 1])
    plot_activation_bars(chart_axis, activation_results)
    return figure


def render_gradcam_visualization(
    input_image: Image.Image, image_name: str, gradcam_result: dict[str, Any]
) -> Any:
    figure = plt.figure(figsize=(15, 9), facecolor="white")
    figure.text(
        0.035, 0.965, "Grad-CAM Saliency Map", ha="left", va="top", fontsize=19,
        fontweight="bold", color="#262730"
    )
    figure.text(
        0.035,
        0.922,
        "Grad-CAM highlights spatial regions that influenced the 50-class scene prediction.",
        ha="left",
        va="top",
        fontsize=10,
        color="#6c757d",
    )
    figure.text(0.035, 0.858, image_name, ha="left", va="top", fontsize=14, fontweight="bold")
    figure.text(
        0.035,
        0.824,
        f"Predicted class: {gradcam_result['predicted_display_name']}  |  "
        f"Probability: {gradcam_result['prediction_score']:.2%}",
        ha="left",
        va="top",
        fontsize=12,
    )
    grid = figure.add_gridspec(1, 2, left=0.035, right=0.97, bottom=0.08, top=0.76, wspace=0.08)
    original_axis = figure.add_subplot(grid[0, 0])
    original_axis.imshow(input_image)
    original_axis.set_xticks([])
    original_axis.set_yticks([])
    original_axis.set_xlabel("Original Image", fontsize=10, color="#6c757d", labelpad=10)
    overlay_axis = figure.add_subplot(grid[0, 1])
    overlay_axis.imshow(gradcam_result["overlay"])
    overlay_axis.set_xticks([])
    overlay_axis.set_yticks([])
    overlay_axis.set_xlabel("Grad-CAM Overlay", fontsize=10, color="#6c757d", labelpad=10)
    for axis in (original_axis, overlay_axis):
        for spine in axis.spines.values():
            spine.set_visible(False)
    return figure


def find_gradcam_layer(model: Any) -> Any:
    configured_name = get_model_config()["gradcam_layer"]
    try:
        configured = model.get_layer(configured_name)
    except (AttributeError, ValueError):
        configured = None
    if configured is not None:
        shape = layer_shape(configured)
        if len(shape) == 4 and shape[1] not in (None, 1) and shape[2] not in (None, 1):
            return configured

    for layer in reversed(model.layers):
        shape = layer_shape(layer)
        if len(shape) == 4 and shape[1] not in (None, 1) and shape[2] not in (None, 1):
            return layer
    raise GradCAMError("No connected spatial feature layer is available for Grad-CAM.")


def generate_gradcam(image_path: Path | str, model: Any = None) -> dict[str, Any]:
    path = Path(image_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise GradCAMError("Grad-CAM supports JPEG and PNG images.")
    if not path.is_file():
        raise GradCAMError(f"Image could not be loaded: {path}")
    try:
        original_image = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise GradCAMError(f"Image could not be loaded: {path}") from exc

    if model is None:
        model = load_conceptlens_model()
    conceptlens_input = preprocess_image_path(path)
    gradcam_layer = find_gradcam_layer(model)
    try:
        grad_model = Model(model.inputs, [gradcam_layer.output, model.output])
    except (ValueError, TypeError) as exc:
        raise GradCAMError(f"Layer '{gradcam_layer.name}' is not connected to model output.") from exc

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(conceptlens_input, training=False)
        predicted_index = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_index]
    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise GradCAMError(f"No gradient was available at '{gradcam_layer.name}'.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))

    heatmap_image = Image.fromarray(np.uint8(heatmap.numpy() * 255), mode="L").resize(
        original_image.size, resample=Image.Resampling.BILINEAR
    )
    resized_heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    heatmap_rgb = plt.colormaps["jet"](resized_heatmap)[..., :3]
    original_array = np.asarray(original_image)
    overlay_array = np.uint8(np.clip(0.60 * original_array + 0.40 * heatmap_rgb * 255, 0, 255))

    index = int(predicted_index.numpy())
    scores = predictions[0].numpy()
    record = class_record(index)
    return {
        "predicted_index": index,
        "predicted_class": record["class_id"],
        "predicted_display_name": record["display_name"],
        "prediction_score": float(scores[index]),
        "top_predictions": prediction_records(scores),
        "probabilities": scores,
        "heatmap": resized_heatmap,
        "overlay": Image.fromarray(overlay_array),
        "layer_name": gradcam_layer.name,
    }


def get_denseActivations(folder_path: Path | str) -> pd.DataFrame:
    folder = Path(folder_path)
    image_paths = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        return pd.DataFrame(columns=ACTIVATION_COLUMNS + ["Class Name"])

    activation_model = activation_model_for(load_conceptlens_model())
    inputs = np.concatenate([preprocess_image_path(path) for path in image_paths], axis=0)
    activations = activation_model.predict(inputs, verbose=0)
    frames = []
    for path, vector in zip(image_paths, activations):
        frame = activation_rows_for(vector, path.name)
        frame["Class Name"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def image_processing_function(unique_folder_path: Path | str) -> pd.DataFrame:
    return get_denseActivations(unique_folder_path)


