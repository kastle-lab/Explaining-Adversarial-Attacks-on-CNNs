#!/usr/bin/env python
"""Run reproducible Gaussian-noise ConceptLens experiments without Streamlit.

For a percentage ``p``, this experiment uses ``sigma = (p / 100) * 40`` in
8-bit pixel-intensity units and independently samples every RGB value from
``Normal(0, sigma**2)``. Values are added to the original pixels, rounded,
and clipped to [0, 255]. The 100% -> sigma 40 mapping is this experiment's
chosen reference protocol, not a universal Gaussian-noise maximum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from conceptlens_model import (
    CLASS_NAMES,
    MODEL_PATH,
    activation_model_for,
    class_record,
    get_model_config,
    load_conceptlens_model,
    preprocess_image_path,
)
from demonstrator import (
    activation_rows_for,
    figure_to_png_bytes,
    generate_gradcam,
    render_activation_visualization,
    render_gradcam_visualization,
)


NOISE_LEVELS = (0, 10, 20, 30, 40, 50, 80)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NOISE_METHOD = "gaussian"
SIGMA_AT_100_PERCENT = 40.0
MANIFEST_SCHEMA_VERSION = 2
MODEL_CONFIG = get_model_config()
PIPELINE_METADATA = {
    "model_checkpoint": MODEL_PATH.name,
    "model_sha256": MODEL_CONFIG["checkpoint_sha256"],
    "labels_file": MODEL_CONFIG["labels"],
    "activation_scene_associations": MODEL_CONFIG["activation_scene_associations"],
    "activation_scene_associations_sha256": MODEL_CONFIG[
        "activation_scene_associations_sha256"
    ],
    "preprocessing": "RGB; resize 224x224; float32; pixel/255.0",
    "report_schema_version": 2,
}


def noise_sigma(noise_level: int) -> float:
    """Return the protocol's Gaussian standard deviation in intensity units."""
    if noise_level not in NOISE_LEVELS:
        raise ValueError(f"Unsupported noise level {noise_level}; expected one of {NOISE_LEVELS}.")
    return (noise_level / 100.0) * SIGMA_AT_100_PERCENT


def apply_rgb_noise(source: Image.Image, noise_level: int, rng: np.random.Generator) -> Image.Image:
    """Add independent zero-mean Gaussian noise to every RGB channel.

    A zero level returns a copy with identical pixel data and image mode.
    """
    sigma = noise_sigma(noise_level)
    if noise_level == 0:
        return source.copy()

    has_alpha = "A" in source.getbands()
    working = source.convert("RGBA" if has_alpha else "RGB")
    pixels = np.asarray(working).copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=pixels[..., :3].shape)
    noisy_rgb = np.clip(pixels[..., :3].astype(np.float32) + np.rint(noise), 0, 255).astype(np.uint8)
    pixels[..., :3] = noisy_rgb
    return Image.fromarray(pixels, mode="RGBA" if has_alpha else "RGB")


def condition_seed(base_seed: int, relative_image_path: str, noise_level: int, noise_method: str = NOISE_METHOD) -> int:
    """Create a platform-independent deterministic NumPy seed for one condition."""
    material = f"{base_seed}|{relative_image_path}|{noise_level}|{noise_method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def composite_on_white(image: Image.Image) -> Image.Image:
    """Flatten alpha onto white before JPEG encoding, which cannot retain alpha."""
    if "A" not in image.getbands():
        return image.convert("RGB")
    background = Image.new("RGB", image.size, "white")
    rgba = image.convert("RGBA")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


class ArtifactConflictError(FileExistsError):
    """Raised when a simple artifact name would replace an existing result."""


def artifact_path(directory: Path, filename: str, collision_suffix: str = "") -> Path:
    """Return a non-overwriting artifact path.

    The legacy simple name is retained for a condition's first attempt. If a
    retained artifact already occupies that name, retries, explicit reruns,
    changed source versions, and same-stem inputs receive a unique suffix.
    """
    target = directory / filename
    if not target.exists():
        return target
    if collision_suffix:
        alternative = target.with_name(f"{target.stem}_{collision_suffix}{target.suffix}")
        if not alternative.exists():
            return alternative
    raise ArtifactConflictError(
        f"Artifact conflict: refusing to overwrite existing file {target}."
    )


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        name: output_dir / name
        for name in (
            "noisy", "gradcam", "activations", "csv", "logs",
            "activation_visualizations", "gradcam_visualizations",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def close_batch_logging() -> None:
    """Release root log handlers so Windows can remove completed test/output trees."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.flush()
        handler.close()
        root_logger.removeHandler(handler)


def csv_append_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    """Open a result CSV for append, retaining its original header and rows."""
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open(newline="", encoding="utf-8") as existing:
            current_header = next(csv.reader(existing), [])
        if current_header != fieldnames:
            raise ValueError(
                f"Existing CSV schema is incompatible with the 50-class pipeline: {path}. "
                "Use a new output directory so legacy results remain unchanged."
            )
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    if needs_header:
        writer.writeheader()
    return handle, writer


def content_hash(path: Path) -> str:
    """Return a stable hash of the exact source bytes without loading the image."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "processing_manifest.json"


def load_manifest(output_dir: Path) -> dict[str, Any]:
    """Load only a ledger tied to this exact 50-class model pipeline."""
    path = manifest_path(output_dir)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if (
                loaded.get("schema_version") == MANIFEST_SCHEMA_VERSION
                and loaded.get("pipeline") == PIPELINE_METADATA
                and isinstance(loaded.get("images"), dict)
            ):
                return loaded
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            raise ValueError(
                f"Existing processing manifest is unreadable: {path}. "
                "Use a new output directory to preserve existing results."
            ) from exc
        raise ValueError(
            f"Existing processing manifest belongs to an incompatible model pipeline: {path}. "
            "Use a new output directory so legacy results remain unchanged."
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline": PIPELINE_METADATA,
        "images": {},
    }


def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically update the ledger; experiment artifacts are never replaced."""
    path = manifest_path(output_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def version_record(manifest: dict[str, Any], image_name: str, source_digest: str) -> dict[str, Any]:
    image_record = manifest["images"].setdefault(image_name, {"versions": {}})
    return image_record["versions"].setdefault(source_digest, {
        "source_hash": source_digest,
        "completed_noise_levels": [],
        "conditions": {},
        "processing_status": "incomplete",
    })


def levels_to_process(record: dict[str, Any], new_only: bool, rerun: bool) -> tuple[int, ...]:
    if rerun:
        return NOISE_LEVELS
    if new_only and record["conditions"]:
        return ()
    return tuple(level for level in NOISE_LEVELS
                 if record["conditions"].get(str(level), {}).get("status") != "success")


def relative_reference(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def activation_rows_for_batch(
    activations: np.ndarray, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Record all 64 activations and evidence-qualified scene associations."""
    frame = activation_rows_for(np.asarray(activations).reshape(-1))
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append({
            **metadata,
            "neuron_id": int(row["Neuron ID"]),
            "activation_value": float(row["Activation Value"]),
            "reference_maximum": float(row["Reference Maximum"]),
            "reference_percentage": (
                float(row["Reference Percentage"])
                if pd.notna(row["Reference Percentage"])
                else ""
            ),
            "reference_available": bool(row["Reference Available"]),
            "scene_association": row["Scene Association"],
            "primary_scene_association_id": row["Primary Scene Association ID"],
            "association_auc": (
                float(row["Association AUC"]) if pd.notna(row["Association AUC"]) else ""
            ),
            "secondary_scene_association_id": row["Secondary Scene Association ID"],
            "secondary_association_auc": (
                float(row["Secondary Association AUC"])
                if pd.notna(row["Secondary Association AUC"])
                else ""
            ),
            "auc_margin": float(row["AUC Margin"]) if pd.notna(row["AUC Margin"]) else "",
            "evidence_tier": row["Evidence Tier"],
            "association_available": bool(row["Association Available"]),
        })
    return rows


def save_activation_visualization(
    image_path: Path, image_name: str, activation_rows: list[dict[str, Any]], output_path: Path
) -> None:
    """Save the same scene-association activation report used by Streamlit."""
    activation_frame = pd.DataFrame({
        "Neuron ID": [row["neuron_id"] for row in activation_rows],
        "Activation Value": [row["activation_value"] for row in activation_rows],
        "Reference Maximum": [row["reference_maximum"] for row in activation_rows],
        "Reference Percentage": [
            np.nan if row["reference_percentage"] == "" else row["reference_percentage"]
            for row in activation_rows
        ],
        "Reference Available": [row["reference_available"] for row in activation_rows],
        "Scene Association": [row["scene_association"] for row in activation_rows],
        "Primary Scene Association ID": [
            row["primary_scene_association_id"] for row in activation_rows
        ],
        "Association AUC": [
            np.nan if row["association_auc"] == "" else row["association_auc"]
            for row in activation_rows
        ],
        "Secondary Scene Association ID": [
            row["secondary_scene_association_id"] for row in activation_rows
        ],
        "Secondary Association AUC": [
            np.nan if row["secondary_association_auc"] == "" else row["secondary_association_auc"]
            for row in activation_rows
        ],
        "AUC Margin": [
            np.nan if row["auc_margin"] == "" else row["auc_margin"]
            for row in activation_rows
        ],
        "Evidence Tier": [row["evidence_tier"] for row in activation_rows],
        "Association Available": [row["association_available"] for row in activation_rows],
    })
    with Image.open(image_path) as input_image:
        figure = render_activation_visualization(input_image, image_name, activation_frame)
    output_path.write_bytes(figure_to_png_bytes(figure))
    plt.close(figure)


def save_gradcam_visualization(
    image_path: Path, image_name: str, gradcam_result: dict[str, Any], output_path: Path
) -> None:
    """Save the same shared Grad-CAM report used for Streamlit downloads."""
    with Image.open(image_path) as original_image:
        figure = render_gradcam_visualization(original_image, image_name, gradcam_result)
    output_path.write_bytes(figure_to_png_bytes(figure))
    plt.close(figure)


def run_batch(input_dir: Path, output_dir: Path, seed: int, *, new_only: bool = False, rerun: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    dirs = make_output_dirs(output_dir)
    logging.basicConfig(filename=dirs["logs"] / "batch_process.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    images = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        close_batch_logging()
        raise ValueError(f"No JPEG or PNG images found in {input_dir}")
    try:
        manifest = load_manifest(output_dir)
    except Exception:
        close_batch_logging()
        raise
    # A single invocation receives one id, while path + content hash make every
    # artifact version unambiguous.  This deliberately leaves failed-attempt
    # files in place for investigation rather than overwriting them on retry.
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work_items: list[tuple[Path, str, dict[str, Any], tuple[int, ...]]] = []
    for source_path in images:
        image_name = source_path.relative_to(input_dir).as_posix()
        source_digest = content_hash(source_path)
        record = version_record(manifest, image_name, source_digest)
        selected_levels = levels_to_process(record, new_only, rerun)
        if selected_levels:
            work_items.append((source_path, source_digest, record, selected_levels))
        else:
            logging.info("Skipping %s; its current content version is already complete", source_path)
    if not work_items:
        # Persist the current pipeline-fingerprinted ledger without touching CSVs.
        save_manifest(output_dir, manifest)
        close_batch_logging()
        return

    source_metadata_fields = ["source_image_path", "source_image_hash", "attempt_id"]
    noise_metadata_fields = [
        "noise_method", "noise_sigma", "reproducibility_seed", "image_width", "image_height",
        "model_checkpoint", "model_sha256", "scene_associations_sha256", "preprocessing",
    ]
    prediction_fields = ["image_name", *source_metadata_fields, "noise_level", *noise_metadata_fields,
                         "predicted_class", "predicted_class_name", "predicted_class_score", *CLASS_NAMES,
                         "processing_status", "error_message"]
    activation_fields = ["image_name", *source_metadata_fields, "noise_level", *noise_metadata_fields,
                         "neuron_id", "activation_value", "reference_maximum",
                         "reference_percentage", "reference_available",
                         "scene_association", "primary_scene_association_id",
                         "association_auc", "secondary_scene_association_id",
                         "secondary_association_auc", "auc_margin", "evidence_tier",
                         "association_available"]
    summary_fields = ["image_name", *source_metadata_fields, "noise_level", *noise_metadata_fields,
                      "original_image_reference", "noisy_image_reference",
                      "predicted_class", "predicted_class_name", "predicted_class_score",
                      "number_of_activation_rows", "activation_results_reference",
                      "gradcam_output_reference", "activation_visualization_reference",
                      "gradcam_visualization_reference", "processing_status", "error_details"]
    prediction_file = activation_file = summary_file = None
    try:
        prediction_file, prediction_writer = csv_append_writer(
            dirs["csv"] / "predictions.csv", prediction_fields
        )
        activation_file, activation_writer = csv_append_writer(
            dirs["csv"] / "activations.csv", activation_fields
        )
        summary_file, summary_writer = csv_append_writer(
            dirs["csv"] / "experiment_summary.csv", summary_fields
        )
    except Exception:
        for handle in (prediction_file, activation_file, summary_file):
            if handle is not None:
                handle.close()
        close_batch_logging()
        raise

    try:
        model = load_conceptlens_model()
        activations_model = activation_model_for(model)
        for source_path, source_digest, record, selected_levels in work_items:
            relative_path = source_path.relative_to(input_dir)
            image_name = relative_path.as_posix()
            artifact_base_name = relative_path.stem
            path_digest = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:8]
            collision_suffix = f"{path_digest}_{source_digest[:8]}_{attempt_id}"
            original_reference = str(source_path)
            for level in selected_levels:
                condition_path = source_path
                sigma = noise_sigma(level)
                reproducibility_seed = condition_seed(seed, image_name, level)
                noisy_reference = ""
                gradcam_reference = ""
                activation_reference = ""
                activation_visualization_reference = ""
                gradcam_visualization_reference = ""
                condition_metadata: dict[str, Any] = {
                    "image_name": image_name,
                    "source_image_path": original_reference,
                    "source_image_hash": source_digest,
                    "attempt_id": attempt_id,
                    "noise_level": level,
                    "noise_method": NOISE_METHOD,
                    "noise_sigma": sigma,
                    "reproducibility_seed": reproducibility_seed,
                    "image_width": "",
                    "image_height": "",
                    "model_checkpoint": PIPELINE_METADATA["model_checkpoint"],
                    "model_sha256": PIPELINE_METADATA["model_sha256"],
                    "scene_associations_sha256": PIPELINE_METADATA[
                        "activation_scene_associations_sha256"
                    ],
                    "preprocessing": PIPELINE_METADATA["preprocessing"],
                }
                prediction_row: dict[str, Any] = {**condition_metadata,
                                                   "processing_status": "failed", "error_message": ""}
                summary_row: dict[str, Any] = {**condition_metadata,
                                                "original_image_reference": original_reference, "noisy_image_reference": "",
                                                "predicted_class": "", "predicted_class_name": "",
                                                "predicted_class_score": "", "number_of_activation_rows": 0,
                                                "activation_results_reference": "", "gradcam_output_reference": "",
                                                "activation_visualization_reference": "", "gradcam_visualization_reference": "",
                                                "processing_status": "failed", "error_details": ""}
                try:
                    # Save a consistently named image for every condition,
                    # including 0%. The 0% artifact is only a reportable copy;
                    # inference still uses the original source path, preserving
                    # the existing zero-noise prediction pipeline.
                    with Image.open(source_path) as source:
                        generated = apply_rgb_noise(source, level, np.random.default_rng(reproducibility_seed))
                    noisy_artifact_path = artifact_path(
                        dirs["noisy"], f"{artifact_base_name}_noise_{level}.jpg", collision_suffix
                    )
                    # JPEG cannot retain alpha: transparent PNGs are composited
                    # on white before quality-95 encoding. JPEG may add
                    # compression artifacts beyond the noise.
                    composite_on_white(generated).save(noisy_artifact_path, format="JPEG", quality=95)
                    noisy_reference = relative_reference(noisy_artifact_path, output_dir)
                    if level:
                        condition_path = noisy_artifact_path
                    summary_row["noisy_image_reference"] = noisy_reference

                    with Image.open(condition_path) as inference_image:
                        width, height = inference_image.size
                        condition_metadata.update({"image_width": width, "image_height": height})
                        prediction_row.update({"image_width": width, "image_height": height})
                        summary_row.update({"image_width": width, "image_height": height})
                    inference_input = preprocess_image_path(condition_path)
                    scores = model.predict(inference_input, verbose=0)[0]
                    predicted_index = int(np.argmax(scores))
                    predicted_record = class_record(predicted_index)
                    prediction_row.update({"predicted_class": predicted_record["class_id"],
                                           "predicted_class_name": predicted_record["display_name"],
                                           "predicted_class_score": float(scores[predicted_index]),
                                           **{name: float(scores[index]) for index, name in enumerate(CLASS_NAMES)}})

                    activations = activations_model.predict(inference_input, verbose=0)
                    rows = activation_rows_for_batch(activations, condition_metadata)
                    activation_path = artifact_path(
                        dirs["activations"], f"{artifact_base_name}_noise_{level}.json", collision_suffix
                    )
                    activation_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
                    activation_reference = relative_reference(activation_path, output_dir)
                    display_name = f"{relative_path.name} (noise {level}%)"
                    activation_visualization_path = artifact_path(
                        dirs["activation_visualizations"],
                        f"{artifact_base_name}_noise_{level}_activations.png",
                        collision_suffix,
                    )
                    save_activation_visualization(condition_path, display_name, rows, activation_visualization_path)
                    activation_visualization_reference = relative_reference(activation_visualization_path, output_dir)

                    gradcam = generate_gradcam(condition_path, model=model)
                    overlay_path = artifact_path(
                        dirs["gradcam"], f"{artifact_base_name}_noise_{level}_gradcam.png", collision_suffix
                    )
                    gradcam["overlay"].save(overlay_path)
                    gradcam_reference = relative_reference(overlay_path, output_dir)
                    gradcam_visualization_path = artifact_path(
                        dirs["gradcam_visualizations"],
                        f"{artifact_base_name}_noise_{level}_gradcam.png",
                        collision_suffix,
                    )
                    save_gradcam_visualization(condition_path, display_name, gradcam, gradcam_visualization_path)
                    gradcam_visualization_reference = relative_reference(gradcam_visualization_path, output_dir)
                    # Append activation rows only once the entire condition has
                    # succeeded, so a Grad-CAM failure cannot create duplicate
                    # activation CSV records when this level is retried.
                    for row in rows:
                        activation_writer.writerow(row)
                    # Keep the prediction source consistent with existing model.predict.
                    prediction_row["processing_status"] = "success"
                    summary_row.update({"predicted_class": prediction_row["predicted_class"],
                                        "predicted_class_name": prediction_row["predicted_class_name"],
                                        "predicted_class_score": prediction_row["predicted_class_score"],
                                        "number_of_activation_rows": len(rows),
                                        "activation_results_reference": activation_reference,
                                        "gradcam_output_reference": gradcam_reference,
                                        "activation_visualization_reference": activation_visualization_reference,
                                        "gradcam_visualization_reference": gradcam_visualization_reference,
                                        "processing_status": "success"})
                except Exception as exc:  # Per-case failure is deliberately non-fatal.
                    message = f"{type(exc).__name__}: {exc}"
                    logging.exception("Failed %s at noise %s", source_path, level)
                    prediction_row["error_message"] = message
                    summary_row["error_details"] = message
                prediction_writer.writerow(prediction_row)
                summary_writer.writerow(summary_row)
                status = prediction_row["processing_status"]
                record["conditions"][str(level)] = {
                    "status": status,
                    "attempt_id": attempt_id,
                    "error_message": prediction_row["error_message"],
                    "artifacts": {
                        "noisy_image": noisy_reference,
                        "activation_results": activation_reference,
                        "gradcam_overlay": gradcam_reference,
                        "activation_visualization": activation_visualization_reference,
                        "gradcam_visualization": gradcam_visualization_reference,
                    },
                }
                record["completed_noise_levels"] = sorted(
                    int(item) for item, state in record["conditions"].items()
                    if state.get("status") == "success"
                )
                record["processing_status"] = (
                    "complete" if set(record["completed_noise_levels"]) == set(NOISE_LEVELS) else "incomplete"
                )
                # Persist each condition so an interruption only leaves that
                # condition to retry rather than duplicating completed work.
                save_manifest(output_dir, manifest)
    finally:
        prediction_file.close()
        activation_file.close()
        summary_file.close()
        close_batch_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible ConceptLens RGB-noise batch experiments.")
    parser.add_argument("--input-dir", default=Path("input_images"), type=Path,
                        help="Directory containing JPEG or PNG input images (default: input_images).")
    parser.add_argument("--output-dir", default=Path("batch_results"), type=Path,
                        help="Directory for experiment outputs and the manifest (default: batch_results).")
    parser.add_argument("--seed", default=42, type=int, help="Base random seed (default: 42).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--new-only", action="store_true",
                      help="Process only source-content versions absent from the manifest; do not retry known failures.")
    mode.add_argument("--rerun", action="store_true",
                      help="Intentionally retry all current image versions; fails clearly rather than overwriting simple-name artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir is not an existing directory: {args.input_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit(f"--output-dir exists and is not a directory: {args.output_dir}")
    try:
        run_batch(args.input_dir, args.output_dir, args.seed, new_only=args.new_only, rerun=args.rerun)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
