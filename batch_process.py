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
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image

from demonstrator import (
    CLASS_NAMES,
    GradCAMError,
    figure_to_png_bytes,
    generate_gradcam,
    get_base_activations,
    load_conceptlens_model,
    render_concept_visualization,
    render_gradcam_visualization,
    process_activation_data,
)


NOISE_LEVELS = (0, 10, 20, 30, 40, 50, 80)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NOISE_METHOD = "gaussian"
SIGMA_AT_100_PERCENT = 40.0


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


def artifact_path(directory: Path, filename: str) -> Path:
    """Return a direct artifact path, refusing to overwrite an existing file."""
    target = directory / filename
    if target.exists():
        raise ArtifactConflictError(
            f"Artifact conflict: refusing to overwrite existing file {target}. "
            "Rename or move the existing artifact before rerunning this condition."
        )
    return target


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        name: output_dir / name
        for name in (
            "noisy", "gradcam", "concepts", "csv", "logs",
            "concept_visualizations", "gradcam_visualizations",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def csv_append_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    """Open a result CSV for append, retaining its original header and rows."""
    needs_header = not path.exists() or path.stat().st_size == 0
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


def load_manifest(output_dir: Path, input_dir: Path, images: list[Path]) -> dict[str, Any]:
    """Load the ledger, importing legacy successful CSV rows on first use.

    Legacy CSVs did not record a content hash.  During this one-time import,
    their successes are associated with the current bytes at that path; every
    later run compares the recorded hash with the source before deciding to
    skip it.
    """
    path = manifest_path(output_dir)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if loaded.get("schema_version") == 1 and isinstance(loaded.get("images"), dict):
                return loaded
        except (json.JSONDecodeError, OSError, AttributeError):
            logging.warning("Ignoring unreadable processing manifest: %s", path)

    manifest: dict[str, Any] = {"schema_version": 1, "images": {}}
    legacy_successes: dict[str, set[int]] = {}
    for filename in ("experiment_summary.csv", "predictions.csv"):
        csv_path = output_dir / "csv" / filename
        if not csv_path.exists():
            continue
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("processing_status") != "success":
                        continue
                    try:
                        level = int(row["noise_level"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if level in NOISE_LEVELS and row.get("image_name"):
                        legacy_successes.setdefault(row["image_name"], set()).add(level)
        except OSError:
            logging.warning("Could not import legacy result CSV: %s", csv_path)

    for source_path in images:
        image_name = source_path.relative_to(input_dir).as_posix()
        completed = legacy_successes.get(image_name, set())
        if not completed:
            continue
        source_digest = content_hash(source_path)
        conditions = {str(level): {"status": "success", "imported_from_legacy_csv": True}
                      for level in sorted(completed)}
        manifest["images"][image_name] = {"versions": {source_digest: {
            "source_hash": source_digest,
            "completed_noise_levels": sorted(completed),
            "conditions": conditions,
            "processing_status": "complete" if set(completed) == set(NOISE_LEVELS) else "incomplete",
        }}}
    return manifest


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


def activation_model_for(model: Any) -> Model:
    """Reuse the exact 64-neuron layer selected by get_denseActivations()."""
    return Model(inputs=model.input, outputs=[layer.output for layer in model.layers[-3:-2]])


def concept_input_for(image_path: Path) -> np.ndarray:
    """Mirror get_denseActivations' Keras load_img and ImageDataGenerator scaling.

    This intentionally differs from Grad-CAM's Pillow resize because the
    Streamlit concept pipeline uses Keras' default target-size interpolation.
    """
    loaded = keras_image.load_img(image_path, target_size=(224, 224))
    pixels = keras_image.img_to_array(loaded)
    return np.expand_dims(pixels, axis=0) / 255.0


def concept_rows_for(
    activations: np.ndarray, calibration: dict[str, list[list[Any]]], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the unchanged spreadsheet mapping, adding only provenance columns."""
    # process_activation_data indexes activations by DataFrame column, exactly
    # as image_processing_function does after get_denseActivations().
    activation_frame = pd.DataFrame(activations, columns=range(activations.shape[1]))
    concepts, non_target_percentages = process_activation_data(activation_frame, calibration)
    rows: list[dict[str, Any]] = []
    for concept, percentage in zip(concepts, non_target_percentages):
        # The existing public result exposes one value per concept, not which
        # mapped neuron caused detection.  Record the calibration neuron(s)
        # faithfully, and leave activation_value blank rather than infer it.
        neuron_ids = [str(values[0]) for values in calibration.get(concept, [])]
        rows.append({
            **metadata,
            "detected_concept": concept,
            "neuron_id": "; ".join(neuron_ids),
            "activation_value": "",
            "non_target_percentage": percentage,
        })
    return rows


def save_concept_visualization(
    image_path: Path, image_name: str, concept_rows: list[dict[str, Any]], output_path: Path
) -> None:
    """Save the same shared concept report used for Streamlit downloads."""
    concepts_frame = pd.DataFrame({
        "Detected Concept": [row["detected_concept"] for row in concept_rows],
        "Non-Target Percentage": [row["non_target_percentage"] for row in concept_rows],
    })
    with Image.open(image_path) as input_image:
        figure = render_concept_visualization(input_image, image_name, concepts_frame)
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
        raise ValueError(f"No JPEG or PNG images found in {input_dir}")
    manifest = load_manifest(output_dir, input_dir, images)
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
        # This also writes the one-time legacy CSV import when no new work is
        # required, without opening or modifying result CSVs.
        save_manifest(output_dir, manifest)
        return

    noise_metadata_fields = ["noise_method", "noise_sigma", "reproducibility_seed", "image_width", "image_height"]
    prediction_fields = ["image_name", "noise_level", *noise_metadata_fields, "predicted_class", "predicted_class_score", *CLASS_NAMES,
                         "processing_status", "error_message"]
    concept_fields = ["image_name", "noise_level", *noise_metadata_fields,
                      "detected_concept", "neuron_id", "activation_value", "non_target_percentage"]
    summary_fields = ["image_name", "noise_level", *noise_metadata_fields,
                      "original_image_reference", "noisy_image_reference",
                      "predicted_class", "predicted_class_score", "number_of_concepts", "concept_results_reference",
                      "gradcam_output_reference", "concept_visualization_reference",
                      "gradcam_visualization_reference", "processing_status", "error_details"]
    prediction_file, prediction_writer = csv_append_writer(dirs["csv"] / "predictions.csv", prediction_fields)
    concept_file, concept_writer = csv_append_writer(dirs["csv"] / "concepts.csv", concept_fields)
    summary_file, summary_writer = csv_append_writer(dirs["csv"] / "experiment_summary.csv", summary_fields)

    try:
        model = load_conceptlens_model()
        activations_model = activation_model_for(model)
        calibration = get_base_activations()
        for source_path, source_digest, record, selected_levels in work_items:
            relative_path = source_path.relative_to(input_dir)
            image_name = relative_path.as_posix()
            artifact_base_name = relative_path.stem
            original_reference = str(source_path)
            for level in selected_levels:
                condition_path = source_path
                sigma = noise_sigma(level)
                reproducibility_seed = condition_seed(seed, image_name, level)
                noisy_reference = ""
                gradcam_reference = ""
                concept_reference = ""
                concept_visualization_reference = ""
                gradcam_visualization_reference = ""
                condition_metadata: dict[str, Any] = {
                    "image_name": image_name,
                    "noise_level": level,
                    "noise_method": NOISE_METHOD,
                    "noise_sigma": sigma,
                    "reproducibility_seed": reproducibility_seed,
                    "image_width": "",
                    "image_height": "",
                }
                prediction_row: dict[str, Any] = {**condition_metadata,
                                                   "processing_status": "failed", "error_message": ""}
                summary_row: dict[str, Any] = {**condition_metadata,
                                                "original_image_reference": original_reference, "noisy_image_reference": "",
                                                "predicted_class": "", "predicted_class_score": "", "number_of_concepts": 0,
                                                "concept_results_reference": "", "gradcam_output_reference": "",
                                                "concept_visualization_reference": "", "gradcam_visualization_reference": "",
                                                "processing_status": "failed", "error_details": ""}
                try:
                    # Save a consistently named image for every condition,
                    # including 0%. The 0% artifact is only a reportable copy;
                    # inference still uses the original source path, preserving
                    # the existing zero-noise prediction pipeline.
                    with Image.open(source_path) as source:
                        generated = apply_rgb_noise(source, level, np.random.default_rng(reproducibility_seed))
                    noisy_artifact_path = artifact_path(
                        dirs["noisy"], f"{artifact_base_name}_noise_{level}.jpg"
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
                        rgb = inference_image.convert("RGB").resize((224, 224))
                        inference_input = np.expand_dims(np.asarray(rgb, dtype=np.float32) / 255.0, axis=0)
                    scores = model.predict(inference_input, verbose=0)[0]
                    predicted_index = int(np.argmax(scores))
                    prediction_row.update({"predicted_class": CLASS_NAMES[predicted_index],
                                           "predicted_class_score": float(scores[predicted_index]),
                                           **{name: float(scores[index]) for index, name in enumerate(CLASS_NAMES)}})

                    # Use the existing UI's concept-specific preprocessing,
                    # not the separate Grad-CAM/Pillow preprocessing above.
                    activations = activations_model.predict(concept_input_for(condition_path), verbose=0)
                    rows = concept_rows_for(activations, calibration, condition_metadata)
                    concept_path = artifact_path(
                        dirs["concepts"], f"{artifact_base_name}_noise_{level}.json"
                    )
                    concept_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
                    concept_reference = relative_reference(concept_path, output_dir)
                    display_name = f"{relative_path.name} (noise {level}%)"
                    concept_visualization_path = artifact_path(
                        dirs["concept_visualizations"],
                        f"{artifact_base_name}_noise_{level}_concepts.png",
                    )
                    save_concept_visualization(condition_path, display_name, rows, concept_visualization_path)
                    concept_visualization_reference = relative_reference(concept_visualization_path, output_dir)

                    gradcam = generate_gradcam(condition_path, model=model)
                    overlay_path = artifact_path(
                        dirs["gradcam"], f"{artifact_base_name}_noise_{level}_gradcam.png"
                    )
                    gradcam["overlay"].save(overlay_path)
                    gradcam_reference = relative_reference(overlay_path, output_dir)
                    gradcam_visualization_path = artifact_path(
                        dirs["gradcam_visualizations"],
                        f"{artifact_base_name}_noise_{level}_gradcam.png",
                    )
                    save_gradcam_visualization(condition_path, display_name, gradcam, gradcam_visualization_path)
                    gradcam_visualization_reference = relative_reference(gradcam_visualization_path, output_dir)
                    # Append concept rows only once the entire condition has
                    # succeeded, so a Grad-CAM failure cannot create duplicate
                    # concept CSV records when this level is retried.
                    for row in rows:
                        concept_writer.writerow(row)
                    # Keep the prediction source consistent with existing model.predict.
                    prediction_row["processing_status"] = "success"
                    summary_row.update({"predicted_class": prediction_row["predicted_class"],
                                        "predicted_class_score": prediction_row["predicted_class_score"],
                                        "number_of_concepts": len(rows), "concept_results_reference": concept_reference,
                                        "gradcam_output_reference": gradcam_reference,
                                        "concept_visualization_reference": concept_visualization_reference,
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
        concept_file.close()
        summary_file.close()


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
