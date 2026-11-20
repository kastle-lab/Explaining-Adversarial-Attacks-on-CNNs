"""Build discovery-only scene mappings and held-out activation rates from Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


NEURON_COUNT = 64
MIN_DISCOVERY_TARGET = 10
MIN_DISCOVERY_POSITIVE = 5
MIN_HELD_OUT_TARGET = 10
MIN_HELD_OUT_NON_TARGET = 30
CUTOFFS = (0, 20, 40, 60)
SPLIT_SEED = "conceptlens-scene-split-v1"
RATE_COLUMNS = (
    "neuron_id", "scene_id", "cutoff_pct",
    "target_activated", "target_total", "target_pct",
    "non_target_activated", "non_target_total", "non_target_pct",
)
ARTIFACT_NAMES = {
    "activation_reference": "activation_reference_50class.csv",
    "activation_scene_associations": "activation_scene_associations_50class.csv",
    "positive_examples": "positive_images_50class.csv",
    "negative_examples": "negative_images_50class.csv",
    "scene_activation_split": "scene_activation_split_50class.csv",
    "scene_activation_rates": "scene_activation_rates_50class.csv",
}


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Calculate tie-aware ROC AUC from ranks without an sklearn dependency."""
    values = np.concatenate([positive, negative])
    ranks = pd.Series(values).rank(method="average").to_numpy()
    positive_count = len(positive)
    negative_count = len(negative)
    rank_sum = ranks[:positive_count].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def evidence_tier(
    maximum: float, primary_count: int, primary_auc: float, auc_margin: float
) -> str:
    if maximum == 0:
        return "inactive"
    if primary_count >= 20 and primary_auc >= 0.85 and auc_margin >= 0.05:
        return "high"
    if primary_auc >= 0.80 and auc_margin >= 0.03:
        return "moderate"
    return "ambiguous"


def cutoff_value(reference_max: float, cutoff_pct: int) -> float:
    return reference_max * (cutoff_pct / 100)


def select_cutoff(activation: float, reference_max: float) -> int | None:
    """Pick the highest strictly exceeded cutoff using the calibration comparison."""
    if reference_max <= 0:
        return None
    return next(
        (cutoff for cutoff in reversed(CUTOFFS) if activation > cutoff_value(reference_max, cutoff)),
        None,
    )


def held_out_rates(values: np.ndarray, is_target: np.ndarray, reference_max: float) -> list[dict]:
    target_total = int(is_target.sum())
    non_target_total = len(is_target) - target_total
    if target_total < MIN_HELD_OUT_TARGET or non_target_total < MIN_HELD_OUT_NON_TARGET:
        raise ValueError("Insufficient held-out scene support.")
    rows = []
    for cutoff in CUTOFFS:
        activated = values > cutoff_value(reference_max, cutoff)
        target_activated = int(np.count_nonzero(activated & is_target))
        non_target_activated = int(np.count_nonzero(activated & ~is_target))
        rows.append(dict(
            cutoff_pct=cutoff, target_activated=target_activated, target_total=target_total,
            target_pct=100 * target_activated / target_total,
            non_target_activated=non_target_activated, non_target_total=non_target_total,
            non_target_pct=100 * non_target_activated / non_target_total,
        ))
    return rows


def split_rows(filenames: pd.Series, scenes: pd.Series) -> pd.Series:
    """Assign approximately one quarter of each scene by stable filename hashes."""
    held_out = pd.Series(False, index=filenames.index)
    keys = filenames.map(lambda name: hashlib.sha256(f"{SPLIT_SEED}:{name}".encode("utf-8")).hexdigest())
    for _, members in filenames.groupby(scenes):
        count = len(members)
        holdout_count = min(count - 1, max(1, (count + 2) // 4))
        ordered = sorted(members.index, key=lambda index: (keys[index], filenames[index]))
        held_out.loc[ordered[:holdout_count]] = True
    return held_out


def list_frame(values: dict[int, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {str(neuron): pd.Series(items, dtype="object") for neuron, items in values.items()}
    )


def fit_discovery(activations: pd.DataFrame, filenames: pd.Series, scenes: pd.Series):
    scene_counts = scenes.value_counts()
    eligible = sorted(scene_counts[scene_counts >= MIN_DISCOVERY_TARGET].index)
    if len(eligible) < 2:
        raise ValueError("Discovery requires at least two scenes with ten or more images.")
    scene_masks = {scene: scenes.to_numpy() == scene for scene in eligible}
    positive_images = {}
    negative_images = {}
    associations = []

    for neuron_id, column in enumerate(activations):
        values = activations[column].to_numpy(dtype=np.float64)
        maximum = float(values.max())
        positive_mask = (values >= cutoff_value(maximum, 80)) & (values > 0)
        negative_mask = (values <= cutoff_value(maximum, 20)) & (values > 0)
        positive_images[neuron_id] = filenames.loc[positive_mask].tolist()
        negative_images[neuron_id] = filenames.loc[negative_mask].tolist()

        scores = []
        for scene, class_mask in scene_masks.items():
            positive, negative = values[class_mask], values[~class_mask]
            scores.append((
                roc_auc(positive, negative), scene, int(class_mask.sum()),
                float(positive.mean()), float(negative.mean()), int(np.count_nonzero(positive)),
            ))

        scores.sort(key=lambda score: (-score[0], score[1]))
        primary_auc, primary_scene, primary_count, primary_mean, other_mean, positive_count = scores[0]
        secondary_auc, secondary_scene, *_ = scores[1]
        margin = primary_auc - secondary_auc
        tier = evidence_tier(maximum, primary_count, primary_auc, margin)
        associations.append(dict(
            neuron_id=neuron_id, reference_max=maximum, reference_sample_count=len(activations),
            positive_example_count=int(positive_mask.sum()), negative_example_count=int(negative_mask.sum()),
            primary_scene=primary_scene if maximum else "", primary_auc=primary_auc if maximum else "",
            secondary_scene=secondary_scene if maximum else "", secondary_auc=secondary_auc if maximum else "",
            auc_margin=margin if maximum else "", primary_class_count=primary_count,
            primary_class_mean=primary_mean, other_classes_mean=other_mean,
            discovery_target_activated=positive_count, evidence_tier=tier,
            method="discovery-only one-vs-rest rank AUC",
        ))
    return associations, positive_images, negative_images


def calibrate(associations: list[dict], activations: pd.DataFrame, scenes: pd.Series, display_names: dict):
    """Measure frozen associations without using held-out activations to select mappings."""
    results = []
    scene_labels = scenes.to_numpy()
    for row in associations:
        neuron = row["neuron_id"]
        scene = row["primary_scene"]
        target = scene_labels == scene
        target_count = int(target.sum())
        non_target_count = len(target) - target_count
        if row["evidence_tier"] == "inactive":
            status = "inactive"
        elif row["evidence_tier"] == "ambiguous":
            status = "ambiguous"
        elif row["discovery_target_activated"] < MIN_DISCOVERY_POSITIVE:
            status = "insufficient_discovery_signal"
        elif target_count < MIN_HELD_OUT_TARGET or non_target_count < MIN_HELD_OUT_NON_TARGET:
            status = "insufficient_held_out_support"
        else:
            status = "retained"
            measurements = held_out_rates(activations.iloc[:, neuron].to_numpy(), target, row["reference_max"])
            results.extend({"neuron_id": neuron, "scene_id": scene, **rate} for rate in measurements)
        row.update(
            mapping_status=status,
            primary_scene_display=display_names[scene] if status == "retained" else "",
            held_out_target_count=target_count,
            held_out_non_target_count=non_target_count,
        )
    return results


def build_artifacts(
    activation_csv: Path, labels_json: Path, output_dir: Path, model_config: Path | None = None
) -> None:
    source_bytes = activation_csv.read_bytes()
    source = pd.read_csv(BytesIO(source_bytes))
    neuron_columns = [str(index) for index in range(NEURON_COUNT)]
    if list(source.columns) != [*neuron_columns, "filenames"]:
        raise ValueError("Activation CSV must contain numbered neuron columns 0 through 63, then filenames.")
    filenames = source["filenames"].str.replace(chr(92), "/", regex=False)
    if not filenames.str.fullmatch(r"[^/]+/[^/]+", na=False).all() or not filenames.str.rsplit("/").str[-1].is_unique:
        raise ValueError("Expected unique image names in scene-folder/filename format.")
    activations = source[neuron_columns].apply(pd.to_numeric, errors="raise")
    values = activations.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Dense ReLU activations must be finite and nonnegative.")
    scenes = filenames.str.split("/").str[0]
    labels = json.loads(labels_json.read_text(encoding="utf-8"))["classes"]
    display_names = {item["class_id"]: item["display_name"] for item in labels}
    if set(scenes) != display_names.keys():
        raise ValueError("Activation filename scene IDs do not match the verified class mapping.")

    held_out = split_rows(filenames, scenes)
    associations, positive, negative = fit_discovery(
        activations.loc[~held_out], filenames.loc[~held_out], scenes.loc[~held_out],
    )
    rates = calibrate(associations, activations.loc[held_out], scenes.loc[held_out], display_names)
    reference = pd.DataFrame(associations)[["neuron_id", "reference_max", "reference_sample_count"]]
    reference["source"] = "Stage 1 activation CSV discovery split"
    split = pd.DataFrame({
        "filename": filenames, "scene_id": scenes, "split": np.where(held_out, "held_out", "discovery"),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "activation_reference": reference,
        "activation_scene_associations": pd.DataFrame(associations),
        "positive_examples": list_frame(positive),
        "negative_examples": list_frame(negative),
        "scene_activation_split": split,
        "scene_activation_rates": pd.DataFrame(rates, columns=RATE_COLUMNS),
    }
    for key, frame in artifacts.items():
        frame.to_csv(output_dir / ARTIFACT_NAMES[key], index=False, lineterminator="\n")
    if model_config is not None:
        config = json.loads(model_config.read_text(encoding="utf-8"))
        for key, name in ARTIFACT_NAMES.items():
            config[key] = name
            config[f"{key}_sha256"] = hashlib.sha256((output_dir / name).read_bytes()).hexdigest().upper()
        config["scene_activation_source_sha256"] = hashlib.sha256(source_bytes).hexdigest().upper()
        config["scene_associations_note"] = (
            "One-vs-rest scene associations use discovery activations. "
            "The rates measure activation on other labeled scenes in the held-out split."
        )
        model_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(~held_out)} discovery rows, {sum(held_out)} held-out rows, {len(rates) // len(CUTOFFS)} retained mappings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-csv", required=True, type=Path)
    parser.add_argument("--labels", default=Path("class_labels_50.json"), type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument("--model-config", type=Path, help="Update the app's artifact paths and SHA-256 hashes.")
    args = parser.parse_args()
    build_artifacts(args.activation_csv, args.labels, args.output_dir, args.model_config)
