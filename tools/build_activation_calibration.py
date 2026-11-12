"""Build reproducible Dense(64) reference artifacts from Stage 1 activations.

This derives statistical scene associations, not object-level ontology concepts.
For each neuron and eligible scene class, it calculates a one-vs-rest ROC AUC
using activation ranks. Classes with fewer than 10 reference images are
excluded from association selection. The original Stage 1 80%/20% maximum
rules are also used to regenerate positive and negative filename lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NEURON_COUNT = 64
MIN_CLASS_SAMPLES = 10


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


def parse_scene_ids(filenames: pd.Series) -> pd.Series:
    return (
        filenames.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[0]
    )


def list_frame(values: dict[int, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {str(neuron): pd.Series(items, dtype="object") for neuron, items in values.items()}
    )


def build_artifacts(activation_csv: Path, labels_json: Path, output_dir: Path) -> None:
    source = pd.read_csv(activation_csv)
    if source.shape[1] != NEURON_COUNT + 1 or source.columns[-1] != "filenames":
        raise ValueError("Activation CSV must contain 64 neuron columns followed by filenames.")

    activations = source.iloc[:, :NEURON_COUNT].apply(pd.to_numeric, errors="raise")
    scenes = parse_scene_ids(source["filenames"])
    label_payload = json.loads(labels_json.read_text(encoding="utf-8"))
    expected_scenes = [item["class_id"] for item in label_payload["classes"]]
    observed_scenes = sorted(scenes.unique())
    if observed_scenes != sorted(expected_scenes):
        raise ValueError("Activation filename scene IDs do not match the verified class mapping.")

    scene_counts = scenes.value_counts()
    eligible_scenes = sorted(
        scene for scene, count in scene_counts.items() if count >= MIN_CLASS_SAMPLES
    )
    positive_images: dict[int, list[str]] = {}
    negative_images: dict[int, list[str]] = {}
    association_rows: list[dict[str, object]] = []
    scene_array = scenes.to_numpy()

    for neuron_id in range(NEURON_COUNT):
        values = activations.iloc[:, neuron_id].to_numpy(dtype=np.float64)
        maximum = float(values.max())
        positive_mask = (values >= 0.80 * maximum) & (values != 0) if maximum else np.zeros(len(values), bool)
        negative_mask = (values <= 0.20 * maximum) & (values != 0) if maximum else np.zeros(len(values), bool)
        positive_images[neuron_id] = source.loc[positive_mask, "filenames"].tolist()
        negative_images[neuron_id] = source.loc[negative_mask, "filenames"].tolist()

        scores: list[tuple[float, str, int, float, float]] = []
        for scene in eligible_scenes:
            class_mask = scene_array == scene
            positive = values[class_mask]
            negative = values[~class_mask]
            scores.append(
                (
                    roc_auc(positive, negative),
                    scene,
                    int(class_mask.sum()),
                    float(positive.mean()),
                    float(negative.mean()),
                )
            )
        scores.sort(reverse=True)
        primary_auc, primary_scene, primary_count, primary_mean, other_mean = scores[0]
        secondary_auc, secondary_scene, _, _, _ = scores[1]
        margin = primary_auc - secondary_auc
        tier = evidence_tier(maximum, primary_count, primary_auc, margin)
        association_rows.append(
            {
                "neuron_id": neuron_id,
                "reference_max": maximum,
                "reference_sample_count": len(source),
                "positive_example_count": int(positive_mask.sum()),
                "negative_example_count": int(negative_mask.sum()),
                "primary_scene": "" if tier == "inactive" else primary_scene,
                "primary_scene_display": (
                    primary_scene.replace("_", " ") if tier in {"high", "moderate"} else ""
                ),
                "primary_auc": "" if tier == "inactive" else primary_auc,
                "secondary_scene": "" if tier == "inactive" else secondary_scene,
                "secondary_auc": "" if tier == "inactive" else secondary_auc,
                "auc_margin": "" if tier == "inactive" else margin,
                "primary_class_count": primary_count,
                "primary_class_mean": primary_mean,
                "other_classes_mean": other_mean,
                "evidence_tier": tier,
                "method": "one-vs-rest rank AUC; eligible classes have >=10 reference images",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(association_rows).to_csv(
        output_dir / "activation_scene_associations_50class.csv", index=False
    )
    list_frame(positive_images).to_csv(output_dir / "positive_images_50class.csv", index=False)
    list_frame(negative_images).to_csv(output_dir / "negative_images_50class.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-csv", required=True, type=Path)
    parser.add_argument("--labels", default=Path("class_labels_50.json"), type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_artifacts(arguments.activation_csv, arguments.labels, arguments.output_dir)
