from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

import batch_process
from batch_process import NOISE_LEVELS, PIPELINE_METADATA, noise_sigma, run_batch
from conceptlens_model import (
    CLASS_NAMES,
    activation_model_for,
    file_sha256,
    get_model_config,
    get_class_records,
    load_conceptlens_model,
    preprocess_image_path,
)
from demonstrator import (
    activation_rows_for,
    generate_gradcam,
    load_activation_scene_associations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = PROJECT_ROOT / "examples" / "highway.jpg"


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_conceptlens_model()
        cls.input_tensor = preprocess_image_path(EXAMPLE_IMAGE)
        cls.scores = cls.model.predict(cls.input_tensor, verbose=0)[0]

    def test_model_and_softmax(self):
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.model.output_shape, (None, 50))
        self.assertEqual(self.input_tensor.shape, (1, 224, 224, 3))
        self.assertTrue(np.isfinite(self.scores).all())
        self.assertTrue((self.scores >= 0).all())
        self.assertAlmostEqual(float(self.scores.sum()), 1.0, places=5)

    def test_verified_label_mapping(self):
        records = get_class_records()
        self.assertEqual(len(records), 50)
        self.assertEqual(tuple(item["class_id"] for item in records), CLASS_NAMES)
        self.assertEqual(CLASS_NAMES[0], "airport_terminal")
        self.assertEqual(CLASS_NAMES[24], "highway")
        self.assertEqual(CLASS_NAMES[49], "youth_hostel")

    def test_activation_vector(self):
        vector = activation_model_for(self.model).predict(self.input_tensor, verbose=0)[0]
        self.assertEqual(vector.shape, (64,))
        self.assertTrue(np.isfinite(vector).all())
        rows = activation_rows_for(vector)
        self.assertEqual(len(rows), 64)
        self.assertEqual(int(rows["Reference Available"].sum()), 61)
        self.assertEqual(int(rows["Association Available"].sum()), 34)

    def test_activation_scene_associations(self):
        associations = load_activation_scene_associations()
        self.assertEqual(len(associations), 64)
        self.assertEqual(
            associations["evidence_tier"].value_counts().to_dict(),
            {"ambiguous": 27, "moderate": 20, "high": 14, "inactive": 3},
        )
        self.assertEqual(associations.loc[33, "primary_scene"], "highway")
        self.assertEqual(associations.loc[33, "evidence_tier"], "high")
        self.assertEqual(associations.loc[40, "primary_scene"], "kitchen")
        self.assertEqual(associations.loc[27, "evidence_tier"], "inactive")

    def test_regenerated_positive_and_negative_examples(self):
        config = get_model_config()
        associations = load_activation_scene_associations()
        positive_path = PROJECT_ROOT / config["positive_examples"]
        negative_path = PROJECT_ROOT / config["negative_examples"]
        self.assertEqual(file_sha256(positive_path), config["positive_examples_sha256"])
        self.assertEqual(file_sha256(negative_path), config["negative_examples_sha256"])
        positive = pd.read_csv(positive_path)
        negative = pd.read_csv(negative_path)
        self.assertEqual(list(positive.columns), [str(index) for index in range(64)])
        self.assertEqual(list(negative.columns), [str(index) for index in range(64)])
        self.assertEqual(
            positive.count().tolist(), associations["positive_example_count"].tolist()
        )
        self.assertEqual(
            negative.count().tolist(), associations["negative_example_count"].tolist()
        )

    def test_gradcam(self):
        result = generate_gradcam(EXAMPLE_IMAGE, model=self.model)
        with Image.open(EXAMPLE_IMAGE) as source:
            width, height = source.size
        self.assertEqual(result["layer_name"], "post_relu")
        self.assertEqual(result["heatmap"].shape, (height, width))
        self.assertEqual(result["overlay"].size, (width, height))
        self.assertGreaterEqual(float(result["heatmap"].min()), 0.0)
        self.assertLessEqual(float(result["heatmap"].max()), 1.0)

    def test_noise_protocol(self):
        self.assertEqual(NOISE_LEVELS, (0, 10, 20, 30, 40, 50, 80))
        self.assertEqual([noise_sigma(level) for level in NOISE_LEVELS], [0, 4, 8, 12, 16, 20, 32])


class EndToEndTests(unittest.TestCase):
    def test_batch_all_levels_and_incremental_skip(self):
        with tempfile.TemporaryDirectory(prefix="conceptlens-test-") as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "highway.jpg").write_bytes(EXAMPLE_IMAGE.read_bytes())

            run_batch(input_dir, output_dir, seed=42)
            summary_path = output_dir / "csv" / "experiment_summary.csv"
            size_before = summary_path.stat().st_size
            run_batch(input_dir, output_dir, seed=42)
            self.assertEqual(summary_path.stat().st_size, size_before)

            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual([int(row["noise_level"]) for row in summary], list(NOISE_LEVELS))
            self.assertTrue(all(row["processing_status"] == "success" for row in summary))

            with (output_dir / "csv" / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 7)
            for row in predictions:
                self.assertAlmostEqual(sum(float(row[name]) for name in CLASS_NAMES), 1.0, places=5)
                self.assertEqual(row["model_sha256"], PIPELINE_METADATA["model_sha256"])
                self.assertEqual(row["preprocessing"], PIPELINE_METADATA["preprocessing"])
                self.assertTrue(row["source_image_path"])
                self.assertEqual(len(row["source_image_hash"]), 64)
                self.assertTrue(row["attempt_id"])

            with (output_dir / "csv" / "activations.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                activations = list(csv.DictReader(handle))
            self.assertEqual(len(activations), 7 * 64)
            self.assertEqual(
                sum(row["association_available"] == "True" for row in activations), 7 * 34
            )
            self.assertTrue(all(row["scene_associations_sha256"] for row in activations))

            manifest = json.loads((output_dir / "processing_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["pipeline"], PIPELINE_METADATA)
            image_record = manifest["images"]["highway.jpg"]
            version = next(iter(image_record["versions"].values()))
            self.assertEqual(version["completed_noise_levels"], list(NOISE_LEVELS))
            self.assertEqual(version["processing_status"], "complete")

    def test_failed_condition_retries_without_overwriting_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="conceptlens-retry-test-") as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "highway.jpg").write_bytes(EXAMPLE_IMAGE.read_bytes())

            real_gradcam = batch_process.generate_gradcam
            calls = 0

            def fail_first_gradcam(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("intentional retry test failure")
                return real_gradcam(*args, **kwargs)

            with patch("batch_process.generate_gradcam", side_effect=fail_first_gradcam):
                run_batch(input_dir, output_dir, seed=42)

            run_batch(input_dir, output_dir, seed=42)

            with (output_dir / "csv" / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                predictions = list(csv.DictReader(handle))
            level_zero = [row for row in predictions if row["noise_level"] == "0"]
            self.assertEqual([row["processing_status"] for row in level_zero], ["failed", "success"])
            self.assertEqual(len({row["attempt_id"] for row in level_zero}), 2)
            self.assertEqual(len(predictions), len(NOISE_LEVELS) + 1)

            noisy_files = list((output_dir / "noisy").glob("*.jpg"))
            self.assertEqual(len(noisy_files), len(NOISE_LEVELS) + 1)
            manifest = json.loads(
                (output_dir / "processing_manifest.json").read_text(encoding="utf-8")
            )
            version = next(
                iter(manifest["images"]["highway.jpg"]["versions"].values())
            )
            self.assertEqual(version["completed_noise_levels"], list(NOISE_LEVELS))
            self.assertEqual(version["processing_status"], "complete")
            self.assertEqual(version["conditions"]["0"]["status"], "success")
            self.assertTrue(version["conditions"]["0"]["artifacts"]["gradcam_overlay"])

    def test_legacy_manifest_is_not_reused(self):
        with tempfile.TemporaryDirectory(prefix="conceptlens-legacy-test-") as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "highway.jpg").write_bytes(EXAMPLE_IMAGE.read_bytes())
            legacy = {"schema_version": 1, "images": {}}
            manifest_path = output_dir / "processing_manifest.json"
            manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible model pipeline"):
                run_batch(input_dir, output_dir, seed=42)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), legacy)

    def test_streamlit_pages_render_without_exceptions(self):
        for relative_path in ("Home.py", "pages/Explore_ConceptLens.py"):
            app = AppTest.from_file(str(PROJECT_ROOT / relative_path), default_timeout=60).run()
            self.assertEqual(len(app.exception), 0, relative_path)


if __name__ == "__main__":
    unittest.main()


