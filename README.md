# ConceptLens: 50-class scene explanations

ConceptLens is a Streamlit application for inspecting a 50-class ResNet50V2 scene classifier. The interactive page preserves the original upload, prediction-score, and Grad-CAM workflow. The batch pipeline additionally records the verified 64-neuron activation representation.

## Model

The configured checkpoint is `models/model_resnet50V2_50classes_ade20k.h5`.

```text
SHA-256: B042699DD5D5AC5BCB367A457081F7385A2EAEBF3B5BB5D5662BE2FFC5360562

RGB image (224 x 224 x 3)
-> ResNet50V2 backbone
-> AveragePooling2D (7 x 7)
-> Flatten (2,048)
-> Dense (64, ReLU)
-> Dropout (0.5)
-> Dense (50, softmax)
```

Inference converts images to RGB, resizes them to 224 x 224, casts to `float32`, and divides pixels by 255. This preserves the supplied Stage 1 training pipeline; do not substitute the standard ResNetV2 `[-1, 1]` preprocessing.

`model_config.json` is the source of truth for checkpoint and calibration paths/hashes. `conceptlens_model.py` validates and caches the model. The original 10-class checkpoint and calibration are preserved in `backups/10-class/` but are not loaded by the application.

## Setup and application

The integration was verified with Python 3.11.9 and TensorFlow 2.13.0 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run Home.py
```

Open **Explore ConceptLens**, select a gallery image or upload up to three JPEG/PNG files, and start processing. Session state retains completed results across download-button reruns.

## Class mapping

`class_labels_50.json` stores the exact output-index mapping. It comes from the Stage 1 training log's `ImageDataGenerator.class_indices` dictionary and is independently corroborated by all 50 directory prefixes in the supplied 1,921-row activation CSV. Index 0 is `airport_terminal`, index 24 is `highway`, and index 49 is `youth_hostel`.

## Grad-CAM

Grad-CAM targets `post_relu`, verified at layer index 189 with shape `(None, 7, 7, 2048)`. It calculates gradients for the predicted 50-class score, applies channel weighting and ReLU/max normalization, then resizes and overlays the heatmap on the source image. This explains classification evidence; it is not object detection.

## 64-neuron scene associations

The new checkpoint's Dense(64) weights differ from the original 10-class model. The old `activation_cutoffs_ensemble.xlsx`, object-level ontology labels, hard-coded maxima, and error-rate percentages therefore cannot be reused.

The available 1,921 labeled activation rows support a new, narrower analysis:

- `activation_reference_50class.csv` stores per-neuron observed maxima.
- `activation_scene_associations_50class.csv` stores descriptive neuron-to-scene associations.
- `positive_images_50class.csv` and `negative_images_50class.csv` regenerate the original Stage 1 example-selection rules.

`tools/build_activation_calibration.py` calculates a one-vs-rest activation ROC AUC for every neuron and scene class with at least 10 reference images. It records the strongest and second-strongest scenes, AUC margin, support, and evidence tier for batch reports and research inspection. The interactive page does not present these statistical associations as the legacy object concepts. The final reference contains:

- 14 high-evidence neurons;
- 20 moderate-evidence neurons;
- 27 ambiguous neurons; and
- 3 inactive neurons (27, 36, and 60).

High evidence requires at least 20 primary-scene examples, AUC at least 0.85, and an AUC lead of at least 0.05 over the second scene. Moderate evidence requires AUC at least 0.80 and a lead of at least 0.03. All other active neurons are marked ambiguous. These are conservative project rules, not universal neuroscience or interpretability standards.

Positive examples have activation at least 80% of the neuron's observed maximum. Negative examples have nonzero activation at most 20% of that maximum, matching the Stage 1 code.

Rebuild the artifacts with:

```powershell
python tools/build_activation_calibration.py `
  --activation-csv "..\Model file for 50 classes\preds_of_64Neurons_denseLayer_1370Images_retest2023June (11).csv" `
  --labels class_labels_50.json `
  --output-dir .
```

Important: these are descriptive scene associations from the supplied reference, not probabilities, causal neuron identities, object-level ontology concepts, or error-rate percentages. Restoring object-level concepts still requires the missing ontology/template, learner output, and independent validation.

## Gaussian-noise batch experiments

```powershell
python batch_process.py --input-dir input_images --output-dir batch_results --seed 42
```

The preserved levels are `0, 10, 20, 30, 40, 50, 80`, with:

```text
sigma = (noise_percent / 100) x 40
```

The deterministic protocol, clipping/rounding, zero-level source inference, manifests, content hashes, per-condition errors, and incremental processing are preserved. CSV rows include the source path/hash and attempt ID. Failed conditions are retried without overwriting retained artifacts; ordinary repeated runs still skip completed conditions without adding CSV rows. Outputs include:

- `csv/predictions.csv`: predicted class and all 50 probabilities;
- `csv/activations.csv`: all 64 activations, normalization, scene associations, AUCs, and evidence tiers;
- `csv/experiment_summary.csv`: status and artifact references;
- activation JSON/plots and Grad-CAM overlays/plots; and
- `processing_manifest.json`: model/calibration fingerprint and per-level state.

The runner rejects legacy/mismatched manifests and CSV schemas rather than mixing old and new results. Use a new output directory when migrating. It preserves the legacy simple artifact name when available and adds a source/attempt suffix when a retained file already occupies that name.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite verifies the checkpoint/hash/shapes, 50 labels, softmax probabilities, Dense(64) extraction, association calibration, regenerated positive/negative lists, Grad-CAM, the noise protocol, all seven batch levels, pipeline-fingerprinted manifests, incremental skipping, retained-artifact failure retry, legacy-manifest rejection, and both Streamlit pages.

See `MODEL_ARCHITECTURE_COMPARISON.md` and `MODEL_MIGRATION_REPORT.md` for the detailed migration record.

## Known limitations

- Scene associations are descriptive statistics from the 1,921 supplied reference images.
- Object-level ontology concepts and error-rate labels remain unavailable without the missing ontology calibration artifacts.
- The complete training corpus was not supplied, so model accuracy/generalization was not re-evaluated.

## Security note

Credential-like patterns were detected in the Stage 1 job script during a count-only scan. No credential value was printed, copied, documented, or committed. Revoke and rotate any exposed GitHub token before sharing the Stage 1 project.
