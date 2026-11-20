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

## Scene Associations and Non-target Activation Rates

Explore displays one bar per retained neuron that activates on the current image. The bar shows the percentage of held-out images labeled as other scenes whose activation exceeded the displayed cutoff. The count is the number of those images activated over the total number of other-scene images. The bars describe this held-out Stage 1 sample. Scene predictions, Grad-CAM, and the separate object-concept status are also available.

### Discovery and held-out rates

- Split each scene by sorting SHA-256 hashes of `conceptlens-scene-split-v1:<scene/filename>`. Assign the first quarter, rounded to the nearest whole image with halves rounded up, to the held-out split. Keep at least one image in each split when a scene has two or more images. The result is **1,435 discovery rows and 486 held-out rows**, covering 50 scene labels.
- Use discovery rows to compute all 64 maxima and rank each neuron's scenes by tie-aware one-vs-rest ROC AUC. Eligible scenes have at least 10 discovery images. Ties use ascending scene ID. High evidence requires at least 20 target images, AUC at least 0.85 and a lead of at least 0.05 over the runner-up. Moderate evidence requires AUC at least 0.80 and a lead of at least 0.03.
- Keep at most one mapping per neuron. It needs high or moderate evidence, a positive maximum, at least five discovery target images with positive activation, and at least 10 target and 30 other-scene images on the held-out split. A failed support check omits the mapping without replacing it with a runner-up. The rebuild retained **21 mappings**. Three neurons were inactive, 25 had ambiguous evidence, and 15 lacked held-out support.
- After fixing the mappings and maxima, count strict activations above 0%, 20%, 40%, and 60% of each discovery maximum. Save target and non-target activated counts, totals, and rates in percent. At runtime the highest strictly exceeded cutoff determines the row. An activation exactly at 20%, 40%, or 60% uses the preceding cutoff. An activation of zero produces no bar.
- The old Stage 1 example-selection rules use discovery rows and discovery maxima. Positive examples reach at least 80%; negative examples are nonzero and reach at most 20%.

| File | Contents |
| --- | --- |
| `scene_activation_split_50class.csv` | Filename, scene label, and split for each source row |
| `activation_reference_50class.csv` | Discovery maxima and sample counts |
| `activation_scene_associations_50class.csv` | Discovery rankings, sample support, and mapping status for all 64 neurons |
| `scene_activation_rates_50class.csv` | 84 held-out rows with both rates, numerators, and denominators |
| `positive_images_50class.csv` and `negative_images_50class.csv` | Discovery example lists |
| `model_config.json` | Source CSV SHA-256 and the hashes and paths of the generated CSVs |

Rebuild from the original Stage 1 activation CSV in the sibling model folder:

```powershell
python tools/build_activation_calibration.py `
  --activation-csv "..\Model file for 50 classes\preds_of_64Neurons_denseLayer_1370Images_retest2023June (11).csv" `
  --labels class_labels_50.json --output-dir . --model-config model_config.json
```

The split holds out activation rows for the scene-rate measurement. The supplied CSV and checkpoint do not establish an independent test of classifier performance.

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

Use a fresh batch output directory for the rebuilt mappings. The batch runner checks existing manifests and CSV schemas against the active calibration. It preserves the legacy simple artifact name when available and adds a source/attempt suffix when a retained file already occupies that name.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite verifies the checkpoint/hash/shapes, 50 labels, softmax probabilities, Dense(64) extraction, association calibration, regenerated positive/negative lists, Grad-CAM, the noise protocol, all seven batch levels, pipeline-fingerprinted manifests, incremental skipping, retained-artifact failure retry, legacy-manifest rejection, and both Streamlit pages.

See `MODEL_ARCHITECTURE_COMPARISON.md` and `MODEL_MIGRATION_REPORT.md` for the detailed migration record.

## Known limitations

- Scene associations and rates describe the discovery and held-out rows of the supplied Stage 1 activation CSV.
- Object-level ontology concepts and error-rate labels remain unavailable without the missing ontology calibration artifacts.
- The complete training corpus was not supplied, so model accuracy/generalization was not re-evaluated.

## Security note

Credential-like patterns were detected in the Stage 1 job script during a count-only scan. No credential value was printed, copied, documented, or committed. Revoke and rotate any exposed GitHub token before sharing the Stage 1 project.
 