# ConceptLens 50-class model migration report

Date completed: 2026-11-01; acceptance reverified: 2026-11-02

## Outcome

ConceptLens now uses the verified 50-output Stage 1 ResNet50V2 checkpoint as its only active inference model. Predictions, labels, Grad-CAM, Streamlit rendering, and Gaussian-noise batch processing all use the centralized configuration and loader in `conceptlens_model.py`.

The original 10-class model and calibration workbook were preserved. Their object-level concept/error-rate analysis is disabled because it is demonstrably incompatible with the new Dense(64) weights. The supplied 1,921-image activation CSV now supports reference normalization plus evidence-qualified statistical scene associations.

## Project discovery

- Existing application: `C:\Users\shank\testing new resnet model\ConceptLens`
- Stage 1 project/artifacts: `C:\Users\shank\testing new resnet model\Model file for 50 classes`

Inspected application sources included `Home.py`, `pages/Explore_ConceptLens.py`, `demonstrator.py`, `batch_process.py`, `README.md`, `requirements.txt`, the old checkpoint, and both Excel workbooks. Inspected Stage 1 sources included `main.py`, `model.py`, `train.py`, `data_preparation.py`, `analyze_activations.py`, `config_generator.py`, `config.json`, `README.md`, the candidate checkpoint, training logs, and the supplied activation CSV.

The available Stage 1 copy did not contain the generated prediction CSV, `positive_images.csv`, `negative_images.csv`, generated neuron configuration files, ontology template, or training/validation image directories described by its README and hard-coded paths.

## Checkpoint selection and backup

The filename `model_resnet50V2_10classes_retest2023June.h5` was not trusted. Both files were loaded and inspected.

| Checkpoint | Outputs | SHA-256 |
|---|---:|---|
| Original ConceptLens | 10 | `33E5A76D96C7CCCB2D2F69FBE99594177438EB48B09A282136A4E95E6516C116` |
| Stage 1 candidate | 50 | `B042699DD5D5AC5BCB367A457081F7385A2EAEBF3B5BB5D5662BE2FFC5360562` |

The selected checkpoint was copied without modification to `models/model_resnet50V2_50classes_ade20k.h5` and its hash was rechecked. `model_config.json` requires that exact hash before loading.

Backups:

- `backups/10-class/model_resnet50V2_10classes_retest2023June.h5`
- `backups/10-class/activation_cutoffs_ensemble.xlsx`
- `backups/10-class/backup_manifest.json`

The workbook backup hash is `CCA801579E5833ADF9B048270653464BD07B5AE241FF76E5B00538FD374874CF`. The historical root copies were not deleted or overwritten.

## Architecture comparison

Both models contain 195 layers, accept `(None, 224, 224, 3)` channels-last RGB input, and use a frozen ResNet50V2 backbone followed by:

```text
AveragePooling2D(7 x 7) -> Flatten(2048) -> Dense(64, ReLU)
-> Dropout(0.5) -> Dense(classes, softmax)
```

| Property | Original | New |
|---|---:|---:|
| Parameter count | 23,696,586 | 23,699,186 |
| Hidden layer | `dense`, index 192, `(None, 64)` | same name/index/shape |
| Output layer | `dense_1`, `(None, 10)` | `dense_1`, `(None, 50)` |
| Grad-CAM layer | `post_relu`, index 189, `(None, 7, 7, 2048)` | same |

Sampled frozen backbone weights (`conv1_conv`, `post_bn`) were exactly equal. The Dense(64) weights were not: kernel L2 difference `27.653305`, bias L2 difference `0.150473`. Output-head weights are shape-incompatible (`64x10` versus `64x50`). See `MODEL_ARCHITECTURE_COMPARISON.md` for the pre-integration report.

## Preprocessing

The Stage 1 data pipeline used `ImageDataGenerator(rescale=1./255)` with target size `(224, 224)` and RGB/channels-last data. The integrated pipeline preserves RGB conversion, 224 x 224 resize, `float32`, and division by 255. It does not introduce the standard ResNetV2 `[-1, 1]` preprocessing because that was not used by the supplied training code.

## Verified class-label mapping

The training log recorded the exact `class_indices` dictionary. All 50 prefixes in the supplied activation CSV independently matched it. `class_labels_50.json` stores this order:

| Index | Class ID | Index | Class ID |
|---:|---|---:|---|
| 0 | `airport_terminal` | 25 | `home_office` |
| 1 | `alley` | 26 | `hotel_room` |
| 2 | `amusement_park` | 27 | `house` |
| 3 | `art_gallery` | 28 | `kitchen` |
| 4 | `art_studio` | 29 | `lighthouse` |
| 5 | `attic` | 30 | `living_room` |
| 6 | `bar` | 31 | `lobby` |
| 7 | `bathroom` | 32 | `mountain` |
| 8 | `beach` | 33 | `mountain_snowy` |
| 9 | `bedroom` | 34 | `nursery` |
| 10 | `bridge` | 35 | `office` |
| 11 | `building_facade` | 36 | `park` |
| 12 | `castle` | 37 | `parlor` |
| 13 | `childs_room` | 38 | `pasture` |
| 14 | `classroom` | 39 | `playroom` |
| 15 | `closet` | 40 | `reception` |
| 16 | `coast` | 41 | `river` |
| 17 | `cockpit` | 42 | `roundabout` |
| 18 | `conference_room` | 43 | `shoe_shop` |
| 19 | `corridor` | 44 | `skyscraper` |
| 20 | `creek` | 45 | `staircase` |
| 21 | `dining_room` | 46 | `street` |
| 22 | `dorm_room` | 47 | `waiting_room` |
| 23 | `game_room` | 48 | `window_seat` |
| 24 | `highway` | 49 | `youth_hostel` |

## Centralized inference

`conceptlens_model.py` now reads and validates configuration/labels, verifies the checkpoint hash, caches the model, centralizes preprocessing, selects explanation layers by name, and converts raw indices through the verified mapping. No application or batch component references the new checkpoint independently.

## Grad-CAM

`post_relu` remains a connected 4D spatial feature map at index 189. Grad-CAM now validates this layer by name/shape, computes the gradient of the predicted 50-class score, performs channel weighting and ReLU/max normalization, resizes the heatmap to the source image, and preserves the existing 60% image/40% `jet` overlay style.

The UI and batch outputs explicitly describe Grad-CAM as classification evidence rather than object detection.

## 64-neuron compatibility decision

The old workbook contains 185 complete mapping rows across 120 unique concepts and depends on old-model neuron semantics and hard-coded maxima. The new Dense(64) weights differ, and the supplied activation distribution also differs: new reference-zero neurons are 27, 36, and 60, whereas the legacy hard-coded vector has a different zero pattern.

Reusing the workbook would therefore display misleading labels/error rates. The workbook remains intact and backed up but is not loaded by the new pipeline.

`activation_reference_50class.csv` is derived from all 1,921 rows of the supplied activation CSV. For each image, all 64 raw activations are retained. Three zero-reference neurons are marked inactive.

`tools/build_activation_calibration.py` now regenerates `positive_images_50class.csv` and `negative_images_50class.csv` with the Stage 1 80%/20% maximum rules. It also calculates tie-aware one-vs-rest activation ROC AUC for every neuron and scene class with at least 10 reference images. `activation_scene_associations_50class.csv` records primary/secondary scenes, AUCs, margin, class support, and evidence tier. The resulting distribution is 14 high, 20 moderate, 27 ambiguous, and 3 inactive neurons. Only high/moderate associations are displayed.

High evidence requires primary class support >=20, AUC >=0.85, and primary-versus-secondary AUC margin >=0.05. Moderate evidence requires AUC >=0.80 and margin >=0.03. These thresholds are conservative project rules and are documented rather than presented as universal standards.

These labels are descriptive scene associations, not causal neuron identities or object-level ontology concepts. Restoring the original object concepts and error-rate percentages still requires the missing ontology/template, learner output, and independent validation.

## Streamlit changes

- Predictions preserve the original single-class/single-score presentation while using the verified 50-class mapping.
- Grad-CAM uses the same cached 50-class model and preprocessing.
- JPEG uploads are supported, up to three files, matching the original interface.
- The main page retains the legacy concept/error-rate section location but reports that its 10-class calibration is unavailable rather than displaying statistical scene associations as object concepts.
- Session-state behavior continues to prevent download reruns from repeating model work.
- Home-page text describes the 50-class prediction and Grad-CAM scope without adding a separate association dashboard.

## Batch-processing changes

The experiment retains levels `0, 10, 20, 30, 40, 50, 80`, `sigma = (noise_percent / 100) * 40`, deterministic condition seeds, pixel rounding/clipping, zero-level source inference, manifests, content hashes, per-condition error capture, and incremental skipping. Manifest schema 2 records the exact checkpoint hash, labels file, scene-association calibration hash, preprocessing fingerprint, and report-schema version; a legacy or mismatched manifest is rejected so incompatible results cannot be reused.

Every condition records its source path/hash and attempt ID. Every successful condition also records all 50 probabilities, all 64 activations with reference metadata, scene associations/AUCs/evidence tiers, activation/Grad-CAM artifacts, reproducibility metadata, status, and errors.

Because legacy concept columns are invalid for the new model, `concepts.csv`/concept artifacts were replaced by `activations.csv`/activation artifacts. Existing CSV headers are validated before append; an incompatible legacy output directory is rejected with instructions to use a new directory. This is the only intentional output-schema change.

The runner never replaces artifacts. It retains the legacy simple filename when unused; retries, explicit reruns, changed source versions, and same-stem collisions receive a unique source/attempt suffix. Failed conditions can therefore be retried while their previous diagnostic artifacts remain intact. Incremental runs still skip completed conditions without appending duplicate CSV rows.

## Verification performed

Environment: Python 3.11.9, TensorFlow 2.13.0, CPU, Windows.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Final result on the 2026-11-02 acceptance pass: **11 tests passed** in 25.421 seconds.

Verified assertions included checkpoint load/hash/shapes, all 50 labels, valid softmax values, 224 x 224 preprocessing, Dense(64) extraction, association calibration and regenerated example lists, Grad-CAM shape/normalization, exact noise mapping, all seven batch levels, 7 prediction rows, 448 activation rows with evidence fields, every artifact family, complete model/calibration/report-fingerprinted manifest state, source hash/path and attempt reporting, incremental skip without CSV growth, an intentional failed condition followed by a successful retained-artifact retry, legacy-manifest rejection without overwrite, and zero Streamlit test exceptions on both pages.

An independent five-image check produced finite, normalized predictions and source-sized Grad-CAM heatmaps for every supplied example: `bedroom.jpeg` -> index 9 `bedroom` (`0.75330305`), `highway.jpg` -> index 24 `highway` (`0.68564445`), `living_space.jpg` -> index 30 `living_room` (`0.90614802`), `road.jpeg` -> index 46 `street` (`0.87507480`), and `vehicle.jpeg` -> index 42 `roundabout` (`0.51642376`). Probability sums were within `0.00000024` of 1.0; all heatmaps were finite, in `[0, 1]`, and matched their source dimensions.

A live headless Streamlit server returned HTTP 200 and `ok` from `/_stcore/health`; the root page returned HTTP 200. Windows browser automation could not complete a file-upload click because its safety layer could not establish the current browser URL with sufficient confidence. The upload processing path itself is covered by direct real-image inference, Streamlit page rendering, and the full batch/end-to-end tests; a manual browser upload remains the only unautomated UI gesture.

## Files modified

- `.gitignore`
- `Home.py`
- `pages/Explore_ConceptLens.py`
- `demonstrator.py`
- `batch_process.py`
- `README.md`

## Files created

- `conceptlens_model.py`
- `model_config.json`
- `class_labels_50.json`
- `activation_reference_50class.csv`
- `activation_scene_associations_50class.csv`
- `positive_images_50class.csv`
- `negative_images_50class.csv`
- `tools/build_activation_calibration.py`
- `models/model_resnet50V2_50classes_ade20k.h5`
- `MODEL_ARCHITECTURE_COMPARISON.md`
- `MODEL_MIGRATION_REPORT.md`
- `tests/test_model_integration.py`
- `backups/10-class/model_resnet50V2_10classes_retest2023June.h5`
- `backups/10-class/activation_cutoffs_ensemble.xlsx`
- `backups/10-class/backup_manifest.json`

## Outstanding issues and limitations

1. Scene associations are descriptive statistics from the supplied reference, not object-level ontology concepts or causal neuron identities.
2. Object-level concepts/error-rate labels still require missing ontology/template, learner output, and independent validation.
3. The activation reference represents 1,921 supplied examples, not the unavailable complete training corpus.
4. Model accuracy was not re-evaluated without the original datasets.
5. A manual browser upload should be performed for final human visual acceptance because automated Windows browser upload was stopped by the browser-URL safety check.

## Security and Git safety

A count-only scan detected two credential-like patterns in the Stage 1 job script. Values were never printed, copied, documented, or committed. Revoke/rotate any exposed GitHub token and remove it from scripts, remotes, and history before sharing that project.

No Git push, commit, remote modification, history rewrite, or source/training-artifact deletion was performed.


