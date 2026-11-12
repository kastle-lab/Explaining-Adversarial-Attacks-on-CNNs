# Pre-integration model comparison

This report records the read-only checkpoint inspection completed before the 50-class integration.

| Property | Original ConceptLens | Stage 1 candidate |
|---|---:|---:|
| SHA-256 | `33E5A76D96C7CCCB2D2F69FBE99594177438EB48B09A282136A4E95E6516C116` | `B042699DD5D5AC5BCB367A457081F7385A2EAEBF3B5BB5D5662BE2FFC5360562` |
| Input | `(None, 224, 224, 3)`, RGB/channels-last | Same |
| Input scaling | `pixel / 255.0` | Same |
| Backbone | ResNet50V2, ImageNet initialization, frozen | Same |
| Layer count | 195 | 195 |
| Spatial feature layer | `post_relu`, index 189, `(None, 7, 7, 2048)` | Same |
| Head | average pool 7x7, flatten, Dense(64/ReLU), Dropout(0.5) | Same |
| Hidden layer | `dense`, index 192, `(None, 64)` | Same shape/name, different learned weights |
| Output layer | `dense_1`, `(None, 10)`, softmax | `dense_1`, `(None, 50)`, softmax |
| Parameters | 23,696,586 | 23,699,186 |

The candidate loaded successfully under TensorFlow 2.13.0. A `(1, 224, 224, 3)` test tensor produced 50 finite, non-negative probabilities summing to `1.0000001192`.

The backbones compare exactly at sampled layers (`conv1_conv` and `post_bn`), while the Dense(64) kernel and bias do not. The Dense(64) kernel L2 difference is `27.653305`; the bias L2 difference is `0.150473`. The legacy spreadsheet and its hard-coded 10-class activation maxima therefore cannot be reused as semantic calibration.

The verified label order comes from the training log's recorded `class_indices` mapping and is independently corroborated by all 50 class prefixes in the supplied 1,921-row activation CSV. It is stored in `class_labels_50.json`.


