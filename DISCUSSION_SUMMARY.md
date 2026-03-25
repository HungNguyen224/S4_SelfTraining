# DAPCN-SSL: Semi-Supervised Satellite Image Segmentation

## Technical Discussion Summary

---

## 1. DAFormer-DAPCN Framework Overview

The DAFormer-DAPCN codebase implements a domain-adaptive semantic segmentation framework that extends the original DAFormer (Hoyer et al., CVPR 2022) with boundary-aware supervision and prototype clustering. The architecture comprises a MiT-B5 encoder with a SepASPP decoder, operating under a self-training paradigm with an Exponential Moving Average (EMA) teacher.

### 1.1 Key Modifications from Conventional DAFormer

The DAPCN extension introduces three principal components beyond the standard DAFormer self-training pipeline:

**AffinityBoundaryLoss.** A pairwise relational loss computed over 4-connected pixel neighbours. Rather than detecting boundaries via gradient operators (Sobel, Laplacian), the affinity approach computes temperature-scaled cosine similarity between adjacent feature vectors and applies binary cross-entropy against a ground-truth affinity map derived from label discontinuities. This formulation is orientation-invariant, making it particularly suitable for satellite imagery where land-cover boundaries have arbitrary orientations.

**DynamicAnchorModule.** A dataset-level persistent prototype bank implemented as `nn.Parameter`. The module performs differentiable EM refinement: an E-step (softmax assignment of features to prototypes) followed by an M-step (weighted mean update of prototype positions). A quality gate, implemented as an MLP with sigmoid activation, filters prototypes below a minimum quality threshold. Optional EMA smoothing stabilises prototype evolution across iterations.

**DAPGLoss.** A composite loss comprising three terms: L_intra (compactness, minimising within-cluster distance), L_inter (separation with margin, maximising between-cluster distance), and L_quality (regularisation encouraging the quality gate to remain active).

### 1.2 Addressing Domain Adaptation Challenges

The modifications address three challenges specific to satellite image segmentation:

1. **No repeatable shape.** Unlike natural images where objects (cars, buildings) have consistent geometric priors, land-cover categories (vegetation, water bodies) exhibit irregular, amorphous boundaries. The AffinityBoundaryLoss captures boundary structure through pairwise feature relationships rather than relying on shape templates, making it effective for arbitrarily shaped regions.

2. **Multiple land categories per image.** Satellite images typically contain 5-7 co-occurring land-cover classes, unlike scene-centric natural images dominated by a single category. The DynamicAnchorModule with `max_groups=96` maintains multiple sub-prototypes per class to capture intra-class diversity (e.g., different vegetation types, varying road surfaces), while the DAPGLoss ensures inter-class separation in the embedding space.

3. **Data bias with few labels.** Rare classes (barren land, water bodies) occupy disproportionately few pixels. Rare Class Sampling (RCS) applies temperature-scaled softmax over inverse class frequencies to oversample crops containing minority classes, directly mitigating the long-tail distribution problem.

---

## 2. Codebase Refactoring

The refactored UDA codebase removed several components inappropriate for satellite imagery:

- **ImageNet Feature Distance** — removed entirely, as satellite spectral characteristics differ fundamentally from ImageNet's natural-image statistics.
- **Legacy BAPCN parameters** — `contrastive_lambda`, `contrastive_temp`, `prototype_ema`, `proto_feature_dim`, `prototype_init_strategy` were pruned.
- **DACS baseline** — `dacs.py` and associated configs removed.
- **Debug utilities** — `visualization.py`, `print_grad_magnitude` removed.

The affinity boundary mode was set as default (`boundary_loss_mode='affinity'`), and `pseudo_weight_ignore_top/bottom` was set to 0 (satellite imagery is overhead with no rectification artefacts).

---

## 3. UDA Training Pipeline

The UDA variant (`DAPCN` in `mmseg/models/uda/dapcn.py`) executes a six-step training loop per iteration:

| Step | Operation | Loss |
|------|-----------|------|
| 1 | Supervised forward on source images | Cross-entropy with ground truth |
| 2 | DAPCN losses on source | Boundary (affinity) + prototype (DAPG) |
| 3 | EMA teacher generates pseudo-labels on target | — |
| 4 | ClassMix: source patches pasted onto target | — |
| 5 | Supervised forward on mixed images | Cross-entropy with mixed labels |
| 6 | DAPCN losses on mixed images | Boundary + prototype (pseudo-weighted) |

Both the student and EMA teacher share the DAFormer architecture. The teacher's weights are updated via exponential moving average: `θ_teacher ← α · θ_teacher + (1 − α) · θ_student`, with `α = 0.999`.

---

## 4. Semi-Supervised Learning (SSL) Version

### 4.1 Relationship to UDA

The SSL module (`DAPCN_SSL` in `mmseg/models/uda/dapcn_ssl.py`) is derived from the UDA module with three targeted modifications, while preserving the entire self-training architecture:

**Semantic renaming: `is_source` → `is_labeled`.** The distinction shifts from cross-domain (source/target) to within-domain (labeled/unlabeled). The training logic is identical — when `is_labeled=True`, ground-truth labels are used; when `is_labeled=False`, pseudo-label confidence weights are applied.

**Pseudo-label warmup.** A linear ramp-up of pseudo-label weight over the first N iterations (default: 1000):

```python
def _get_pseudo_weight_scale(self):
    if self.pseudo_label_warmup_iters <= 0:
        return 1.0
    return min(1.0, self.local_iter / self.pseudo_label_warmup_iters)
```

This prevents confirmation bias during early training when the teacher is unreliable due to limited labeled supervision. The UDA version does not require warmup because the fully-labeled source domain provides sufficient signal from iteration 0.

**SSLDataset replaces UDADataset.** Both datasets return identical dictionary keys (`img`, `img_metas`, `gt_semantic_seg`, `target_img`, `target_img_metas`), enabling the training loop to remain structurally unchanged. The SSLDataset partitions a single domain via a split file rather than pairing two separate domains.

### 4.2 SSL Training Pipeline

| Step | Operation | Loss |
|------|-----------|------|
| 1 | Supervised forward on labeled images | Cross-entropy with ground truth |
| 2 | DAPCN losses on labeled images | Boundary (affinity) + prototype (DAPG) |
| 3 | EMA teacher generates pseudo-labels on unlabeled | — |
| 4 | ClassMix: labeled patches pasted onto unlabeled | — |
| 5 | Supervised forward on mixed images | Cross-entropy with mixed labels |
| 6 | DAPCN losses on mixed images | Boundary + prototype (pseudo-weighted, warmup-scaled) |

### 4.3 Dataset Preparation

**Directory structure:**

```
data/satellite/
├── images/
│   ├── train/          # All training images
│   └── val/            # Validation images (fully labeled)
├── labels/
│   ├── train/          # Pixel-wise class ID maps (uint8, 0–N, 255=ignore)
│   └── val/
├── splits/
│   └── labeled.txt     # Basenames of labeled training images (no suffix)
├── sample_class_stats.json       # Per-image class pixel counts (for RCS)
└── samples_with_class.json       # Mapping: class → [file, pixel_count] (for RCS)
```

**Split mechanism.** The labeled/unlabeled partition is controlled entirely by `splits/labeled.txt`. The `CustomDataset` with `split='splits/labeled.txt'` loads only listed images as the labeled subset. The `CustomDataset` without a `split` field loads all images as the unlabeled pool. The unlabeled pool intentionally includes labeled images — they benefit from consistency regularisation via pseudo-labels and ClassMix.

**Data flow summary:**

| Split | Source | Labels Used? | Purpose |
|-------|--------|-------------|---------|
| Labeled (train) | Filtered by `labeled.txt` | Yes — ground truth | Supervised CE + DAPCN losses |
| Unlabeled (train) | All training images | No — pseudo-labels from EMA teacher | Self-training + ClassMix |
| Validation | `images/val/` | Yes — mIoU evaluation | Monitor generalisation every 4k iters |

**Rare Class Sampling (RCS) statistics generation:**

```python
import json, os, numpy as np
from PIL import Image

data_root = 'data/satellite'
ann_dir = os.path.join(data_root, 'labels/train')

sample_class_stats = []
samples_with_class = {}

for fname in sorted(os.listdir(ann_dir)):
    label = np.array(Image.open(os.path.join(ann_dir, fname)))
    entry = {'file': fname}
    for cls_id in np.unique(label):
        if cls_id == 255:
            continue
        n_pixels = int(np.sum(label == cls_id))
        entry[str(cls_id)] = n_pixels
        samples_with_class.setdefault(str(cls_id), []).append([fname, n_pixels])
    sample_class_stats.append(entry)

with open(os.path.join(data_root, 'sample_class_stats.json'), 'w') as f:
    json.dump(sample_class_stats, f)
with open(os.path.join(data_root, 'samples_with_class.json'), 'w') as f:
    json.dump(samples_with_class, f)
```

---

## 5. Ablation Study Design

A comprehensive ablation study was designed with 23 experiments across three groups, plus the baseline.

### 5.1 Group A — Component Ablation

Each experiment removes one component to measure its individual contribution.

| Config | Ablated Component | Research Question |
|--------|-------------------|-------------------|
| A1 | Boundary loss (λ_b = 0) | Does boundary-aware supervision improve edge quality? |
| A2 | Prototype loss (λ_p = 0) | Does prototype clustering help intra-class diversity? |
| A3 | Pseudo-label warmup | Is gradual warmup necessary for SSL stability? |
| A4 | ClassMix | Does class-based mixing regularise pseudo-labels? |
| A5 | Rare Class Sampling | Does RCS mitigate class imbalance? |
| A6 | All DAPCN + RCS | Total DAPCN contribution vs. vanilla self-training |

### 5.2 Group B — Hyperparameter Sensitivity

| Parameter | Values Tested | Baseline |
|-----------|--------------|----------|
| boundary_λ | 0.1, 0.3, **0.5**, 0.7, 1.0 | 0.5 |
| proto_λ | 0.05, **0.1**, 0.2, 0.5 | 0.1 |
| max_groups | 32, 64, **96**, 128 | 96 |
| warmup_iters | 500, **1000**, 2000, 5000 | 1000 |

### 5.3 Group C — Boundary Mode Comparison

| Mode | Characteristic |
|------|---------------|
| Sobel | Axis-aligned gradients, orientation-sensitive |
| Laplacian | Isotropic 2nd-order, noise-sensitive |
| Hybrid | Affinity + binary boundary combined |
| **Affinity (baseline)** | **Pairwise feature similarity, orientation-invariant** |

### 5.4 Running Ablation Experiments

```bash
# Full ablation suite
bash tools/run_ablation_ssl.sh all 0 0        # all experiments, GPU 0, seed 0

# Individual groups
bash tools/run_ablation_ssl.sh component 0 0   # Group A only
bash tools/run_ablation_ssl.sh hyperparam 0 0  # Group B only
bash tools/run_ablation_ssl.sh boundary 0 0    # Group C only

# Proto lambda sweep only
bash tools/run_ablation_proto_lambda.sh 0 0    # GPU 0, seed 0

# Collect results
python tools/collect_ablation_results.py
```

**Arguments:** The first integer specifies the CUDA GPU device index; the second specifies the random seed for reproducibility. Multiple seeds (e.g., 0, 42, 123) should be used to report mean ± standard deviation.

---

## 6. Project File Structure

### 6.1 Core Modules

| File | Description |
|------|-------------|
| `mmseg/models/uda/dapcn.py` | UDA training module (6-step self-training loop) |
| `mmseg/models/uda/dapcn_ssl.py` | SSL training module (warmup + same-domain adaptation) |
| `mmseg/models/uda/dynamic_anchor.py` | DynamicAnchorModule (learnable prototypes + EM refinement) |
| `mmseg/models/uda/uda_decorator.py` | Base class wrapping student model + EMA teacher |
| `mmseg/models/uda/utils/dapcn_utils.py` | Boundary extraction (Sobel/Laplacian) + boundary GT computation |

### 6.2 Loss Functions

| File | Description |
|------|-------------|
| `mmseg/models/losses/affinity_boundary_loss.py` | 4-neighbour pairwise affinity loss |
| `mmseg/models/losses/dapg_loss.py` | L_intra + L_inter + L_quality prototype loss |

### 6.3 Datasets

| File | Description |
|------|-------------|
| `mmseg/datasets/ssl_dataset.py` | SSLDataset (labeled/unlabeled from same domain, RCS support) |
| `mmseg/datasets/uda_dataset.py` | UDADataset (source/target cross-domain) |
| `mmseg/datasets/builder.py` | Dataset factory with SSLDataset and UDADataset handling |

### 6.4 Configurations

| File | Description |
|------|-------------|
| `configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py` | Full SSL training config |
| `configs/daformer/satellite_uda_dapcn_daformer_mitb5.py` | Full UDA training config |
| `configs/daformer/ablation_ssl/` | 22 ablation study configs (A1–A6, B1–B4, C1–C3) |
| `configs/_base_/ssl/dapcn_ssl.py` | Base SSL config (DAPCN_SSL type + defaults) |
| `configs/_base_/uda/dapcn.py` | Base UDA config |
| `configs/_base_/datasets/ssl_satellite_512x512.py` | SSL dataset config template |
| `configs/_base_/datasets/uda_satellite_512x512.py` | UDA dataset config template |

### 6.5 Tools

| File | Description |
|------|-------------|
| `tools/run_ablation_ssl.sh` | Full ablation runner (all groups) |
| `tools/run_ablation_proto_lambda.sh` | Proto-λ sweep runner |
| `tools/collect_ablation_results.py` | Result aggregation → console table + CSV |

---

## 7. Training Commands

```bash
# Semi-supervised learning
python tools/train.py configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py

# Unsupervised domain adaptation
python tools/train.py configs/daformer/satellite_uda_dapcn_daformer_mitb5.py
```

### 7.1 Key Hyperparameters (SSL Baseline)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `alpha` | 0.999 | EMA teacher momentum |
| `pseudo_threshold` | 0.968 | Confidence threshold for pseudo-labels |
| `boundary_lambda` | 0.5 | Boundary loss weight |
| `proto_lambda` | 0.1 | Prototype grouping loss weight |
| `boundary_loss_mode` | affinity | Orientation-invariant for satellite |
| `max_groups` | 96 | Prototype count (high for land-cover diversity) |
| `pseudo_label_warmup_iters` | 1000 | Linear ramp-up period |
| `max_iters` | 40,000 | Total training iterations |
| `lr` | 6e-5 | AdamW learning rate |
| `crop_size` | 512 × 512 | Training crop resolution |

---

---

## 8. Prototype-Based Pseudo-Label Correction

### 8.1 Motivation

The EMA teacher generates pseudo-labels based solely on its current predictive confidence. However, in semi-supervised settings with limited labeled data, the teacher's early predictions exhibit systematic biases — particularly for rare classes and boundary regions. The DynamicAnchorModule learns K class-agnostic prototypes that capture the dataset's feature distribution through differentiable EM refinement, independent of class labels. These prototypes provide a complementary signal that can regularise and correct the teacher's predictions.

### 8.2 Mathematical Formulation

The corrected probability for each pixel j is computed as:

```
p^c_j = sum_{i=1}^{K} f_theta(PT_i) * a_ij
```

where:

- `PT_i` is the i-th prototype learned by the DynamicAnchorModule (a vector in the decoder's feature space, dimension C)
- `f_theta(.)` is the decoder's `conv_seg` classifier — a 1×1 convolution mapping feature_dim → num_classes, followed by softmax — which projects a prototype into class probability space
- `a_ij` is the soft assignment (adjacency) between pixel j and prototype i, computed via temperature-scaled cosine similarity in the DynamicAnchorModule's E-step

The final blended probability combines teacher prediction with prototype correction:

```
p_final = (1 - alpha) * p_teacher + alpha * p^c
```

where `alpha` (default 0.5) controls the correction strength.

### 8.3 Why `conv_seg` Is the Correct Bridge

The `conv_seg` layer in `BaseDecodeHead` is a 1×1 convolution with weight shape `(num_classes, channels, 1, 1)`. Since the DynamicAnchorModule's prototypes live in the same C-dimensional feature space as the decoder's output features (both have `feature_dim = channels`), applying `conv_seg` to a prototype is mathematically equivalent to computing a linear classifier's prediction for that prototype. The softmax over the resulting logits yields a valid class probability distribution.

### 8.4 Implementation Details

The correction is implemented in `_correct_pseudo_labels()` with five internal steps:

```python
def _correct_pseudo_labels(self, target_img, target_img_metas, ema_softmax):
    # Step 1: Extract student features (no gradient needed)
    with torch.no_grad():
        student_feat = self.get_model().extract_feat(target_img)
    decode_head = self.get_model().decode_head
    feat = decode_head._transform_inputs(student_feat)
    if isinstance(feat, list):
        feat = feat[-1]
    B, C_feat, Hf, Wf = feat.shape

    # Step 2: DynamicAnchorModule → prototypes + soft assignments
    assign, proto, quality = self.dynamic_anchor(feat)

    # Step 3: Project prototypes through classifier f_theta
    proto_4d = proto.unsqueeze(-1).unsqueeze(-1)   # (K, C, 1, 1)
    proto_logits = decode_head.conv_seg(proto_4d)   # (K, num_classes, 1, 1)
    proto_probs = torch.softmax(
        proto_logits.squeeze(-1).squeeze(-1), dim=1)  # (K, num_classes)

    # Step 4: Corrected probability via matrix multiplication
    corrected_probs = torch.mm(assign, proto_probs)   # (N, num_classes)
    corrected_probs = corrected_probs.reshape(
        B, Hf, Wf, num_classes).permute(0, 3, 1, 2)

    # Upsample to match teacher resolution if needed
    _, _, H_tea, W_tea = ema_softmax.shape
    if (Hf, Wf) != (H_tea, W_tea):
        corrected_probs = F.interpolate(
            corrected_probs, size=(H_tea, W_tea),
            mode='bilinear', align_corners=False)

    # Step 5: Blend teacher + prototype correction
    alpha = self.proto_correction_alpha
    blended = (1 - alpha) * ema_softmax + alpha * corrected_probs
    return blended
```

### 8.5 Delayed Activation

Prototype correction is activated only after `proto_correction_start_iter` (default: 1000) iterations. During the initial phase, prototypes are still being shaped by the DAPGLoss via labeled supervision — enabling correction too early would inject noise from uninformed prototypes into the pseudo-labels, destabilising training.

The activation condition in `forward_train` (Step 3b):

```python
use_correction = (
    self.proto_correction
    and self.dynamic_anchor is not None
    and self.local_iter >= self.proto_correction_start_iter
)
if use_correction:
    ema_softmax = self._correct_pseudo_labels(
        target_img, target_img_metas, ema_softmax)
```

### 8.6 Downstream Impact

Because the correction modifies `ema_softmax` before `torch.max()` is applied, it simultaneously affects three downstream quantities:

1. **Hard pseudo-label** (`pseudo_label`): the argmax class assignment may change for pixels where prototype correction shifts the dominant class.
2. **Confidence weight** (`pseudo_weight`): the max probability changes, affecting which pixels exceed the confidence threshold τ=0.968.
3. **Mixed label/weight after ClassMix**: since ClassMix operates on the derived pseudo-labels and weights, the correction propagates through the entire mixed supervision pathway.

---

## 9. Supervised vs. Mixed Supervision Weights

### 9.1 Labeled Supervision (Step 1)

Labeled images use uniform weight 1.0 for all valid pixels. The `forward_train` call at Step 5 passes `seg_weight=None` to the decode head's `losses()` method, which treats all non-ignore pixels equally:

```python
clean_losses = self.get_model().forward_train(
    img, img_metas, gt_semantic_seg, return_feat=True)
```

### 9.2 Mixed Supervision (Step 5)

Mixed images use confidence-scaled, warmup-modulated, spatially-mixed weights. The pseudo-weight undergoes three transformations:

1. **Confidence ratio**: fraction of pixels exceeding threshold τ=0.968, broadcast to a uniform spatial map.
2. **Warmup scaling**: multiplied by `_get_pseudo_weight_scale()`, linearly ramping from 0 to 1 over the first N iterations.
3. **ClassMix spatial mixing**: `strong_transform()` applies the binary class mask, setting weight to 1.0 in labeled-patch regions and to the pseudo-weight in unlabeled-background regions.

```python
# Step 5 passes pseudo_weight as seg_weight:
model_output = self.get_model().forward_train(
    mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
```

This means labeled pixels within the mixed image receive weight 1.0 (from `gt_pixel_weight`), while unlabeled pixels receive the confidence-scaled pseudo-weight. The two populations are blended spatially by the ClassMix mask.

---

## 10. The `torch.max` vs. `torch.argmax` Distinction

In the pseudo-label derivation (line 546 of `dapcn_ssl.py`):

```python
pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
```

`torch.max(tensor, dim)` returns a named tuple `(values, indices)`, providing both the maximum probability value and its index (the predicted class) in a single operation. This is more efficient than calling `torch.argmax` separately, because:

- `pseudo_label` (indices) is used as the hard pseudo-label target for cross-entropy.
- `pseudo_prob` (values) is used to compute the confidence weight — pixels whose max probability exceeds τ=0.968 are considered reliable.

Both quantities are needed simultaneously, making `torch.max` the natural choice over `torch.argmax` (which returns only the indices and would require a separate `torch.gather` or indexing operation to retrieve the corresponding probabilities).

---

## 11. The `train_step` Method

### 11.1 Role in the Training Loop

The `train_step` method is the entry point that MMSeg's `IterBasedRunner` calls at every training iteration:

```python
def train_step(self, data_batch, optimizer, **kwargs):
    optimizer.zero_grad()
    log_vars = self(**data_batch)
    optimizer.step()

    log_vars.pop('loss', None)
    outputs = dict(
        log_vars=log_vars, num_samples=len(data_batch['img_metas']))
    return outputs
```

### 11.2 Line-by-Line Analysis

**`optimizer.zero_grad()`** — Clears all accumulated gradients from the previous iteration. Essential because PyTorch accumulates gradients by default.

**`log_vars = self(**data_batch)`** — The `self(...)` call triggers Python's `__call__` mechanism, routing through `nn.Module.forward()` to `forward_train()`. The `data_batch` dict is unpacked as keyword arguments — its keys (`img`, `img_metas`, `gt_semantic_seg`, `target_img`, `target_img_metas`) map exactly to the `forward_train` signature.

Inside `forward_train`, six backward passes accumulate gradients onto the same parameter set:

| Backward call | Source | retain_graph? | Reason |
|---------------|--------|---------------|--------|
| `clean_loss.backward(retain_graph=True)` | Step 1: supervised CE | Yes | Step 2 reuses `src_feat` |
| `src_dapcn_loss.backward()` | Step 2: labeled DAPCN | No | Last use of `src_feat` graph |
| *(no backward)* | Step 3: teacher inference | — | `torch.no_grad()` context |
| *(no backward)* | Step 4: ClassMix augmentation | — | Pure tensor manipulation |
| `target_loss.backward(retain_graph=True)` | Step 5: mixed CE | Yes | Step 6 reuses `tgt_feat` |
| `tgt_dapcn_loss.backward()` | Step 6: mixed DAPCN | No | Last use of `tgt_feat` graph |

**`optimizer.step()`** — Applies the accumulated gradient in a single AdamW update. The total gradient is:

```
∇θ = ∇L_CE^labeled + ∇L_DAPCN^labeled + ∇L_CE^mixed + ∇L_DAPCN^mixed
```

Note: the EMA teacher is not updated here — it was already updated at the start of `forward_train` via `_update_ema()`.

**`log_vars.pop('loss', None)`** — Removes the aggregated scalar loss key (inserted by `_parse_losses`) because the runner only needs individual loss components for TensorBoard/console logging.

**Return value** — The `outputs` dict contains `log_vars` (scalar loss values for logging) and `num_samples` (batch size for averaging by the runner).

### 11.3 Why `retain_graph=True` on Steps 1 and 5

Steps 1 and 5 both extract decoder features (`src_feat`, `tgt_feat`) that are reused in Steps 2 and 6 respectively. Without `retain_graph=True`, PyTorch would free the intermediate computation graph after the first `.backward()`, making the subsequent DAPCN loss backward pass impossible since it depends on the same feature tensors. The last backward in each pair does not need `retain_graph` because no further backward passes will use that graph.

### 11.4 Calling Context

The `IterBasedRunner` calls `train_step` in a loop:

```python
# Simplified IterBasedRunner.run()
for i, data_batch in enumerate(data_loader):
    outputs = model.train_step(data_batch, optimizer)
    # Log outputs, save checkpoints, run hooks...
```

The `data_batch` comes from `SSLDataset.__getitem__`, which pairs one labeled sample with one unlabeled sample and returns the five-key dict that `forward_train` expects.

---

## 12. Ablation Configs for Prototype Correction

Three additional ablation configs were created to evaluate the prototype-based pseudo-label correction:

| Config | Setting | Research Question |
|--------|---------|-------------------|
| A7_no_proto_correction | `proto_correction=False` | Does prototype correction improve pseudo-label quality? |
| B5_correction_alpha_02 | `proto_correction_alpha=0.2` | Is a lighter correction blending more stable? |
| B5_correction_alpha_08 | `proto_correction_alpha=0.8` | Does stronger prototype influence help or harm? |

These bring the total ablation count to 25 experiments (7 component + 14 hyperparameter + 3 boundary mode + 1 baseline).

### 12.1 Updated Configuration Parameters

The following parameters were added to both the base config (`configs/_base_/ssl/dapcn_ssl.py`) and the main training config:

```python
proto_correction=True,          # Enable prototype-based correction
proto_correction_alpha=0.5,     # Blending weight (0=pure teacher, 1=pure prototype)
proto_correction_start_iter=1000,  # Delay activation until prototypes stabilise
```

---

## 13. Updated Project File Structure

### 13.1 New Files (This Session)

| File | Description |
|------|-------------|
| `configs/daformer/ablation_ssl/A7_no_proto_correction.py` | Component ablation: disable correction |
| `configs/daformer/ablation_ssl/B5_correction_alpha_02.py` | Hyperparameter: α=0.2 |
| `configs/daformer/ablation_ssl/B5_correction_alpha_08.py` | Hyperparameter: α=0.8 |

### 13.2 Modified Files (This Session)

| File | Changes |
|------|---------|
| `mmseg/models/uda/dapcn_ssl.py` | Added `_correct_pseudo_labels()`, integration in `forward_train` Step 3b, new `__init__` parameters |
| `configs/_base_/ssl/dapcn_ssl.py` | Added `proto_correction`, `proto_correction_alpha`, `proto_correction_start_iter` |
| `configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py` | Added proto_correction parameters |

---

*Generated from technical discussion on DAFormer-DAPCN semi-supervised satellite segmentation framework. Updated with prototype-based pseudo-label correction, train_step analysis, and supervision weight clarifications.*
