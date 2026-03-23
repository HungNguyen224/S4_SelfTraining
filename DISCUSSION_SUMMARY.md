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

*Generated from technical discussion on DAFormer-DAPCN semi-supervised satellite segmentation framework.*
