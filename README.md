# CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification

Official implementation of the paper:

**CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification**

Manel Khazri Khelifi, Adel Ammar, Wadii Boulila, Imed Riadh Farah

---

## Overview

CAB-SFDA adapts a source-trained model to an unlabeled target domain **without access to source data**. It combines a frozen classifier, a class-aware momentum prototype memory with inverse-square-root class weighting, and a Prototype-Guided class-Adaptive Threshold (PGAT) for pseudo-label selection.

| Component | Description | Code |
| :--- | :--- | :--- |
| **Frozen Classifier** | Anchors source decision boundaries, preventing catastrophic forgetting | `adapt.py`, `config.py` |
| **Class-Aware Momentum Prototype Memory** | Stable semantic anchors with inverse-square-root class-balanced weighting | `memory.py` |
| **Prototype-Guided Class-Adaptive Threshold (PGAT)** | Per-class adaptive threshold with a one-step causal lag | `threshold.py` |
| **Unified Target-Only Objective** | Information maximization + consistency + class-balanced pseudo-labeling + prototype attraction | `losses.py`, `adapt.py` |

---

## Evaluation protocol

| Setting | Value |
| :--- | :--- |
| Source pretraining | 50 epochs, Adam, lr 1e-3, batch 128 |
| Source-free adaptation | 15 epochs, Adam, lr 2e-4, batch 128|
| Frozen during adaptation | classifier **and** all BatchNorm statistics (`CFG.freeze_bn = True`) |
| Checkpoint | **final epoch only**|
| Target labels | used **once**, after adaptation, for the final evaluation |
| Seeds | 0, 1, 2 (results reported as mean ± std) |
| Backbone | ImageNet-pretrained ResNet-50, 128-d projection head, 128×128 input |

All settings live in [`config.py`](config.py).

---

## Repository layout

```
CAB-SFDA-PGAT/
├── CAB_SFDA_PGAT.py          # end-to-end driver (one task, one seed)
├── config.py                 # tasks, seeds, hyper-parameters, protocol
├── model.py                  # ResNet backbone + projection head + classifier
├── dataloaders.py            # datasets, weak / strong augmentations
├── losses.py                 # information maximization, entropy, temperature-scaled KL
├── memory.py                 # class-aware momentum prototype memory
├── threshold.py              # PGAT (per-class adaptive threshold)
├── train_source.py           # source pretraining (final-epoch checkpoint)
├── adapt.py                  # CAB-SFDA source-free adaptation loop
├── evaluate.py               # single post-adaptation evaluation
├── per_class_metrics.py      # confusion matrix + per-class figure
├── diagnostics_pgat.py       # PGAT threshold trajectories, feature extraction
├── run_all.sh                # all tasks × all seeds
│
├── run_baseline.py           # runs one SFDA baseline
├── baselines/                # SHOT, NRC, CoWA, MBNS, G-SFDA, AaD, DaC, SMNA (+ shared helpers)
│
├── ablation_losses.py        # loss-wise ablation
├── ablation_mechanisms.py    # mechanism-wise ablation (A–E)
├── ablation_backbone.py      # ResNet-18 / 50 / 101
├── ablation_imbalance.py     # controlled imbalance on the MLRSN target
│
├── collect_results.py        # gathers runs/ into Results_PGAT/{TASK}/seed_N/
├── aggregated_results.py     # OA / BAcc / macro-F1, mean ± std over seeds
├── aggregate_results.py      # + per-class metrics and mean confusion matrices
├── per_class_tables.py       # per-class recall / precision / F1 tables
├── pgat_diagnostics_table.py # PGAT final-state diagnostics tables
├── plot_mean_cm.py           # seed-averaged confusion matrices
├── feature_metrics.py        # t-SNE figures + silhouette / MMD / centroid metrics
├── analysis/
│   └── statistical_tests.py  # paired t-tests vs. baselines
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Data

All datasets are public. Download them from their official sources and keep only the classes shared by each source/target pair.

### Cross-scene benchmark (7 shared classes)

Shared classes: Farmland, Forest, Industrial, Meadow, Parking, Residential, River.

| Dataset | Description | Official link |
| :--- | :--- | :--- |
| **AID** | 30 scene classes, 220–420 images per class, 600×600 pixels, 0.5–8 m resolution. Collected from Google Earth. | https://captain-whu.github.io/AID/ |
| **CLRS** | 25 scene classes, 600 images per class, 256×256 pixels, 0.26–8.85 m resolution. Collected from Google Earth, Bing Maps and Tianditu. | https://github.com/lehaifeng/CLRS |
| **MLRSNet** | 46 scene categories, 1,500–3,000 images per category, 256×256 pixels, 0.1–10 m resolution. | https://github.com/cugbrs/MLRSNet |
| **RSSCN7** | 7 scene classes (grass, forest, farmland, parking, residential, industrial, river/lake), 400 images per class, 400×400 pixels. | https://github.com/palewithout/RSSCN7 |

### Cross-sensor benchmark I: NWPU-RESISC45 ↔ NaSC-TG2 (10 shared classes)

Shared classes: Beach, Circular Farmland, Cloud, Desert, Forest, Mountain, Rectangular Farmland, Residential, River, Snowberg.

| Dataset | Description | Official link |
| :--- | :--- | :--- |
| **NWPU-RESISC45** | 45 scene classes, 700 images per class, 256×256 pixels, 0.2–30 m resolution. Created by Northwestern Polytechnical University. | https://gcheng-nwpu.github.io |
| **NaSC-TG2** | 10 scene classes from the Tiangong-2 space station. This work uses the RGB version (NaSC-TG2-RGB, 128×128 pixels). | https://captain-whu.github.io/BED4RS/ |

### Cross-sensor benchmark II: WHU-RS19 ↔ EuroSAT (6 shared classes)

Shared classes: Farmland, Forest, Industry, Meadow, Residential, River.

| Dataset | Description | Official link |
| :--- | :--- | :--- |
| **WHU-RS19** | 19 land-use classes, ~50 images per class, 600×600 pixels, 0.5 m resolution. Released by Wuhan University in 2012. | https://captain-whu.github.io/BED4RS/ |
| **EuroSAT** | 10 land-use classes, 2,000–3,000 images per class, Sentinel-2. This work uses the RGB version (64×64 pixels). | https://github.com/phelber/eurosat |

### Expected layout

`ImageFolder` format, with identical class folders in every split:

```
$DATA_ROOT/data/
├── source_aid_split/
│   ├── train/
│   │   ├── Farmland/
│   │   ├── Forest/
│   │   ├── Industrial/
│   │   ├── Meadow/
│   │   ├── Parking/
│   │   ├── Residential/
│   │   └── River/
│   └── test/
│       └── ... (same classes)
├── target_clrs_split/{train,test}/<class>/*.jpg
└── ...    # source_<name>_split / target_<name>_split for every task
```

Folder names are `source_<dataset>_split` and `target_<dataset>_split`, with the lower-case names used in `config.py` (`aid`, `clrs`, `mlrsn`, `rsscn7`, `nwpu`, `nasc`, `eurosat`, `whu`).

---

## Tasks

| Cross-scene | | | | Cross-sensor |
| :--- | :--- | :--- | :--- | :--- |
| A2C, A2M, A2R | C2A, C2M, C2R | M2A, M2C, M2R | R2A, R2C, R2M | N2G, G2N, E2W, W2E |

A = AID, C = CLRS, M = MLRSNet, R = RSSCN7, N = NWPU-RESISC45, G = NaSC-TG2, E = EuroSAT, W = WHU-RS19.

---

## Running CAB-SFDA

All commands are run from the repository root. The task, seed and data path are read from environment variables.

**Linux / macOS**

```bash
export DATA_ROOT=/path/to/datasets
TASK=A2C SEED=0 python CAB_SFDA_PGAT.py      # one task, one seed
bash run_all.sh                              # all 16 tasks × seeds 0,1,2
bash run_all.sh A2C A2M                      # selected tasks × seeds 0,1,2
```

**Windows (cmd)**

```bat
set DATA_ROOT=D:\path\to\datasets
set TASK=A2C
set SEED=0
python CAB_SFDA_PGAT.py
```

The source model is trained automatically on the first run of a task/seed, then reused by CAB-SFDA, the baselines and the ablations.



---

## Baselines

```bash
TASK=A2C SEED=0 python run_baseline.py --method shot
# methods: shot, nrc, cowa, mbns, gsfda, aad, dac, smna
```

The baselines are **re-implemented** inside this code base under the same protocol as CAB-SFDA: same source model, backbone, epochs, learning rate and final-epoch evaluation. They follow the core idea of each method but are simplified compared with the official implementations.

---

## Ablation studies

| Study | Script | Notes |
| :--- | :--- | :--- |
| Loss-wise (w/o L_pl, L_im, L_proto, L_cons) | `ablation_losses.py` | |
| Mechanism-wise (A–E) | `ablation_mechanisms.py` | variants below |
| Backbone (ResNet-18 / 50 / 101) | `ablation_backbone.py` | trains one source model per backbone |
| Imbalance sensitivity (ratios 1.0–5.0) | `ablation_imbalance.py` | requires an MLRSN-target task: `TASK=A2M`, `C2M` or `R2M` |
| Hyper-parameter sensitivity (λ_pl, λ_cons, γ, η) | `config.py` | edit the values, then run `CAB_SFDA_PGAT.py` |

```bash
TASK=A2M SEED=0 python ablation_mechanisms.py
```

Mechanism-wise variants (rows marked *ours* are the full method, identical to the main results):

| Group | Variants |
| :--- | :--- |
| (A) Class balancing | unweighted · inverse frequency · **inverse square root** *(ours)* |
| (B) Prototype memory | none · static (epoch 1 only) · batch (no EMA) · EMA updated from **all** predictions · **class-wise EMA from PGAT-accepted samples** *(ours)* |
| (C) Threshold | fixed 0.90 · fixed 0.80 · single class-agnostic adaptive threshold · no causal lag · no prototype warm-up · **full PGAT** *(ours)* |
| (D) Loss refinement | one-way KL · entropy-only IM (no diversity term) · **symmetric KL + IM** *(ours)* |
| (E) Classifier / BN | trainable classifier · frozen classifier + BN updated · **frozen classifier + frozen BN** *(ours)* |

---

## Results: aggregation, tables and figures

```bash
RESULTS_DIR=runs python aggregated_results.py --all --method cab_sfda   # read runs/{TASK}/seed_N/cab_sfda/
RESULTS_DIR=runs python aggregated_results.py --all --method shot       # any baseline
python analysis/statistical_tests.py                                    # paired t-tests, all 16 tasks
```

The table and figure scripts read one folder per seed. Gather the outputs first:

```bash
python collect_results.py                        # runs/ -> Results_PGAT/{TASK}/seed_N/
python aggregated_results.py --all               # mean ± std per task group (cross-scene, NWPU↔NaSC, EuroSAT↔WHU)
python per_class_tables.py                       # per-class recall / precision / F1 
python pgat_diagnostics_table.py                 # PGAT final-state diagnostics 
python plot_mean_cm.py --all                     # seed-averaged, row-normalized confusion matrices
python feature_metrics.py --all --tsne-summary   # t-SNE figures + feature-alignment metrics
```

Set `RESULTS_DIR` to read from a folder other than `./Results_PGAT`.

**t-SNE.** For each task, the figures show the representative seed (the seed whose OA is closest to the three-seed mean). The three-seed mean ± std of the matching metric (MMD for domain alignment, silhouette for category alignment). Both domains are embedded by the same network: the source model before adaptation, the adapted model after.

---

## Citation

If you use this code, please cite:

```bibtex
@article{khelifi2026cabsfda,
  title   = {{CAB-SFDA}: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification},
  author  = {Khelifi, Manel Khazri and Ammar, Adel and Boulila, Wadii and Farah, Imed Riadh},
  journal = {},
  year    = {2026}
}
```
