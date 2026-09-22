# CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification

Official implementation of the paper:

**CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification**

Manel Khazri Khelifi, Adel Ammar, Wadii Boulila, Imed Riadh Farah

---

## Overview

CAB-SFDA is a source-free domain adaptation (SFDA) framework for remote sensing scene classification. It adapts a pretrained source model to an unlabeled target domain **without accessing source data** during adaptation.

## Key Components

| Component | Description |
| :--- | :--- |
| **Frozen Classifier** | Anchors source decision boundaries, preventing catastrophic forgetting |
| **Class-Aware Momentum Prototype Memory** | Stable semantic anchors with inverse-square-root class-balanced weighting |
| **Prototype-Guided Class-Adaptive Threshold (PGAT)** | Per-class adaptive threshold with a one-step causal lag |
| **Unified Target-Only Objective** | Information maximization + consistency + class-balanced pseudo-labeling + prototype attraction |

## Repository Structure
```
CAB-SFDA-PGAT/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── 3_CAB_SFDA_PGAT_v2_AID_CLRS.ipynb
└── data/
    └── README.md
```
## Requirements

```bash
pip install -r requirements.txt
```
```
data/
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
└── target_clrs_split/
    ├── train/
    │   └── ... (same classes)
    └── test/
        └── ... (same classes)
```
```bibtex
@article{khelifi2026cabsfda,
  title={CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification},
  author={Khelifi, Manel Khazri and Ammar, Adel and Boulila, Wadii and Farah, Imed Riadh},
  journal={},
  year={2026}
}
```
