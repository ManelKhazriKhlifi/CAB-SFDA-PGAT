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
├── requirements.txt
├── CAB_SFDA_PGAT.ipynb
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
## Datasets

All datasets used in this study are publicly available. Please download them from their official sources.

### Cross-Sensor Benchmark I: NWPU-RESISC45 ↔ NaSC-TG2

| Dataset | Description | Official Link |
| :--- | :--- | :--- |
| **NWPU-RESISC45** | 45 scene classes, 700 images per class, 256×256 pixels, 0.2–30m resolution. Created by Northwestern Polytechnical University. | https://gcheng-nwpu.github.io |
| **NaSC-TG2** | 10 scene classes, hyperspectral imagery from Tiangong-2, 0.40–1.04 µm spectral range. Created by Chinese Academy of Sciences. | https://captain-whu.github.io/BED4RS/ |

### Cross-Sensor Benchmark II: WHU-RS19 ↔ EuroSAT

| Dataset | Description | Official Link |
| :--- | :--- | :--- |
| **WHU-RS19** | 19 land-use classes, 50 images per class, 600×600 pixels, 0.5m resolution. Released by Wuhan University in 2012. | https://captain-whu.github.io/BED4RS/ |
| **EuroSAT** | 10 land-use classes, 2,000–3,000 images per class, 64×64 pixels, 13 spectral bands from Sentinel-2. | https://github.com/phelber/eurosat |

### Cross-Scene Benchmark

| Dataset | Description | Official Link |
| :--- | :--- | :--- |
| **AID** | 30 scene classes, 220–420 images per class, 600×600 pixels, 0.5–8m resolution. Collected from Google Earth. | https://captain-whu.github.io/AID/ |
| **CLRS** | 25 scene classes, 600 images per class, 256×256 pixels, 0.26–8.85m resolution. Collected from Google Earth, Bing Maps, and Tianditu. | https://github.com/lehaifeng/CLRS |
| **MLRSNet** | 46 scene categories, 1,500–3,000 images per category, 256×256 pixels, 0.1–10m resolution. | https://github.com/cugbrs/MLRSNet |
| **RSSCN7** | 7 scene classes (grass, forest, farmland, parking, residential, industrial, river/lake), 400 images per class, 400×400 pixels. | https://github.com/palewithout/RSSCN7 |

### Citations
```bibtex
@misc{khelifi2026cabsfda,
  title={{CAB-SFDA}: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification},
  author={Khelifi, Manel Khazri and Ammar, Adel and Boulila, Wadii and Farah, Imed Riadh},
  year={2026},
  howpublished={\url{https://github.com/ManelKhazriKhlifi/CAB-SFDA-PGAT}},
  note={GitHub repository}
}
```
```bibtex
@article{khelifi2026cabsfda,
  title={CAB-SFDA: Class-Aware Balanced Source-Free Domain Adaptation for Remote Sensing Scene Classification},
  author={Khelifi, Manel Khazri and Ammar, Adel and Boulila, Wadii and Farah, Imed Riadh},
  journal={},
  year={2026}
}
```
