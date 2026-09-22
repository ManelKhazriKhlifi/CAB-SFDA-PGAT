# Data Preparation

This folder is a placeholder for the datasets used in CAB-SFDA. Please download the datasets from their official sources and organize them as described below.

## Directory Structure

For each transfer task, create a source and a target folder with an **80/20 train/test split**. Each class must have its own subfolder.

Example for **AID → CLRS**:

```text
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
│       ├── Farmland/
│       ├── Forest/
│       ├── Industrial/
│       ├── Meadow/
│       ├── Parking/
│       ├── Residential/
│       └── River/
└── target_clrs_split/
    ├── train/
    │   ├── Farmland/
    │   ├── Forest/
    │   ├── Industrial/
    │   ├── Meadow/
    │   ├── Parking/
    │   ├── Residential/
    │   └── River/
    └── test/
        ├── Farmland/
        ├── Forest/
        ├── Industrial/
        ├── Meadow/
        ├── Parking/
        ├── Residential/
        └── River/
