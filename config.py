import os
import random
import numpy as np
import torch


TASK_CONFIGS = {
    # ---- cross-scene (7 classes) ----
    "A2C": {"source": "AID",    "target": "CLRS",    "num_classes": 7},
    "A2M": {"source": "AID",    "target": "MLRSN",   "num_classes": 7},
    "A2R": {"source": "AID",    "target": "RSSCN7",  "num_classes": 7},
    "C2A": {"source": "CLRS",   "target": "AID",     "num_classes": 7},
    "C2M": {"source": "CLRS",   "target": "MLRSN",   "num_classes": 7},
    "C2R": {"source": "CLRS",   "target": "RSSCN7",  "num_classes": 7},
    "M2A": {"source": "MLRSN",  "target": "AID",     "num_classes": 7},
    "M2C": {"source": "MLRSN",  "target": "CLRS",    "num_classes": 7},
    "M2R": {"source": "MLRSN",  "target": "RSSCN7",  "num_classes": 7},
    "R2A": {"source": "RSSCN7", "target": "AID",     "num_classes": 7},
    "R2C": {"source": "RSSCN7", "target": "CLRS",    "num_classes": 7},
    "R2M": {"source": "RSSCN7", "target": "MLRSN",   "num_classes": 7},
    # ---- cross-sensor ----
    "N2G": {"source": "NWPU",    "target": "NaSC",     "num_classes": 10},
    "G2N": {"source": "NaSC",    "target": "NWPU",     "num_classes": 10},
    "E2W": {"source": "EuroSAT", "target": "WHU",      "num_classes": 6},
    "W2E": {"source": "WHU",     "target": "EuroSAT", "num_classes": 6},
}

TASK = os.environ.get("TASK", "A2C")
SEED = int(os.environ.get("SEED", "0"))
_tc  = TASK_CONFIGS[TASK]


def seed_everything(seed: int = 0):
    """Deterministic seeding across Python, NumPy, PyTorch, CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


seed_everything(SEED)


class CFG:
    task         = TASK
    seed         = SEED
    source_name  = _tc["source"]
    target_name  = _tc["target"]
    num_classes  = _tc["num_classes"]

    backbone = "resnet50"
    img_size = 128

    # ---- Source pretraining: 50 epochs ----
    src_epochs     = 50
    src_batch_size = 128
    src_lr         = 1e-3

    # ---- Source-free adaptation: 15 epochs, NO early stopping ----
    sf_epochs     = 15
    sf_batch_size = 128
    sf_lr         = 2e-4
    # NOTE: sf_patience removed intentionally.
    # The classifier and all batch-normalization statistics are kept fixed.
    # Same BN policy for CAB-SFDA in the main runs and the ablations.
    freeze_bn     = True

    # ---- Loss weights ----
    lambda_im    = 0.3
    lambda_cons  = 1.0
    lambda_pl    = 1.0
    lambda_proto = 0.5

    # ---- PGAT ----
    conf_thresh_init         = 0.90
    conf_thresh_min          = 0.70
    conf_thresh_max          = 0.95
    conf_thresh_momentum     = 0.90
    min_thresh_samples       = 3
    proto_reliability_weight = 0.50
    threshold_gamma          = 0.10
    proto_min_count          = 100

    # ---- Consistency / memory ----
    temp_cons      = 0.5
    proto_momentum = 0.9
    feat_dim       = 128

    # ---- Runtime ----
    num_workers = 0        # 0 avoids DataLoader deadlock on Windows
    pin_memory  = True
    amp         = True

    data_root = os.environ.get("DATA_ROOT", "D:/Downloads/data")
    out_dir   = f"./runs/{TASK}/seed_{SEED}"


os.makedirs(CFG.out_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")