"""
config.py
Shared configuration for all CAB-SFDA ablation scripts.
"""
import os

class CFG:
    # ---- Data ----
    source_name = "AID"
    target_name = "CLRS"
    num_classes = 7
    DATA_ROOT = os.environ.get("DATA_ROOT", "D:/Downloads/data")

    # ---- Source training ----
    src_epochs = 20
    src_batch_size = 128
    src_lr = 1e-3

    # ---- Source-free adaptation ----
    sf_epochs = 30
    sf_batch_size = 128
    sf_lr = 2e-4
    sf_patience = 4

    # ---- Loss weights ----
    lambda_im = 0.3
    lambda_cons = 1.0
    lambda_pl = 1.0
    lambda_proto = 0.5

    # ---- PGAT ----
    conf_thresh_init = 0.90
    conf_thresh_min = 0.70
    conf_thresh_max = 0.95
    conf_thresh_momentum = 0.90
    min_thresh_samples = 3
    proto_reliability_weight = 0.50
    threshold_gamma = 0.10
    proto_min_count = 100

    # ---- Misc ----
    temp_cons = 0.5
    proto_momentum = 0.9
    feat_dim = 128
    num_workers = 4
    out_dir = "./cab_sfda_ckpts"

    # ---- Backbone ----
    backbone = "resnet50"   # "resnet18" | "resnet50" | "resnet101"
