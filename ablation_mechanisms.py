import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from dataloaders import build_loaders
from train_source import get_or_train_source
from evaluate import run_final_evaluation
from model import ResNetBackbone
from threshold import PrototypeGuidedClassThreshold
from losses import info_max_loss, entropy_min_loss, kl_with_temperature

# Import the original ClassMemory just to use its default weights calculation
from memory import ClassMemory


def run_ablation_adaptation(src_ckpt_path, loaders, 
                            trainable_classifier=False, freeze_bn=None,
                            balancing="inv_sqrt",
                            memory_mode="momentum",
                            pgat_mode="full",
                            consistency="symmetric",
                            im_mode="im",
                            out_dir=None):
    """Self-contained adaptation loop supporting all mechanism-ablation variants.

    Defaults == CAB-SFDA exactly as in adapt.py, incl. freeze_bn=CFG.freeze_bn,
    so the "(ours)" variants are the same method as the main runs.
    """
    freeze_bn = CFG.freeze_bn if freeze_bn is None else freeze_bn
    out_dir   = out_dir or os.path.join(CFG.out_dir, "ablation_mechanism", "default")
    os.makedirs(out_dir, exist_ok=True)
    arch   = CFG.backbone
    epochs = CFG.sf_epochs
    lr     = CFG.sf_lr

    student = ResNetBackbone(num_classes=CFG.num_classes,
                             feat_dim=CFG.feat_dim,
                             arch=arch).to(DEVICE)
    student.load_state_dict(torch.load(src_ckpt_path, map_location=DEVICE))

    # (E) Classifier policy
    if trainable_classifier:
        for p in student.classifier.parameters():
            p.requires_grad_(True)
    else:
        for p in student.classifier.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [p for p in list(student.features.parameters())
                    + list(student.feature_head.parameters())
                    + list(student.classifier.parameters())
         if p.requires_grad],
        lr=lr,
    )

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    # (B) Prototype memory setup
    memory = ClassMemory(num_classes=CFG.num_classes,
                         feat_dim=CFG.feat_dim,
                         momentum=CFG.proto_momentum)
    
    # (C) Threshold setup
    warmup_count = 0 if pgat_mode == "no_warmup" else CFG.proto_min_count
    adaptive_thresh = PrototypeGuidedClassThreshold(
        num_classes=CFG.num_classes,
        init=CFG.conf_thresh_init,
        tau_min=CFG.conf_thresh_min,
        tau_max=CFG.conf_thresh_max,
        momentum=CFG.conf_thresh_momentum,
        min_samples=CFG.min_thresh_samples,
        reliability_weight=CFG.proto_reliability_weight,
        gamma=CFG.threshold_gamma,
        proto_min_count=warmup_count,
    )

    accept_history = []

    for ep in range(1, epochs + 1):
        student.train()
        if freeze_bn:
            for m in student.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"Ablation {CFG.task} ep {ep}/{epochs}", leave=False)
        accepted, observed = 0, 0

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                logits_w, feats_w = student(xw, return_feat=True)
                logits_s, feats_s = student(xs, return_feat=True)
            logits_w = logits_w.float(); feats_w = feats_w.float()
            logits_s = logits_s.float(); feats_s = feats_s.float()

            # (D) IM variant: full IM (ours) vs entropy-only (no diversity term)
            if im_mode == "entropy_only":
                loss_im = entropy_min_loss(logits_w)
            else:
                loss_im = info_max_loss(logits_w)

            # (D) Consistency
            kl_ws = kl_with_temperature(logits_w, logits_s, T=CFG.temp_cons)
            if consistency == "symmetric":
                kl_sw = kl_with_temperature(logits_s, logits_w, T=CFG.temp_cons)
                loss_cons = 0.5 * (kl_ws + kl_sw)
            else:
                loss_cons = kl_ws

            p_w = torch.softmax(logits_w, dim=1)
            conf, y_hat = p_w.max(dim=1)

            # (C) Mask selection
            if pgat_mode == "fixed_090":
                mask = conf.detach() >= 0.90
            elif pgat_mode == "fixed_080":
                mask = conf.detach() >= 0.80
            elif pgat_mode == "class_agnostic":
                # one shared threshold (all entries equal), updated post-step
                mask = conf.detach() >= adaptive_thresh.thresholds[0]
            elif pgat_mode == "no_lag":
                adaptive_thresh.update(conf.detach(), y_hat.detach(),
                                       feats_w.detach(), memory)
                mask = adaptive_thresh.get_mask(conf.detach(), y_hat.detach())
            else:  # full / no_warmup
                mask = adaptive_thresh.get_mask(conf.detach(), y_hat.detach())

            accepted += int(mask.sum().item())
            observed += int(mask.numel())

            # (A) Class-balancing pseudo-label loss
            if mask.sum() > 0:
                if balancing == "none":
                    cw = torch.ones(CFG.num_classes, device=DEVICE)
                elif balancing == "inv_freq":
                    inv = 1.0 / (memory.counts + 1e-6)
                    cw = (inv / inv.sum().clamp_min(1e-6) * CFG.num_classes).detach()
                else:  # inv_sqrt
                    cw = memory.get_weights()
                
                ce = F.cross_entropy(logits_s[mask], y_hat[mask], reduction="none")
                loss_pl = (ce * cw[y_hat[mask]]).mean()
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            # Prototype attraction
            if mask.sum() > 0 and memory_mode != "none" and memory.initialized.any():
                loss_proto = memory.proto_loss(feats_w[mask], y_hat[mask])
            else:
                loss_proto = torch.tensor(0.0, device=DEVICE)

            loss = (CFG.lambda_im * loss_im
                    + CFG.lambda_cons * loss_cons
                    + CFG.lambda_pl * loss_pl
                    + CFG.lambda_proto * loss_proto)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # (B) Memory update logic
            if mask.sum() > 0 and memory_mode != "none":
                if memory_mode == "static":
                    if ep == 1:  # only update in first epoch
                        memory.update(feats_w.detach()[mask], y_hat.detach()[mask])
                elif memory_mode == "batch":
                    # Batch mode (no EMA)
                    for c in range(CFG.num_classes):
                        idx = (y_hat[mask] == c).nonzero(as_tuple=False).flatten()
                        if idx.numel() > 0:
                            fmean = feats_w[mask][idx].mean(dim=0)
                            memory.prototypes[c] = fmean.detach()
                            memory.initialized[c] = True
                            memory.counts[c] += idx.numel()
                elif memory_mode == "ema_unfiltered":
                    pass  # handled below: updated from ALL predictions
                else:  # momentum (ours): prototypes updated from accepted samples only
                    memory.update(feats_w.detach()[mask], y_hat.detach()[mask])
            if memory_mode == "ema_unfiltered":
                # B4: same per-class EMA, but NOT gated by PGAT — every
                # prediction (incl. rejected, low-confidence ones) updates mu_c
                memory.update(feats_w.detach(), y_hat.detach())

            # Threshold update
            if pgat_mode == "class_agnostic":
                adaptive_thresh.update_global(conf.detach(), y_hat.detach(),
                                              feats_w.detach(), memory)
            elif pgat_mode not in ("fixed_090", "fixed_080", "no_lag"):
                adaptive_thresh.update(conf.detach(), y_hat.detach(),
                                       feats_w.detach(), memory)

        accept_history.append(accepted / max(observed, 1))

    final_path = os.path.join(out_dir, "student_final.pth")
    torch.save(student.state_dict(), final_path)
    return {"final_ckpt": final_path, "accept_history": accept_history}


def run(loaders):
    src_ckpt = get_or_train_source(loaders)

    variants = {
        # ---- (A) Class-balancing strategy ----
        "A1_unweighted":              dict(balancing="none"),
        "A2_inv_freq":                dict(balancing="inv_freq"),
        "A3_inv_sqrt (ours)":         dict(balancing="inv_sqrt"),

        # ---- (B) Prototype memory ----
        "B1_no_prototype":            dict(memory_mode="none"),
        "B2_static_prototype":        dict(memory_mode="static"),
        "B3_batch_prototype":         dict(memory_mode="batch"),
        "B4_EMA_unfiltered":          dict(memory_mode="ema_unfiltered"),
        "B5_momentum_classaware (ours)": dict(memory_mode="momentum"),

        # ---- (C) Threshold mechanism ----
        "C1_fixed_0.90":              dict(pgat_mode="fixed_090"),
        "C2_fixed_0.80_tuned":        dict(pgat_mode="fixed_080"),
        "C3_class_agnostic":          dict(pgat_mode="class_agnostic"),
        "C4_no_causal_lag":           dict(pgat_mode="no_lag"),
        "C5_no_warmup":               dict(pgat_mode="no_warmup"),
        "C6_full_PGAT (ours)":        dict(pgat_mode="full"),

        # ---- (D) Loss refinements ----
        "D1_one_way_KL":              dict(consistency="one_way"),
        "D2_entropy_only_IM":         dict(im_mode="entropy_only"),
        "D3_symmetric_KL_IM (ours)":  dict(consistency="symmetric"),

        # ---- (E) Classifier state ----
        "E1_trainable_clf":           dict(trainable_classifier=True,  freeze_bn=True),
        "E2_frozen_clf_BNtrain":      dict(trainable_classifier=False, freeze_bn=False),
        "E3_frozen_frozenBN (ours)":  dict(trainable_classifier=False, freeze_bn=True),
    }

    root = os.path.join(CFG.out_dir, "ablation_mechanism")
    results = {}
    
    print(f"\nStarting mechanism ablation: {len(variants)} variants.")
    print(f"Task: {CFG.task} | Seed: {CFG.seed}\n")

    for name, kw in variants.items():
        print(f"\n{'=' * 70}")
        print(f"  Evaluating variant: {name}")
        print(f"  Params: {kw}")
        print(f"{'=' * 70}")

        try:
            # Call the SELF-CONTAINED adaptation loop
            var_dir = os.path.join(root, re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_"))
            out = run_ablation_adaptation(src_ckpt, loaders, out_dir=var_dir, **kw)
            
            report = run_final_evaluation(out["final_ckpt"], loaders,
                                          out_dir=var_dir, method=name)

            results[name] = {
                "final_acc":         float(report["overall_acc"] * 100),
                "balanced_acc":      float(report["balanced_acc"] * 100),
                "macro_f1":          float(report["macro_f1"] * 100),
                "per_class_f1":      [float(x) * 100 for x in report["per_class_f1"]],
                "accept_rate_final": float(out["accept_history"][-1] * 100),
            }
            print(f"  → OA = {results[name]['final_acc']:.2f}%   "
                  f"Acc.rate = {results[name]['accept_rate_final']:.2f}%")
                  
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[name] = {"error": str(e)}

        path = os.path.join(root, "ablation_mechanism.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  Summary — Mechanism-wise Ablation")
    print("=" * 80)
    
    current_group = None
    for name, r in results.items():
        group = name[0]
        if group != current_group:
            print(f"\n--- Group ({group}) ---")
            current_group = group
            
        if "error" in r:
            print(f"  {name:<32}  ERROR: {r['error'][:30]}")
        else:
            min_f1 = min(r["per_class_f1"]) if r["per_class_f1"] else 0.0
            print(f"  {name:<32}  OA = {r['final_acc']:6.2f}%  "
                  f"Min-F1 = {min_f1:6.2f}%  "
                  f"Acc.Rate = {r['accept_rate_final']:5.2f}%")

    print(f"\nAll results saved to: {os.path.join(root, 'ablation_mechanism.json')}")


if __name__ == "__main__":
    run(build_loaders())