"""
adapt.py
The core CAB-SFDA adaptation loop.
"""
import os
import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG
from model import ResNetBackbone
from memory import ClassMemory
from threshold import PrototypeGuidedClassThreshold
from losses import info_max_loss, standard_im_loss, consistency_loss
from dataloaders import build_loaders
from evaluate import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_adaptation(abl, tag="default"):
    """
    abl: object with attributes:
        disable_pl, disable_im, disable_proto, disable_cons,
        class_balancing, proto_mode, threshold_mode,
        loss_refinement, classifier_state, imbalance_ratio
    """
    os.makedirs(CFG.out_dir, exist_ok=True)

    # ---- DataLoaders ----
    (_, _, tgt_unl_loader, tgt_test_loader, _) = build_loaders(abl.imbalance_ratio)

    # ---- Build student from source checkpoint ----
    src_ckpt = os.path.join(CFG.out_dir, "source_aid_resnet50.pth")
    student = ResNetBackbone(num_classes=CFG.num_classes,
                             feat_dim=CFG.feat_dim,
                             backbone_name=CFG.backbone).to(device)
    student.load_state_dict(torch.load(src_ckpt, map_location=device))

    # ---- Classifier / BN freeze policy (ablation E) ----
    for p in student.classifier.parameters():
        p.requires_grad_(False)

    freeze_bn = (abl.classifier_state == "frozen_bn_frozen")

    # ---- Optimizer ----
    params = list(student.features.parameters()) + \
             list(student.feature_head.parameters())
    if abl.classifier_state == "trainable":
        params += list(student.classifier.parameters())
    optimizer = torch.optim.Adam([p for p in params if p.requires_grad],
                                 lr=CFG.sf_lr)

    # ---- Prototype memory ----
    memory = ClassMemory(
        num_classes=CFG.num_classes,
        feat_dim=CFG.feat_dim,
        momentum=CFG.proto_momentum,
        mode=abl.proto_mode,
        device=device,
    )

    # ---- Threshold (PGAT ablation) ----
    thr_init = {"fixed_080": 0.80}.get(abl.threshold_mode, CFG.conf_thresh_init)
    adaptive_thresh = PrototypeGuidedClassThreshold(
        num_classes=CFG.num_classes,
        init=thr_init,
        tau_min=CFG.conf_thresh_min,
        tau_max=CFG.conf_thresh_max,
        momentum=CFG.conf_thresh_momentum,
        min_samples=CFG.min_thresh_samples,
        reliability_weight=CFG.proto_reliability_weight,
        gamma=CFG.threshold_gamma,
        proto_min_count=CFG.proto_min_count,
        mode=abl.threshold_mode,
        device=device,
    )

    # ---- Histories ----
    epoch_results = []
    best_tgt = -1.0
    patience = 0
    results_path = os.path.join(CFG.out_dir, f"adaptation_{tag}.json")

    # ---- Loop ----
    for ep in range(1, CFG.sf_epochs + 1):
        student.train()
        if freeze_bn:
            for m in student.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.eval()

        accepted, observed = 0, 0
        pbar = tqdm(tgt_unl_loader, desc=f"[{tag}] Epoch {ep}/{CFG.sf_epochs}")

        for xw, xs, _ in pbar:
            xw, xs = xw.to(device), xs.to(device)
            optimizer.zero_grad()

            logits_w, feats_w = student(xw, return_feat=True)
            logits_s, feats_s = student(xs, return_feat=True)

            # --- IM loss ---
            if not abl.disable_im:
                if abl.loss_refinement == "standard_im":
                    loss_im = standard_im_loss(logits_w)
                else:
                    loss_im = info_max_loss(logits_w)
            else:
                loss_im = torch.tensor(0.0, device=device)

            # --- Consistency loss ---
            if not abl.disable_cons:
                kl_mode = "one_way" if abl.loss_refinement == "one_way_kl" else "symmetric"
                loss_cons = consistency_loss(logits_w, logits_s,
                                             T=CFG.temp_cons, mode=kl_mode)
            else:
                loss_cons = torch.tensor(0.0, device=device)

            # --- Predictions ---
            p_w = torch.softmax(logits_w, dim=1)
            conf, y_hat = p_w.max(dim=1)

            # --- Causal lag toggle ---
            if abl.threshold_mode == "per_class_no_lag":
                adaptive_thresh.update(conf.detach(), y_hat.detach(),
                                       feats_w.detach(), memory)

            mask = adaptive_thresh.get_mask(conf.detach(), y_hat.detach())
            accepted += int(mask.sum().item())
            observed += int(mask.numel())

            # --- Pseudo-label loss ---
            if (not abl.disable_pl) and mask.sum() > 0:
                cw = memory.get_weights(mode=abl.class_balancing)
                ce = F.cross_entropy(logits_s[mask], y_hat[mask], reduction="none")
                loss_pl = (ce * cw[y_hat[mask]]).mean()
            else:
                loss_pl = torch.tensor(0.0, device=device)

            # --- Prototype loss ---
            if (not abl.disable_proto) and mask.sum() > 0 and memory.initialized.any():
                loss_proto = memory.proto_loss(feats_w[mask], y_hat[mask])
            else:
                loss_proto = torch.tensor(0.0, device=device)

            # --- Total ---
            loss = (CFG.lambda_im * loss_im +
                    CFG.lambda_cons * loss_cons +
                    CFG.lambda_pl * loss_pl +
                    CFG.lambda_proto * loss_proto)
            loss.backward()
            optimizer.step()

            # --- Post-step: memory update ---
            if mask.sum() > 0:
                memory.update(feats_w.detach()[mask], y_hat.detach()[mask])

            # --- Post-step: threshold update (default causal lag) ---
            if abl.threshold_mode != "per_class_no_lag":
                adaptive_thresh.update(conf.detach(), y_hat.detach(),
                                       feats_w.detach(), memory)

            pbar.set_postfix(
                IM=f"{loss_im.item():.3f}", Cons=f"{loss_cons.item():.3f}",
                PL=f"{loss_pl.item():.3f}", Proto=f"{loss_proto.item():.3f}",
                Acc=f"{100.0 * accepted / max(observed, 1):.1f}%",
            )

        # --- Epoch evaluation ---
        acc_tgt = accuracy(student, tgt_test_loader)
        if acc_tgt >= best_tgt:
            best_tgt = acc_tgt
            patience = 0
        else:
            patience += 1

        epoch_results.append({
            "epoch": ep,
            "target_test_acc": float(acc_tgt),
            "best_target_test_acc": float(best_tgt),
            "class_thresholds": adaptive_thresh.thresholds.detach().cpu().tolist(),
            "confidence_reliability": adaptive_thresh.conf_reliability.detach().cpu().tolist(),
            "prototype_reliability": adaptive_thresh.proto_reliability.detach().cpu().tolist(),
            "joint_reliability": adaptive_thresh.joint_reliability.detach().cpu().tolist(),
            "prototype_active": adaptive_thresh.prototype_active.detach().cpu().tolist(),
            "prototype_counts": memory.counts.detach().cpu().tolist(),
            "pseudo_label_acceptance_rate": 100.0 * accepted / max(observed, 1),
        })
        with open(results_path, "w") as f:
            json.dump(epoch_results, f, indent=2)

        print(f"[{tag}] Epoch {ep}: acc = {acc_tgt:.2f}% (best {best_tgt:.2f}%)")

        if patience >= CFG.sf_patience:
            print(f"Early stopping at epoch {ep} (best {best_tgt:.2f}%)")
            break

    return best_tgt, epoch_results
