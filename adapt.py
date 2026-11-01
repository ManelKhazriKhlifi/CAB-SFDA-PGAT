import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE
from model import ResNetBackbone
from memory import ClassMemory
from threshold import PrototypeGuidedClassThreshold
from losses import info_max_loss, kl_with_temperature


def run_adaptation(src_ckpt_path, loaders, arch=None,
                   epochs=None, lr=None, verbose=True,
                   trainable_classifier=False,
                   freeze_bn=None,                # None → CFG.freeze_bn
                   out_dir=None):                 # per-method / per-variant folder
    arch   = arch   or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr
    freeze_bn = CFG.freeze_bn if freeze_bn is None else freeze_bn
    out_dir   = out_dir or os.path.join(CFG.out_dir, "cab_sfda")
    os.makedirs(out_dir, exist_ok=True)

    student = ResNetBackbone(num_classes=CFG.num_classes,
                             feat_dim=CFG.feat_dim,
                             arch=arch).to(DEVICE)
    student.load_state_dict(torch.load(src_ckpt_path, map_location=DEVICE))

    # ---- Classifier policy -------------------------------------------
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

    memory = ClassMemory(num_classes=CFG.num_classes,
                         feat_dim=CFG.feat_dim,
                         momentum=CFG.proto_momentum)

    adaptive_thresh = PrototypeGuidedClassThreshold(
        num_classes=CFG.num_classes,
        init=CFG.conf_thresh_init,
        tau_min=CFG.conf_thresh_min,
        tau_max=CFG.conf_thresh_max,
        momentum=CFG.conf_thresh_momentum,
        min_samples=CFG.min_thresh_samples,
        reliability_weight=CFG.proto_reliability_weight,
        gamma=CFG.threshold_gamma,
        proto_min_count=CFG.proto_min_count,
    )

    threshold_history = []
    accept_history    = []

    results_log_path = os.path.join(out_dir, "adaptation_results.json")
    epoch_results    = []

    for ep in range(1, epochs + 1):
        student.train()
        if freeze_bn:
            for m in student.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"CAB-SFDA {CFG.task} seed={CFG.seed} ep {ep}/{epochs}")
        accepted, observed = 0, 0

        for xw, xs, _ in pbar:
            xw, xs = xw.to(DEVICE), xs.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                logits_w, feats_w = student(xw, return_feat=True)
                logits_s, feats_s = student(xs, return_feat=True)
            logits_w = logits_w.float(); feats_w = feats_w.float()
            logits_s = logits_s.float(); feats_s = feats_s.float()

            loss_im = info_max_loss(logits_w)
            kl_ws   = kl_with_temperature(logits_w, logits_s, T=CFG.temp_cons)
            kl_sw   = kl_with_temperature(logits_s, logits_w, T=CFG.temp_cons)
            loss_cons = 0.5 * (kl_ws + kl_sw)

            p_w = torch.softmax(logits_w, dim=1)
            conf, y_hat = p_w.max(dim=1)

            mask = adaptive_thresh.get_mask(conf.detach(), y_hat.detach())
            accepted += int(mask.sum().item())
            observed += int(mask.numel())

            if mask.sum() > 0:
                cw = memory.get_weights()
                ce = F.cross_entropy(logits_s[mask], y_hat[mask], reduction="none")
                loss_pl = (ce * cw[y_hat[mask]]).mean()
            else:
                loss_pl = torch.tensor(0.0, device=DEVICE)

            if mask.sum() > 0 and memory.initialized.any():
                loss_proto = memory.proto_loss(feats_w[mask], y_hat[mask])
            else:
                loss_proto = torch.tensor(0.0, device=DEVICE)

            loss = (CFG.lambda_im    * loss_im
                    + CFG.lambda_cons * loss_cons
                    + CFG.lambda_pl   * loss_pl
                    + CFG.lambda_proto * loss_proto)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if mask.sum() > 0:
                memory.update(feats_w.detach()[mask], y_hat.detach()[mask])
            adaptive_thresh.update(conf.detach(), y_hat.detach(),
                                   feats_w.detach(), memory)

            pbar.set_postfix(
                IM=f"{loss_im.item():.3f}",
                Cons=f"{loss_cons.item():.3f}",
                PL=f"{loss_pl.item():.3f}",
                Proto=f"{loss_proto.item():.3f}",
                Accept=f"{100.0 * accepted / max(observed, 1):.1f}%",
            )

        tau_e   = adaptive_thresh.thresholds.detach().cpu().numpy().copy()
        conf_e  = adaptive_thresh.conf_reliability.detach().cpu().numpy().copy()
        proto_e = adaptive_thresh.proto_reliability.detach().cpu().numpy().copy()
        joint_e = adaptive_thresh.joint_reliability.detach().cpu().numpy().copy()
        act_e   = adaptive_thresh.prototype_active.detach().cpu().numpy().copy()
        acc_rate = accepted / max(observed, 1)

        threshold_history.append(tau_e)
        accept_history.append(acc_rate)

        if verbose:
            print(f"[CAB-SFDA] Epoch {ep}/{epochs} "
                  f"acceptance={100.0*acc_rate:.2f}%")

        # NOTE: no target-test evaluation, no best_tgt, no early stopping
        epoch_results.append({
            "epoch": ep,
            "class_thresholds":       [round(float(x), 4) for x in tau_e],
            "confidence_reliability": [round(float(x), 4) for x in conf_e],
            "prototype_reliability":  [round(float(x), 4) for x in proto_e],
            "joint_reliability":      [round(float(x), 4) for x in joint_e],
            "prototype_active":       [bool(x) for x in act_e],
            "prototype_counts":       [int(x) for x in memory.counts.detach().cpu().tolist()],
            "pseudo_label_acceptance_rate": round(100.0 * acc_rate, 2),
        })
        with open(results_log_path, "w") as f:
            json.dump(epoch_results, f, indent=2)

    final_ckpt_path = os.path.join(out_dir, "student_final.pth")
    torch.save(student.state_dict(), final_ckpt_path)
    print("Saved FINAL-epoch adapted model:", final_ckpt_path)

    return {
        "final_ckpt": final_ckpt_path,
        "out_dir": out_dir,
        "threshold_history": threshold_history,
        "accept_history": accept_history,
        "model": student,
    }