"""CAB-SFDA adaptation (paper Algorithm 1-3) with every ablation as a config switch.

Per batch:
  1. forward weak and strong views
  2. pseudo-labels y_hat = argmax p_w; acceptance mask from the PGAT thresholds of
     previous batches (causal lag)
  3. class weights and prototypes from previous batches
  4. L = l_pl * L_pl + l_im * L_im + l_proto * L_proto + l_cons * L_cons; optimizer step
  5. memory update with the (pre-step) weak features of the accepted samples
  6. PGAT threshold update (used from the next batch on)

Diagnostics are accumulated over every epoch (not the last batch only), so the
reported acceptance rate, mean confidence q, prototype agreement r and joint
reliability R refer to the same samples. Hidden target labels are used only
inside `EpochStats` to log pseudo-label quality.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .losses import consistency_loss, im_loss, weighted_pl_loss
from .memory import ClassMemory
from .optim import Stepper, ema_update
from .thresholds import PGAT, build_policy
from .utils import get_device, log, set_bn_mode


class EpochStats:
    def __init__(self, K, device):
        z = lambda dt=torch.float64: torch.zeros(K, dtype=dt, device=device)
        self.K = K
        self.n_pred, self.n_acc = z(), z()
        self.sum_conf, self.sum_r, self.n_r = z(), z(), z()
        self.acc_correct, self.true_seen, self.true_acc = z(), z(), z()
        self.loss_sums, self.n_batches = {}, 0

    @torch.no_grad()
    def add(self, conf, pred, mask, feats, memory, y_hidden, losses, proto_min):
        K = self.K
        oh = F.one_hot(pred, K).double()
        self.n_pred += oh.sum(0)
        self.n_acc += oh[mask].sum(0)
        self.sum_conf += (oh * conf.double()[:, None]).sum(0)
        if memory is not None and memory.enabled:
            ready = (memory.initialized & (memory.counts >= proto_min))[pred]
            if ready.any():
                cos = (F.normalize(feats[ready], dim=1) * F.normalize(memory.prototypes[pred[ready]], dim=1)).sum(1)
                self.sum_r += (oh[ready] * ((1 + cos.double()) / 2)[:, None]).sum(0)
                self.n_r += oh[ready].sum(0)
        # ---- logging only: hidden labels never reach a loss --------------------
        if y_hidden is not None:
            y = y_hidden.to(pred.device)
            yo = F.one_hot(y, K).double()
            self.true_seen += yo.sum(0)
            self.true_acc += yo[mask].sum(0)
            self.acc_correct += oh[mask & (pred == y)].sum(0)
        for k, v in losses.items():
            self.loss_sums[k] = self.loss_sums.get(k, 0.0) + float(v)
        self.n_batches += 1

    def summary(self, lam_r):
        def arr(t):
            return t.cpu().numpy()
        n_pred, n_acc = arr(self.n_pred), arr(self.n_acc)
        q = np.divide(arr(self.sum_conf), n_pred, out=np.full(self.K, np.nan), where=n_pred > 0)
        r = np.divide(arr(self.sum_r), arr(self.n_r), out=np.full(self.K, np.nan), where=arr(self.n_r) > 0)
        R = np.where(np.isnan(r), q, lam_r * q + (1 - lam_r) * r)
        acc_correct, true_seen, true_acc = arr(self.acc_correct), arr(self.true_seen), arr(self.true_acc)
        pl_prec = np.divide(acc_correct, n_acc, out=np.full(self.K, np.nan), where=n_acc > 0)
        pl_rec = np.divide(acc_correct, true_seen, out=np.full(self.K, np.nan), where=true_seen > 0)
        obs = n_pred.sum()
        corr = np.corrcoef(n_acc, true_seen)[0, 1] if n_acc.std() > 0 and true_seen.std() > 0 else np.nan
        return {
            "n_observed": int(obs), "n_accepted": int(n_acc.sum()),
            "accept_rate": 100.0 * n_acc.sum() / max(1, obs),
            "pl_accuracy_accepted": 100.0 * acc_correct.sum() / max(1, n_acc.sum()),
            "pl_accuracy_all": None,       # filled by caller (argmax accuracy over all observed)
            "per_class": {
                "n_pred": n_pred.tolist(), "n_accepted": n_acc.tolist(),
                "accept_rate": (100 * np.divide(n_acc, n_pred, out=np.full(self.K, np.nan),
                                                where=n_pred > 0)).tolist(),
                "q": q.tolist(), "r": r.tolist(), "R": R.tolist(),
                "pl_precision": (100 * pl_prec).tolist(), "pl_recall": (100 * pl_rec).tolist(),
                "true_count_seen": true_seen.tolist(), "true_count_accepted": true_acc.tolist(),
            },
            "corr_accepted_vs_true_counts": None if np.isnan(corr) else float(corr),
            "losses": {k: v / max(1, self.n_batches) for k, v in self.loss_sums.items()},
        }


@torch.no_grad()
def _static_prototypes(model, data, memory, cfg, device):
    """'static' memory ablation: prototypes from the source model's confident target predictions."""
    model.eval()
    feats, preds = [], []
    for x, _ in data.adapt_eval:
        z, f = model(x.to(device), return_feat=True)
        p = z.float().softmax(1)
        conf, pred = p.max(1)
        keep = conf >= cfg.tau_base
        feats.append(f.float()[keep])
        preds.append(pred[keep])
    memory.init_static(torch.cat(feats), torch.cat(preds))


def adapt(model, data, cfg, info, logger):
    """Default: an EMA teacher (weak view) gives the pseudo-labels, PGAT confidences, memory
    features and the distillation target; the student is trained on the strong view (L_pl, L_cons)
    and the weak view (L_im, L_proto). `teacher: none` makes the student its own teacher (ablation)."""
    device = get_device()
    K = info["num_classes"]
    stepper = Stepper(model, cfg, len(data.train), device)
    amp = stepper.amp
    teacher = None
    if cfg.teacher == "ema":
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    elif cfg.teacher != "none":
        raise ValueError(f"teacher must be 'ema' or 'none', got {cfg.teacher}")

    memory = ClassMemory(K, cfg.feat_dim, cfg.proto_momentum, cfg.memory_mode, device)
    if cfg.memory_mode == "static":
        _static_prototypes(model, data, memory, cfg, device)
    policy = build_policy(cfg, K, device, data.n_adapt)

    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        set_bn_mode(model, cfg.bn_mode)
        stats = EpochStats(K, device)
        argmax_correct, t0, skipped0 = 0, time.time(), stepper.skipped
        for batch in data.train:
            xw = batch["weak"].to(device, non_blocking=True)
            xs = batch["strong"].to(device, non_blocking=True)
            idx = batch["idx"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                zw, fw = model(xw, return_feat=True)
                zs, _ = model(xs, return_feat=True)
                if teacher is not None:
                    with torch.no_grad():
                        zt, ft = teacher(xw, return_feat=True)
            zw, fw, zs = zw.float(), fw.float(), zs.float()
            if teacher is None:
                zt, ft = zw.detach(), fw.detach()
            zt, ft = zt.float(), ft.float()

            with torch.no_grad():
                probs = zt.softmax(1)
                conf, pred = probs.max(1)
                if policy.update_before_mask:             # FreeMatch / 'no causal lag' ablation
                    policy.update(probs, conf, pred, ft, memory, idx)
                mask = policy.mask(conf, pred, idx)
                w = memory.class_weights(cfg.weighting)

            l_pl = weighted_pl_loss(zs[mask], pred[mask], w)
            l_im = im_loss(zw, cfg.im_mode) if cfg.lambda_im > 0 else zw.new_zeros(())
            l_proto = (memory.proto_loss(fw[mask], pred[mask], cfg.proto_loss)
                       if cfg.lambda_proto > 0 else zw.new_zeros(()))
            if cfg.lambda_cons > 0:
                # one-way: KL(teacher weak || student strong); symmetric (ablation): student weak <-> strong
                src = zt if (teacher is not None and cfg.cons_mode == "oneway") else zw
                l_cons = consistency_loss(src, zs, cfg.temp_cons, cfg.cons_mode)
            else:
                l_cons = zw.new_zeros(())
            loss = (cfg.lambda_pl * l_pl + cfg.lambda_im * l_im
                    + cfg.lambda_proto * l_proto + cfg.lambda_cons * l_cons)
            stepper.step(loss)
            if teacher is not None:
                ema_update(teacher, model, cfg.ema_momentum)

            with torch.no_grad():
                if cfg.memory_update == "gated":
                    memory.update(ft[mask], pred[mask])
                elif cfg.memory_update == "all":
                    memory.update(ft, pred)
                else:
                    raise ValueError(f"memory_update {cfg.memory_update}")
                if not policy.update_before_mask:
                    policy.update(probs, conf, pred, ft, memory, idx)
                y_log = batch["y_hidden"] if cfg.log_pl_quality else None   # logging only
                if y_log is not None:
                    argmax_correct += int((pred.cpu() == y_log).sum())
                stats.add(conf, pred, mask, ft, memory, y_log,
                          {"total": loss.item(), "pl": l_pl.item(), "im": l_im.item(),
                           "proto": l_proto.item(), "cons": l_cons.item()}, cfg.proto_min_count)

        s = stats.summary(cfg.lambda_r if cfg.threshold != "pgat_conf_only" else 1.0)
        s["pl_accuracy_all"] = 100.0 * argmax_correct / max(1, s["n_observed"])
        n_pred = np.asarray(s["per_class"]["n_pred"], dtype=float)
        frac = n_pred / max(1.0, n_pred.sum())
        s["pred_class_entropy"] = float(-(frac[frac > 0] * np.log(frac[frac > 0])).sum() / np.log(K))
        s.update({"epoch": ep, "iterations": stepper.it, "seconds": time.time() - t0,
                  "lr": stepper.current_lr(), "skipped_steps": stepper.skipped - skipped0,
                  "policy": policy.snapshot(), "memory_counts": memory.counts.tolist(),
                  "sum_counts": int(memory.counts.sum()),
                  "class_weights": memory.class_weights(cfg.weighting).tolist(),
                  "prototype_initialized": memory.initialized.tolist()})
        history.append(s)
        logger(s)
        log(f"{info['task']} ep {ep}/{cfg.epochs} accept {s['accept_rate']:.1f}% "
            f"PL-acc(acc.) {s['pl_accuracy_accepted']:.1f}% loss {s['losses'].get('total', float('nan')):.3f} "
            f"pred-entropy {s['pred_class_entropy']:.2f} mean tau {np.mean(s['policy']['tau']):.4f}")

    last = history[-1]
    n_seen_total = sum(h["n_observed"] for h in history)
    diag = {
        "accept_rate_final_epoch": last["accept_rate"],
        "sum_counts": last["sum_counts"],
        "sum_counts_cap": n_seen_total,          # every observed sample accepted in every epoch
        "sum_counts_within_cap": last["sum_counts"] <= n_seen_total,
        "tau_mean_final": float(np.mean(last["policy"]["tau"])),
        "tau_final": last["policy"]["tau"],
        "q_mean_final_epoch": float(np.nanmean(last["per_class"]["q"])),
        "r_mean_final_epoch": (float(np.nanmean(last["per_class"]["r"]))
                               if not np.all(np.isnan(last["per_class"]["r"])) else None),
        "R_mean_final_epoch": float(np.nanmean(last["per_class"]["R"])),
        "prototypes_active_final": int(sum(
            1 for c, ok in enumerate(memory.initialized.tolist())
            if ok and memory.counts[c] >= cfg.proto_min_count)) if memory.enabled else 0,
        "pl_accuracy_accepted_final_epoch": last["pl_accuracy_accepted"],
        "is_pgat": isinstance(policy, PGAT),
        "pred_class_entropy_final_epoch": last["pred_class_entropy"],
        "skipped_steps": stepper.skipped,
        "teacher": cfg.teacher,
    }
    return {"history": history, "diagnostics": diag}
