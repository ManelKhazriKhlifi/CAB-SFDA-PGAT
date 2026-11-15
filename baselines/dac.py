
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, DEVICE
from losses import info_max_loss
from _common import load_student, make_optimizer, save_results, RunLogger


def _supcon_loss(f, y, tau=0.1):
    n = f.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=f.device)
    sim = (f @ f.t() / tau).masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=f.device)
    log_prob = log_prob.masked_fill(~pos, 0.0)
    return -(log_prob.sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()


@torch.no_grad()
def _divide(student, dataset, conf_thresh, amp_on):
    loader = DataLoader(dataset, batch_size=CFG.sf_batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    feats = torch.zeros(len(dataset), CFG.feat_dim, device=DEVICE)
    probs = torch.zeros(len(dataset), CFG.num_classes, device=DEVICE)
    student.eval()
    for xw, _, idx in loader:
        with torch.cuda.amp.autocast(enabled=amp_on):
            z, f = student(xw.to(DEVICE), return_feat=True)
        idx = idx.to(DEVICE)
        feats[idx] = F.normalize(f.float(), dim=1)
        probs[idx] = F.softmax(z.float(), dim=1)

    conf, pred = probs.max(dim=1)
    src_like = conf >= conf_thresh
    w = F.one_hot(pred, CFG.num_classes).float() * src_like[:, None].float()
    empty = w.sum(dim=0) == 0
    w[:, empty] = probs[:, empty]
    protos = F.normalize(w.t() @ feats, dim=1)
    pseudo = (feats @ protos.t()).argmax(dim=1)
    return pseudo, src_like


def run(src_ckpt_path, loaders, arch=None, epochs=None, lr=None, verbose=True,
        conf_thresh=0.9, cls_par=0.3, lambda_local=0.5, backbone_lr_mult=0.1):
    arch   = arch or CFG.backbone
    epochs = epochs or CFG.sf_epochs
    lr     = lr     or CFG.sf_lr

    student = load_student(src_ckpt_path, arch=arch, freeze_classifier=True)
    if backbone_lr_mult == 1.0:
        optim = make_optimizer(student, lr)
    else:   
        optim = torch.optim.Adam([
            {"params": student.features.parameters(), "lr": lr * backbone_lr_mult},
            {"params": student.feature_head.parameters(), "lr": lr}])
    logger  = RunLogger("dac", src_ckpt_path,
                        {"arch": arch, "epochs": epochs, "lr": lr,
                         "conf_thresh": conf_thresh, "cls_par": cls_par,
                         "lambda_local": lambda_local,
                         "backbone_lr_mult": backbone_lr_mult})

    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    accepts = []

    for ep in range(1, epochs + 1):
        pseudo, src_like = _divide(student, loaders["tgt_unl"].dataset, conf_thresh, amp_on)
        student.train()
        pbar = tqdm(loaders["tgt_unl"],
                    desc=f"DaC {CFG.task} s{CFG.seed} ep {ep}/{epochs}")

        for xw, xs, idx in pbar:
            xw, xs, idx = xw.to(DEVICE), xs.to(DEVICE), idx.to(DEVICE)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                z_w, _   = student(xw, return_feat=True)
                z_s, f_s = student(xs, return_feat=True)
            z_w = z_w.float(); z_s = z_s.float(); f_s = f_s.float()
            y = pseudo[idx]

            loss_im = info_max_loss(z_w)
            loss_pl = F.cross_entropy(z_s, y)

            tgt_spec = ~src_like[idx]
            if tgt_spec.sum() > 1:
                loss_local = _supcon_loss(F.normalize(f_s[tgt_spec], dim=1), y[tgt_spec])
            else:
                loss_local = torch.tensor(0.0, device=DEVICE)

            loss = loss_im + cls_par * loss_pl + lambda_local * loss_local

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            logger.step(ep, im=loss_im, pl=loss_pl, local=loss_local, total=loss,
                        target_unique=tgt_spec.float().mean())
            if verbose:
                pbar.set_postfix(IM=f"{loss_im.item():.3f}",
                                 PL=f"{loss_pl.item():.3f}",
                                 Loc=f"{int(tgt_spec.sum())}/{tgt_spec.numel()}")

        accepts.append(src_like.float().mean().item())
        logger.end_epoch(ep, accepts[-1], student)

    return save_results(student, "dac", accepts, logger=logger)
