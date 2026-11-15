
import glob
import os
import json
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm

from config import CFG, DEVICE
from model import ResNetBackbone, accuracy


def train_source(model, train_loader, test_loader,
                 epochs=CFG.src_epochs, lr=CFG.src_lr,
                 log_path=None, target_test_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    amp_on = False
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    log = []
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Source ep {ep}/{epochs}")
        loss_sum, n_batches = 0.0, 0
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Source training loss is {loss.item()} at epoch {ep}; "
                                   "the source model would be broken, stopping.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = loss_sum / max(n_batches, 1)
        acc_src = accuracy(model, test_loader)
        print(f"[Source] Epoch {ep}: train loss = {train_loss:.4f}  "
              f"source test acc = {acc_src:.2f}%")
        log.append({"epoch": ep,
                    "source_train_loss": float(train_loss),
                    "source_test_acc": float(acc_src)})

    
    if target_test_loader is not None:
        src_only_acc = accuracy(model, target_test_loader)
        print(f"Source-only accuracy on {CFG.target_name} test: "
              f"{src_only_acc:.2f}%")
        log.append({
            "source_only_on_target": {
                "target": CFG.target_name,
                "accuracy": float(src_only_acc),
            }
        })

    if log_path is None:
        log_path = os.path.join(CFG.src_dir, "source_training_results.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print("Saved source-training results:", log_path)
    return model


def _copy_shared_source(ckpt_path):
    """Reuse a source model already trained by another method for the same task/seed,
    so every method of a task/seed adapts from the identical source model."""
    pattern = os.path.join(CFG.runs_root, CFG.task, "*", f"seed_{CFG.seed}",
                           "source", "source_model_final.pth")
    for other in sorted(glob.glob(pattern)):
        if os.path.abspath(other) == os.path.abspath(ckpt_path):
            continue
        other_dir = os.path.dirname(other)
        os.makedirs(CFG.src_dir, exist_ok=True)
        shutil.copy2(other, ckpt_path)
        log = os.path.join(other_dir, "source_training_results.json")
        if os.path.exists(log):
            shutil.copy2(log, os.path.join(CFG.src_dir, "source_training_results.json"))
        print("Reused source model trained in:", other_dir)
        return True
    return False


def get_or_train_source(loaders):
    """Return path to final-epoch source checkpoint (reuse or train if missing)."""
    ckpt_path = os.path.join(CFG.src_dir, "source_model_final.pth")
    if not os.path.exists(ckpt_path):
        _copy_shared_source(ckpt_path)
    if os.path.exists(ckpt_path):
        print("Found source checkpoint:", ckpt_path)
        # Still log the source-only accuracy if not present in JSON
        log_path = os.path.join(CFG.src_dir, "source_training_results.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
            has_entry = any(isinstance(e, dict) and "source_only_on_target" in e
                            for e in log)
            if not has_entry:
                model = ResNetBackbone(num_classes=CFG.num_classes,
                                       feat_dim=CFG.feat_dim,
                                       arch=CFG.backbone).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                src_only_acc = accuracy(model, loaders["tgt_test"])
                print(f"Source-only accuracy on {CFG.target_name} test: "
                      f"{src_only_acc:.2f}%")
                log.append({
                    "source_only_on_target": {
                        "target": CFG.target_name,
                        "accuracy": float(src_only_acc),
                    }
                })
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=2)
        return ckpt_path

    print("Training source model ...")
    model = ResNetBackbone(num_classes=CFG.num_classes,
                           feat_dim=CFG.feat_dim,
                           arch=CFG.backbone).to(DEVICE)
    model = train_source(
        model,
        loaders["src_train"],
        loaders["src_test"],
        target_test_loader=loaders["tgt_test"],   # ← pass it in
    )
    torch.save(model.state_dict(), ckpt_path)
    print("Saved final-epoch source model:", ckpt_path)
    return ckpt_path