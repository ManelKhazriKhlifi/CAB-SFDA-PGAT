import os
import json
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
    amp_on = CFG.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    log = []
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Source ep {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_on):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc_src = accuracy(model, test_loader)
        print(f"[Source] Epoch {ep}: source test acc = {acc_src:.2f}%")
        log.append({"epoch": ep, "source_test_acc": float(acc_src)})

    # ---- Source-only accuracy on TARGET test set (this is the line you want) ----
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
        log_path = os.path.join(CFG.out_dir, "source_training_results.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print("Saved source-training results:", log_path)
    return model


def get_or_train_source(loaders):
    """Return path to final-epoch source checkpoint (train if missing)."""
    ckpt_path = os.path.join(CFG.out_dir, "source_model_final.pth")
    if os.path.exists(ckpt_path):
        print("Found source checkpoint:", ckpt_path)
        # Still log the source-only accuracy if not present in JSON
        log_path = os.path.join(CFG.out_dir, "source_training_results.json")
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