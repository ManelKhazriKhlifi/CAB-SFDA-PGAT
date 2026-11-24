"""Unit tests for the core components (python -m pytest tests/ or python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cabsfda.config import load_config  # noqa: E402
from cabsfda.data import imbalanced_subset  # noqa: E402
from cabsfda.losses import consistency_loss, weighted_pl_loss  # noqa: E402
from cabsfda.memory import ClassMemory  # noqa: E402
from cabsfda.metrics import classification_metrics  # noqa: E402
from cabsfda.thresholds import FlexMatch, build_policy  # noqa: E402


def test_class_weights():
    m = ClassMemory(4, 8)
    assert torch.allclose(m.class_weights(), torch.ones(4))              # nothing accepted yet
    m.update(torch.randn(10, 8), torch.tensor([0] * 9 + [1]))
    w = m.class_weights()
    assert abs(float(w.mean()) - 1) < 1e-6
    assert torch.isclose(w[1] / w[0], torch.tensor(3.0))                 # sqrt(9 / 1)
    assert torch.isclose(w[2], w[1]) and torch.isclose(w[3], w[1])        # unseen classes = max seen weight
    assert m.counts.tolist() == [9, 1, 0, 0]


def test_memory_ema_and_init():
    m = ClassMemory(2, 3, momentum=0.9)
    m.update(torch.ones(2, 3), torch.tensor([0, 0]))
    assert torch.allclose(m.prototypes[0], torch.ones(3))                # first batch initialises
    m.update(torch.zeros(1, 3), torch.tensor([0]))
    assert torch.allclose(m.prototypes[0], torch.full((3,), 0.9))
    assert not m.initialized[1]


def test_pgat_range_and_tau_min_inert():
    cfg = load_config()
    torch.manual_seed(0)
    pols = []
    for tmin in (0.5, 0.8):
        c = cfg.copy()
        c.tau_min = tmin
        pols.append(build_policy(c, 5, "cpu", 100))
    mem = ClassMemory(5, 16)
    for _ in range(200):
        conf = torch.rand(64)
        pred = torch.randint(0, 5, (64,))
        feats = torch.randn(64, 16)
        mem.update(feats, pred)
        for p in pols:
            p.update(None, conf, pred, feats, mem)
    for p in pols:
        assert (p.tau >= 0.85 - 1e-6).all() and (p.tau <= 0.95 + 1e-6).all()
    assert torch.allclose(pols[0].tau, pols[1].tau)                      # tau_min never binds at gamma=0.1


def test_pgat_lag_flag():
    cfg = load_config({"threshold_lag": False})
    assert build_policy(cfg, 3, "cpu", 10).update_before_mask
    assert not build_policy(load_config(), 3, "cpu", 10).update_before_mask


def test_flexmatch_warmup():
    p = FlexMatch(3, "cpu", n_samples=10, p_cutoff=0.9)
    assert torch.allclose(p.tau, torch.zeros(3))                         # warm-up: everything accepted
    p.update(None, torch.tensor([0.95, 0.95, 0.5]), torch.tensor([0, 0, 1]), None, None, torch.tensor([0, 1, 2]))
    assert p.tau[0] > p.tau[1]


def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 5, 300))
    m = classification_metrics(p, y, 5)
    assert abs(m["oa"] - 100 * accuracy_score(y, p)) < 1e-9
    assert abs(m["balanced_acc"] - 100 * balanced_accuracy_score(y, p)) < 1e-9
    assert abs(m["macro_f1"] - 100 * f1_score(y, p, average="macro")) < 1e-9


def test_imbalanced_subset():
    labels = np.repeat(np.arange(7), 50)
    idx, counts, ranking = imbalanced_subset(labels, 7, 5.0, 40, seed=0)
    assert counts[0] == 40 and counts[6] == 8 and len(idx) == counts.sum()
    assert all(np.bincount(labels[idx], minlength=7) == counts)


def test_losses():
    zw, zs = torch.randn(8, 4), torch.randn(8, 4)
    assert torch.isclose(consistency_loss(zw, zs, 0.5), consistency_loss(zs, zw, 0.5))
    w = torch.tensor([1.0, 2.0, 1.0, 1.0])
    y = torch.tensor([1, 1, 0, 2])
    ce = torch.nn.functional.cross_entropy(zs[:4], y, reduction="none")
    assert torch.isclose(weighted_pl_loss(zs[:4], y, w), (ce * w[y]).mean())


def test_proto_loss_cosine_scale_free():
    m = ClassMemory(3, 8)
    f = torch.randn(6, 8)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    m.update(f, y)
    a = m.proto_loss(f, y, "cosine")
    m2 = ClassMemory(3, 8)
    m2.update(10 * f, y)
    assert torch.isclose(a, m2.proto_loss(10 * f, y, "cosine"), atol=1e-6)      # invariant to feature scale
    assert m2.proto_loss(10 * f, y, "mse") > 50 * m.proto_loss(f, y, "mse")     # MSE is not


def test_optimizer_groups_and_poly_schedule():
    from cabsfda.config import load_config
    from cabsfda.models import build_model
    from cabsfda.optim import Stepper, ema_update
    cfg = load_config({"arch": "resnet18", "pretrained": False, "amp": False, "epochs": 2})
    model = build_model(cfg, 4)
    st = Stepper(model, cfg, iters_per_epoch=5, device=torch.device("cpu"))
    lrs = {g["name"]: g["lr"] for g in st.opt.param_groups}
    assert np.isclose(lrs["backbone"], cfg.lr * cfg.backbone_lr_mult) and np.isclose(lrs["head"], cfg.lr)
    assert all(not p.requires_grad for p in model.classifier.parameters())       # frozen classifier
    x = torch.randn(4, 3, 32, 32)
    for _ in range(10):
        st.step(model(x).logsumexp(1).mean())
    assert np.isclose(st.lr_factor(), (1 + 10) ** -0.75)                         # end of the schedule
    teacher = build_model(cfg, 4)
    teacher.load_state_dict(model.state_dict())
    before = [p.clone() for p in teacher.parameters()]
    st.step(model(x).logsumexp(1).mean())
    ema_update(teacher, model, 0.9)
    moved = [not torch.equal(a, b) for a, b in zip(before, teacher.parameters())]
    assert any(moved)


def test_snd_detects_collapse():
    from cabsfda.metrics import unsupervised_scores
    rng = np.random.default_rng(0)
    good = np.eye(5)[rng.integers(0, 5, 200)] * 8 + rng.normal(0, 0.5, (200, 5))
    bad = np.zeros((200, 5))
    bad[:, 0] = 8
    bad += rng.normal(0, 0.5, (200, 5))
    sg, sb = unsupervised_scores(good), unsupervised_scores(bad)
    assert sg["pred_class_entropy"] > 0.9 and sb["pred_class_entropy"] < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
