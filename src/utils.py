"""Utilities: metrics, seeding, JSON I/O."""

import math
import random
import json
import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(total_steps, warmup_ratio=0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


@torch.no_grad()
def compute_metrics(logits, target, valid, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs > threshold) & (valid > 0.5)
    tb = (target > 0.5) & (valid > 0.5)
    tp = (pred & tb).sum().item()
    fp = (pred & (~tb)).sum().item()
    fn = ((~pred) & tb).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": iou, "precision": pr, "recall": rc, "f1": f1}


def confusion_at_threshold(preds, thr):
    tp = fp = fn = 0
    for d in preds:
        v = d["valid"]
        p = (d["prob"] >= thr) & v
        t = (d["target"] == 1) & v
        tp += int((p & t).sum())
        fp += int((p & ~t & v).sum())
        fn += int((~p & t).sum())
    return tp, fp, fn


def metrics_from_confusion(tp, fp, fn, eps=1e-9):
    iou = tp / (tp + fp + fn + eps)
    pr = tp / (tp + fp + eps)
    rc = tp / (tp + fn + eps)
    f1 = 2 * pr * rc / (pr + rc + eps)
    return {"iou": iou, "precision": pr, "recall": rc, "f1": f1}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
