"""Train HeteroNetCD on EO-SAR change detection data. this script reproduces training
from scratch and requires the SARhub SAR-MoCo weights.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import rasterio

from src.dataset import HeteroChangeDataset
from src.model import HeteroNetCD
from src.sar_preprocessing import create_3channel_sar
from src.utils import (
    set_seed, cosine_with_warmup, compute_metrics, save_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to dataset root containing train/, val/, test/")
    p.add_argument("--sar_weights", type=str, default="./sarhub_resnet50.pt",
                   help="Path to SARhub MoCo ResNet50 weights")
    p.add_argument("--output_dir", type=str, default="./runs/heteronetcd")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def compute_sar_stats(train_root):
    """Compute SAR channel statistics from the training split."""
    train_root = Path(train_root)
    names = sorted(p.name for p in (train_root / "pre-event").glob("*.tif"))
    sums = np.zeros(3); sqs = np.zeros(3); count = 0
    for name in tqdm(names, desc="computing SAR stats"):
        with rasterio.open(train_root / "pre-event" / name) as src:
            pre = src.read()
        with rasterio.open(train_root / "post-event" / name) as src:
            post = src.read()[0]
        valid = (pre.sum(0) > 0) & (post > 0)
        sar = create_3channel_sar(post) / 255.0
        v = sar[valid]
        sums += v.sum(0)
        sqs += (v ** 2).sum(0)
        count += len(v)
    mean = sums / count
    var = sqs / count - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    return {"sar_mean": mean.tolist(), "sar_std": std.tolist()}


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAR stats: load or compute
    stats_path = output_dir / "sar3_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            sar_stats = json.load(f)
    else:
        print("[Computing SAR stats from training data...]")
        sar_stats = compute_sar_stats(Path(args.data_path) / "train")
        with open(stats_path, "w") as f:
            json.dump(sar_stats, f, indent=2)
    print(f"SAR stats: {sar_stats}")

    # Augmentation config
    aug_config = {
        "eo_brightness_contrast_limit": 0.3,
        "eo_gamma_limit": (70, 130),
        "eo_hue_shift": 15,
        "eo_saturation_shift": 20,
        "eo_value_shift": 15,
        "eo_clahe_clip": 2.0,
        "sar_brightness_limit": 0.15,
        "sar_contrast_limit": 0.10,
        "geometric_shift_limit": 0.03,
        "geometric_scale_limit": 0.05,
        "geometric_rotate_limit": 15,
    }

    # Data
    train_ds = HeteroChangeDataset(
        split_root=Path(args.data_path) / "train",
        sar_mean=sar_stats["sar_mean"], sar_std=sar_stats["sar_std"],
        augment=True, aug_config=aug_config,
    )
    val_ds = HeteroChangeDataset(
        split_root=Path(args.data_path) / "val",
        sar_mean=sar_stats["sar_mean"], sar_std=sar_stats["sar_std"],
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroNetCD(
        sar_weights_path=args.sar_weights,
        eo_weights="ssl",                    # TorchGeo SSL Sentinel-2
        decoder_channels=(256, 128, 64, 32, 16),
        seg_head_weight_std=1.0,
        seg_head_bias=-2.0,
        use_attention=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HeteroNetCD: {n_params/1e6:.1f}M parameters]")

    # Loss
    from torch.nn.functional import binary_cross_entropy_with_logits
    loss_cfg = {
        "focal_bce_weight": 0.5, "focal_bce_gamma": 2.0, "focal_bce_pos_weight": 2.0,
        "focal_tversky_weight": 0.5, "focal_tversky_alpha": 0.5,
        "focal_tversky_beta": 0.5, "focal_tversky_gamma": 1.0,
    }

    def focal_bce(logits, target, valid, gamma=2.0, pos_weight=2.0):
        pw = torch.tensor([pos_weight], device=logits.device)
        bce = binary_cross_entropy_with_logits(logits.float(), target.float(),
                                                reduction="none", pos_weight=pw)
        p = torch.sigmoid(logits.float())
        pt = p * target + (1 - p) * (1 - target)
        focal = ((1 - pt).clamp(min=1e-6) ** gamma) * bce
        return (focal * valid).sum() / valid.sum().clamp(min=1)

    def focal_tversky(logits, target, valid, alpha=0.5, beta=0.5, gamma=1.0):
        probs = torch.sigmoid(logits.float()) * valid
        target = target * valid
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return (1 - tversky) ** gamma

    def total_loss(logits, target, valid):
        fb = focal_bce(logits, target, valid,
                       gamma=loss_cfg["focal_bce_gamma"],
                       pos_weight=loss_cfg["focal_bce_pos_weight"])
        ft = focal_tversky(logits, target, valid,
                           alpha=loss_cfg["focal_tversky_alpha"],
                           beta=loss_cfg["focal_tversky_beta"],
                           gamma=loss_cfg["focal_tversky_gamma"])
        return loss_cfg["focal_bce_weight"] * fb + loss_cfg["focal_tversky_weight"] * ft

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(total_steps, 0.05),
    )

    use_bf16 = torch.cuda.is_available()
    history = []
    best_val_iou = 0.0
    start_epoch = 1

    # Resume
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        history = ck.get("history", [])
        best_val_iou = ck.get("best_iou", 0.0)
        start_epoch = ck["epoch"] + 1
        for _ in range((start_epoch - 1) * len(train_loader)):
            scheduler.step()
        print(f"[Resumed from epoch {start_epoch}, best val IoU {best_val_iou:.4f}]")

    ckpt_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"E{epoch}/{args.epochs} train")
        for batch in pbar:
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(pre, post, valid)
                loss = total_loss(logits, target, valid)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{np.mean(train_losses[-50:]):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Validation
        model.eval()
        tp = fp = fn = 0
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"E{epoch}/{args.epochs} val"):
                pre = batch["pre"].to(device)
                post = batch["post"].to(device)
                target = batch["target"].to(device)
                valid = batch["valid"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(pre, post, valid)
                    vl = total_loss(logits, target, valid)
                val_losses.append(vl.item())
                stats = compute_metrics(logits, target, valid, threshold=0.5)
                tp += stats["tp"]; fp += stats["fp"]; fn += stats["fn"]

        iou = tp / max(tp + fp + fn, 1)
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.time() - t0

        print(f"E{epoch:02d} | train={train_loss:.4f} val={val_loss:.4f} "
              f"| IoU={iou:.4f} P={pr:.4f} R={rc:.4f} F1={f1:.4f} | {elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "iou": iou, "precision": pr, "recall": rc, "f1": f1, "time": elapsed,
        })

        if iou > best_val_iou:
            best_val_iou = iou
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "history": history,
                "best_iou": best_val_iou,
                "sar_stats": sar_stats,            # bundled in checkpoint
                "default_threshold": 0.005,        # known good for cross-event
                "config": {
                    "model_name": "HeteroNetCD",
                    "decoder_channels": [256, 128, 64, 32, 16],
                    "cross_modal_attention": True,
                    "seg_head_weight_std": 1.0,
                    "seg_head_bias": -2.0,
                    "loss": loss_cfg,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio": 0.05,
                    "seed": args.seed,
                },
            }, ckpt_path)
            print(f"  ✓ saved best (val IoU {best_val_iou:.4f})")

        save_json(history, history_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("="*70)
    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("="*70)


if __name__ == "__main__":
    main()
