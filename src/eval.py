"""Evaluate HeteroNetCD on a held-out split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import albumentations as A

from src.dataset import read_tif, list_basenames, EO_MEAN, EO_STD
from src.sar_preprocessing import create_3channel_sar
from src.model import HeteroNetCD
from src.utils import (
    confusion_at_threshold, metrics_from_confusion, save_json, set_seed,
)
from src.visualize import visualize_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to dataset root, or a directory containing "
                        "pre-event/, post-event/, target/ subdirs")
    p.add_argument("--split", type=str, default=None,
                   help="Optional split subdirectory name (e.g. 'test'). If given, "
                        "data_path/split is used.")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to HeteroNetCD checkpoint")
    p.add_argument("--output_dir", type=str, default="./eval_output")
    p.add_argument("--threshold", type=float, default=None,
                   help="Threshold for binary prediction. If not given, "
                        "uses default from checkpoint config.")
    p.add_argument("--visualize", action="store_true",
                   help="Write per-image qualitative visualizations.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (auto-detect by default)")
    return p.parse_args()


def resolve_data_root(data_path, split):
    """Find the directory containing pre-event/, post-event/, target/."""
    root = Path(data_path)
    if split:
        root = root / split
    if not (root / "pre-event").is_dir():
        raise FileNotFoundError(
            f"Expected pre-event/ subdirectory under {root}. "
            f"Pass --data_path correctly, or use --split <name> if the data "
            f"lives in a split subfolder.")
    return root


@torch.no_grad()
def run_inference(model, data_root, sar_mean, sar_std, device):
    normalize_eo = A.Normalize(mean=EO_MEAN, std=EO_STD, max_pixel_value=255.0)
    normalize_sar = A.Normalize(mean=sar_mean, std=sar_std, max_pixel_value=1.0)
    names = list_basenames(data_root)
    preds = []
    for name in tqdm(names, desc="inference"):
        pre_rgb = read_tif(data_root / "pre-event" / name)
        post_raw = read_tif(data_root / "post-event" / name)[0]
        tgt = read_tif(data_root / "target" / name)[0]
        valid_np = (pre_rgb.sum(0) > 0) & (post_raw > 0)
        target_bin = (tgt >= 2).astype(np.uint8)  # mandatory binary remap

        post_3ch = create_3channel_sar(post_raw)
        pre_hwc = pre_rgb.transpose(1, 2, 0).astype(np.uint8)
        pre_norm = normalize_eo(image=pre_hwc)["image"]
        post_norm = normalize_sar(image=post_3ch / 255.0)["image"]

        pre_t = torch.from_numpy(pre_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        post_t = torch.from_numpy(post_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        vmsk_t = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        use_bf16 = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(pre_t, post_t, vmsk_t)
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        prob[~valid_np] = 0
        preds.append({
            "name": name, "prob": prob,
            "target": target_bin, "valid": valid_np,
            "pre_rgb": pre_rgb, "post_raw": post_raw,
        })
    return preds


def scene_id(name):
    return "_".join(name.split("_")[:2])


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device: {device}]")

    # Load checkpoint
    print(f"[Loading checkpoint: {args.weights}]")
    ck = torch.load(args.weights, map_location=device, weights_only=False)

    # Pull metadata from checkpoint
    ck_config = ck.get("config", {})
    sar_stats = ck.get("sar_stats", None)
    if sar_stats is None:
        raise ValueError(
            "Checkpoint missing SAR normalization stats. "
            "Re-export the checkpoint with sar_stats included."
        )
    default_threshold = ck.get("default_threshold", 0.005)
    seed = ck_config.get("seed", 42)
    set_seed(seed)

    # Build model and load weights
    decoder_channels = tuple(ck_config.get("decoder_channels", [256, 128, 64, 32, 16]))
    use_attention = ck_config.get("cross_modal_attention", True)
    model = HeteroNetCD(
        sar_weights_path=None,        # not needed for inference
        eo_weights=None,              # weights come from checkpoint
        decoder_channels=decoder_channels,
        seg_head_weight_std=ck_config.get("seg_head_weight_std", 1.0),
        seg_head_bias=ck_config.get("seg_head_bias", -2.0),
        use_attention=use_attention,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[Loaded model: epoch {ck.get('epoch', '?')}, "
          f"best val IoU {ck.get('best_iou', '?')}]")

    # Resolve data
    data_root = resolve_data_root(args.data_path, args.split)
    print(f"[Data root: {data_root}]")

    # Inference
    preds = run_inference(
        model, data_root, sar_stats["sar_mean"], sar_stats["sar_std"], device,
    )
    print(f"[{len(preds)} images processed]")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Threshold sweep
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                  0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    print("\n[Threshold sweep:]")
    sweep = {}
    for thr in thresholds:
        tp, fp, fn = confusion_at_threshold(preds, thr)
        m = metrics_from_confusion(tp, fp, fn)
        sweep[f"{thr:.3f}"] = m
        print(f"  thr={thr:.3f}: IoU={m['iou']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}")
    best_thr_str = max(sweep, key=lambda t: sweep[t]["iou"])

    # Per-scene
    per_scene = {}
    for d in preds:
        per_scene.setdefault(scene_id(d["name"]), []).append(d)
    per_scene_metrics = {}
    if len(per_scene) > 1:
        print(f"\n[Per-scene IoU @ best thr {best_thr_str}:]")
        for sid, items in per_scene.items():
            tp, fp, fn = confusion_at_threshold(items, float(best_thr_str))
            m = metrics_from_confusion(tp, fp, fn)
            per_scene_metrics[sid] = m
            print(f"  {sid}: n={len(items)} IoU={m['iou']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}")

    # Print main result at default threshold
    eval_threshold = args.threshold if args.threshold is not None else default_threshold
    eval_threshold_str = f"{eval_threshold:.3f}"
    if eval_threshold_str not in sweep:
        tp, fp, fn = confusion_at_threshold(preds, eval_threshold)
        sweep[eval_threshold_str] = metrics_from_confusion(tp, fp, fn)
    m_eval = sweep[eval_threshold_str]
    print(f"\n[Headline result @ threshold {eval_threshold:.3f}:]")
    print(f"  IoU       = {m_eval['iou']:.4f}")
    print(f"  Precision = {m_eval['precision']:.4f}")
    print(f"  Recall    = {m_eval['recall']:.4f}")
    print(f"  F1        = {m_eval['f1']:.4f}")
    print(f"\n[Best threshold on this split: {best_thr_str} "
          f"(IoU {sweep[best_thr_str]['iou']:.4f})]")

    # Save metrics
    summary = {
        "weights": str(args.weights),
        "data_root": str(data_root),
        "default_threshold_from_checkpoint": default_threshold,
        "evaluation_threshold": eval_threshold,
        "metrics_at_eval_threshold": m_eval,
        "threshold_sweep": sweep,
        "best_threshold_on_this_split": best_thr_str,
        "best_metrics_on_this_split": sweep[best_thr_str],
        "per_scene_metrics": per_scene_metrics,
        "num_images": len(preds),
    }
    save_json(summary, output_dir / "metrics.json")
    print(f"\n[Saved metrics: {output_dir / 'metrics.json'}]")

    # Visualization
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"\n[Writing visualizations at threshold {eval_threshold} to {vis_dir}]")
        for idx, d in enumerate(tqdm(preds, desc="viz")):
            pred = (d["prob"] >= eval_threshold).astype(np.uint8)
            visualize_prediction(
                pre_rgb=d["pre_rgb"],
                post_sar=d["post_raw"],
                target=d["target"],
                prob=d["prob"],
                pred=pred,
                name=d["name"],
                threshold=eval_threshold,
                save_path=vis_dir / f"{idx+1:03d}_{d['name'].replace('.tif','')}.png",
                idx=idx,
            )


if __name__ == "__main__":
    main()
