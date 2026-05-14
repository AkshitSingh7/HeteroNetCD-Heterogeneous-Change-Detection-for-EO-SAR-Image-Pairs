"""Qualitative visualization of predictions."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_prediction(
    pre_rgb,
    post_sar,
    target,
    prob,
    pred,
    name: str,
    threshold: float,
    save_path: Path,
    idx: int = 0,
):
    """5-panel figure: EO, SAR, target, probability heatmap, binary prediction."""
    eo_vis = pre_rgb.transpose(1, 2, 0).astype(np.uint8) if pre_rgb.ndim == 3 else pre_rgb
    sar_vis = post_sar.astype(np.float32)
    sar_vis = (sar_vis - sar_vis.min()) / (sar_vis.max() - sar_vis.min() + 1e-6)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(eo_vis); axes[0].set_title("EO PRE"); axes[0].axis("off")
    axes[1].imshow(sar_vis, cmap="gray"); axes[1].set_title("SAR POST"); axes[1].axis("off")
    axes[2].imshow(target, cmap="gray"); axes[2].set_title("TARGET"); axes[2].axis("off")
    im_p = axes[3].imshow(prob, cmap="viridis", vmin=0, vmax=max(0.02, prob.max()))
    axes[3].set_title(f"PRED PROB (max={prob.max():.3f})")
    axes[3].axis("off")
    plt.colorbar(im_p, ax=axes[3], fraction=0.046)
    axes[4].imshow(pred, cmap="gray")
    axes[4].set_title(f"PRED @ {threshold:.3f}")
    axes[4].axis("off")
    plt.suptitle(f"{idx+1:03d} | {name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close()
