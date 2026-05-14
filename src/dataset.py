"""HeteroChangeDataset: paired EO/SAR data loader."""

from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A

from .sar_preprocessing import create_3channel_sar


EO_MEAN = [0.485, 0.456, 0.406]
EO_STD = [0.229, 0.224, 0.225]


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read()


def list_basenames(split_root):
    return sorted(p.name for p in (Path(split_root) / "pre-event").glob("*.tif"))


def build_training_augmentations(config=None):
    """Build augmentation pipelines for training."""
    cfg = config or {}
    eo_color = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("eo_brightness_contrast_limit", 0.3),
                contrast_limit=cfg.get("eo_brightness_contrast_limit", 0.3),
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=cfg.get("eo_gamma_limit", (70, 130)), p=1.0),
        ], p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("eo_hue_shift", 15),
            sat_shift_limit=cfg.get("eo_saturation_shift", 20),
            val_shift_limit=cfg.get("eo_value_shift", 15),
            p=0.4,
        ),
        A.CLAHE(clip_limit=cfg.get("eo_clahe_clip", 2.0), p=0.3),
    ])
    sar_aug = A.RandomBrightnessContrast(
        brightness_limit=cfg.get("sar_brightness_limit", 0.15),
        contrast_limit=cfg.get("sar_contrast_limit", 0.10),
        p=0.5,
    )
    joint = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=cfg.get("geometric_shift_limit", 0.03),
            scale_limit=cfg.get("geometric_scale_limit", 0.05),
            rotate_limit=cfg.get("geometric_rotate_limit", 15),
            p=0.5,
            border_mode=0,
        ),
    ], additional_targets={
        "post": "image", "target": "mask", "valid": "mask",
    })
    return eo_color, sar_aug, joint


class HeteroChangeDataset(Dataset):
    """Paired EO/SAR dataset with mandatory binary label remap.

    Output per item:
        pre:    (3, H, W) float32, normalized EO (ImageNet stats)
        post:   (3, H, W) float32, normalized 3-ch SAR
        target: (1, H, W) float32, binary {0, 1}
        valid:  (1, H, W) float32, valid pixel mask
        name:   str
    """

    def __init__(
        self,
        split_root,
        sar_mean,
        sar_std,
        augment=False,
        aug_config=None,
    ):
        self.split_root = Path(split_root)
        self.names = list_basenames(self.split_root)
        self.augment = augment

        self.normalize_eo = A.Normalize(mean=EO_MEAN, std=EO_STD, max_pixel_value=255.0)
        self.normalize_sar = A.Normalize(mean=sar_mean, std=sar_std, max_pixel_value=1.0)
        if augment:
            self.eo_color, self.sar_aug, self.joint = build_training_augmentations(aug_config)
        else:
            self.eo_color = self.sar_aug = self.joint = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        pre_rgb = read_tif(self.split_root / "pre-event" / name)
        post_raw = read_tif(self.split_root / "post-event" / name)[0]
        target_raw = read_tif(self.split_root / "target" / name)[0]

        valid = ((pre_rgb.sum(0) > 0) & (post_raw > 0)).astype(np.uint8)
        # Mandatory binary remap: {0,1}->0, {2,3}->1
        tgt_bin = (target_raw >= 2).astype(np.uint8)

        pre_hwc = pre_rgb.transpose(1, 2, 0).astype(np.uint8)

        if self.augment:
            post_aug = self.sar_aug(image=post_raw.astype(np.uint8))["image"]
            post_raw_use = post_aug.astype(np.float32)
        else:
            post_raw_use = post_raw

        post_3ch = create_3channel_sar(post_raw_use)

        if self.augment:
            pre_hwc = self.eo_color(image=pre_hwc)["image"]
            aug = self.joint(
                image=pre_hwc, post=post_3ch,
                target=tgt_bin, valid=valid,
            )
            pre_hwc = aug["image"]
            post_3ch = aug["post"]
            tgt_bin = aug["target"]
            valid = aug["valid"]

        pre_norm = self.normalize_eo(image=pre_hwc)["image"]
        post_norm = self.normalize_sar(image=post_3ch / 255.0)["image"]

        return {
            "pre":    torch.from_numpy(pre_norm.transpose(2, 0, 1)).float(),
            "post":   torch.from_numpy(post_norm.transpose(2, 0, 1)).float(),
            "target": torch.from_numpy(tgt_bin).unsqueeze(0).float(),
            "valid":  torch.from_numpy(valid).unsqueeze(0).float(),
            "name":   name,
        }
