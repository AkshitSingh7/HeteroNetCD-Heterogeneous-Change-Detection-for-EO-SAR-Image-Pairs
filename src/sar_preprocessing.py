"""SAR preprocessing: Lee filter + 3-channel representation."""

import numpy as np
from scipy import ndimage


def apply_lee_filter(img_uint8: np.ndarray, size: int = 7,
                     look_factor: float = 1.0) -> np.ndarray:
    """Apply Lee speckle filter to single-channel SAR."""
    img = img_uint8.astype(np.float32)
    mean = ndimage.uniform_filter(img, size=size)
    sq = ndimage.uniform_filter(img ** 2, size=size)
    var = np.maximum(sq - mean ** 2, 0)
    overall_mean = mean.mean()
    noise_var = look_factor * overall_mean ** 2
    weight = (var - noise_var).clip(min=0) / np.maximum(var, 1e-6)
    out = mean + weight * (img - mean)
    return np.clip(out, 0, 255).astype(np.float32)


def create_3channel_sar(raw: np.ndarray) -> np.ndarray:
    """Stack [raw, Lee-filtered, local-texture] into a 3-channel SAR input.

    raw: HxW uint8 or float32, single-channel SAR
    returns: HxWx3 float32
    """
    raw = raw.astype(np.float32)
    lee = apply_lee_filter(raw)
    local_mean = ndimage.uniform_filter(lee, (7, 7))
    local_sq = ndimage.uniform_filter(lee ** 2, (7, 7))
    texture = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    texture = np.clip(texture * 5.0, 0, 255)
    return np.stack([raw, lee, texture], axis=-1)
