"""Pure encoder functions for the Option C layout.

Tested at small scale on `vulus98/panoinfinigen-option-c-test` (20 indoor rows)
and on `prs-eth/ZuriPano` (100 outdoor LiDAR rows). The `encode_depth_viz_png`
implementation mirrors PaGeR inference's `prepare_depth_for_logging`:

    log(depth) -> median_filter(size=3) -> stretch to [0, 1] -> Spectral cmap.

Two robustness extensions for arbitrary input data:

1. Invalid pixels (`depth == 0`) are filled with the median valid log-depth
   *before* the median filter, so edges between valid and invalid regions
   don't pull the filter toward `log(0)`-equivalents.
2. The stretch uses 1st-99th percentile (rather than raw min/max) so a few
   outliers don't compress the colormap into a narrow band.

Invalid pixels are then forced to black in the final RGB output.
"""
from __future__ import annotations

import io

import matplotlib
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import median_filter

SPECTRAL = matplotlib.colormaps["Spectral"]


def encode_depth_png(depth_f32: np.ndarray, max_m: float) -> bytes:
    """Encode metric depth as 16-bit single-channel PNG with a fixed scale.

    Decode formula: ``np.asarray(img, np.float32) * max_m / 65535.0``.

    Args:
        depth_f32: ``(H, W)`` float depth in metres. May contain zeros for
            invalid pixels — they round-trip as zero.
        max_m: upper depth cap; depths above it are clipped.
    """
    scale = 65535.0 / max_m
    px = (np.clip(depth_f32, 0.0, max_m) * scale).round().astype(np.uint16)
    buf = io.BytesIO()
    PILImage.fromarray(px, mode="I;16").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def transcode_image_to_jpg(image_bytes: bytes, quality: int = 95) -> bytes:
    """Re-encode the source `image` cell as JPEG (default quality 95).

    Synthetic Infinigen renders are stored as ~7 MB PNG by the upstream
    pipeline. Transcoding to JPG q=95 shrinks them ~4-5x with no visible
    loss, which is what makes the HF Data Studio rows endpoint stay under
    the worker's 30 s timeout at length=100.
    """
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def encode_normals_jpg(normals_f32: np.ndarray, quality: int = 95) -> bytes:
    """Encode unit-normal field as 8-bit RGB JPEG.

    Maps ``[-1, 1]`` -> ``[0, 255]`` per channel and writes JPG (default
    q=95). The resulting image is directly viewable as a standard
    "normal map" in the HF data viewer.

    NaN pixels (sky / invalid regions in the source) are replaced with 0
    before the cast.

    Why JPG, not PNG: per-cell sizes on urban shards drop from ~8 MB PNG
    to ~2 MB JPG q=95, with ~1-degree directional error after decode
    (smaller than the float16 -> uint8 cast we accepted in the first place).
    See ``RESEARCH_REPORT.md`` next to this file for the full rationale.

    Decode formula: ``np.asarray(img, np.float32) / 127.5 - 1.0``.
    """
    if normals_f32.ndim != 3 or normals_f32.shape[-1] != 3:
        raise ValueError(f"expected normals shape (H, W, 3), got {normals_f32.shape}")
    arr = np.nan_to_num(normals_f32, nan=0.0, posinf=1.0, neginf=-1.0)
    px = np.clip((arr + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(px, mode="RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def encode_depth_viz_png(depth_f32: np.ndarray) -> bytes:
    """Spectral-colored log-depth preview (8-bit RGB PNG). Preview only."""
    valid = depth_f32 > 0
    if not valid.any():
        rgb = np.zeros((*depth_f32.shape, 3), dtype=np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=6)
        return buf.getvalue()

    log_d = np.empty_like(depth_f32, dtype=np.float32)
    log_d[valid] = np.log(depth_f32[valid])
    log_d[~valid] = float(np.median(log_d[valid]))  # fill before filtering
    log_d = median_filter(log_d, size=3)

    lo = float(np.percentile(log_d[valid], 1))
    hi = float(np.percentile(log_d[valid], 99))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((log_d - lo) / (hi - lo), 0.0, 1.0)

    rgba = SPECTRAL(norm.astype(np.float32))
    rgb = (rgba[..., :3] * 255).round().astype(np.uint8)
    rgb[~valid] = 0
    buf = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def decode_npy_or_npz(b: bytes) -> np.ndarray:
    """Decode a raw .npy or .npz blob and return the single array inside."""
    arr = np.load(io.BytesIO(b), allow_pickle=False)
    if hasattr(arr, "files"):  # NpzFile
        arr = arr[arr.files[0]]
    return arr
