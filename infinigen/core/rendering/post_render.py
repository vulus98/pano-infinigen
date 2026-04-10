# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import logging
import os

import OpenEXR
import Imath # Needed for robust EXR loading

# ruff: noqa: E402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.

import colorsys
from pathlib import Path

import cv2
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


_REORIENT_REFERENCE_C2W = np.array(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _get_relative_yaw_deg(c2w_1: np.ndarray, c2w_2: np.ndarray) -> float:
    R1 = c2w_1[:3, :3]
    R2 = c2w_2[:3, :3]

    R_rel = R1.T @ R2

    pitch = np.arcsin(np.clip(R_rel[1, 2], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-6:
        yaw = np.arctan2(-R_rel[0, 2], R_rel[2, 2])
    else:
        yaw = 0.0

    return float(np.degrees(yaw))


def roll_normals_equirect(normals: np.ndarray, shift_x: float) -> np.ndarray:
    if normals.ndim != 3:
        raise ValueError(f"Expected normals to have 3 dims (H,W,3 or 3,H,W), got {normals.shape}")

    angle = -2.0 * np.pi * shift_x
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array(
        [[cos_a, 0.0, -sin_a], [0.0, 1.0, 0.0], [sin_a, 0.0, cos_a]],
        dtype=normals.dtype,
    )

    if normals.shape[-1] == 3:
        # (H, W, 3) case - native layout processing for performance
        H, W, _ = normals.shape
        # Flatten to (N, 3). No copy if input is contiguous.
        n_flat = normals.reshape(-1, 3) 
        
        # v' = R v. For row vectors: v' = v R.T
        rotated_flat = n_flat @ R.T
        
        # Result is (H, W, 3) contiguous
        return rotated_flat.reshape(H, W, 3)

    elif normals.shape[0] == 3:
         # (3, H, W) case
        _, H, W = normals.shape
        n_flat = normals.reshape(3, -1)
        rotated = (R @ n_flat).reshape(3, H, W)
        return rotated
    else:
        raise ValueError(f"Expected last or first dim to be 3, got {normals.shape}")


def reorient_surface_normals_from_camview(
    surface_normals: np.ndarray, camview_T: np.ndarray
) -> np.ndarray:
    if camview_T.shape != (4, 4):
        raise ValueError(f"Expected camview_T shape (4,4), got {camview_T.shape}")

    # Match the convention used in reorient_normals.py: convert Blender C2W to an OpenCV-like frame.
    yaw_deg = _get_relative_yaw_deg(camview_T, _REORIENT_REFERENCE_C2W)
    shift_x = yaw_deg / 360.0
    return roll_normals_equirect(surface_normals, -shift_x)


def load_exr(path):
    path = str(path)
    if not (Path(path).exists() and Path(path).suffix == ".exr"):
        raise ValueError(f"Invalid EXR path: {path}")

    # Robust loading using OpenEXR if possible
    try:
        import OpenEXR
        import Imath
        exr_file = OpenEXR.InputFile(path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        channels = list(header['channels'].keys())

        # Decide which channels to read (Normal maps are often XYZ, Colors RGB)
        # We want to emulate OpenCV's BGR return format if possible for consistency.
        # Blender 5.0 multilayer EXR uses dotted layer names like "Normal.X" /
        # "ViewLayer.Normal.X" / "Image.R" etc., so we also try a suffix match.
        def _find_channels(suffix_groups):
            """Return channel names matching the requested suffix list, or None."""
            for suffixes in suffix_groups:
                matched = []
                for s in suffixes:
                    found = next((c for c in channels if c == s or c.endswith("." + s)), None)
                    if found is None:
                        matched = None
                        break
                    matched.append(found)
                if matched is not None:
                    return matched
            return None

        # Try RGB first (returned as BGR), then XYZ (returned as ZYX), with
        # multilayer-aware suffix matching for both.
        C = _find_channels([
            ['B', 'G', 'R'],
            ['Z', 'Y', 'X'],
        ])
        if C is None:
            # Fallback to OpenCV standard (may fail on multilayer EXR)
            return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        arrs = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(height, width) for c in C]
        return np.dstack(arrs)

    except Exception as e:
        logger.warning(f"OpenEXR direct load failed ({e}), falling back to OpenCV for {path}")
        return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def load_single_channel(p):
    import OpenEXR
    import Imath
    file = OpenEXR.InputFile(str(p))
    channel, channel_type = next(iter(file.header()["channels"].items()))
    match str(channel_type.type):
        case "FLOAT":
            np_type = np.float32
        case _:
            np_type = np.uint8
    data = np.frombuffer(file.channel(channel, channel_type.type), np_type)
    dw = file.header()["dataWindow"]
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    return data.reshape(sz)


def load_depth(p):
    return load_single_channel(p)


def load_normals(p):
    # Depending on how load_exr reads channels, this reordering is important
    # If load_exr returns BGR (Z, Y, X), then:
    # 2 -> X
    # 0 -> Z
    # 1 -> Y
    # load_exr(p)[..., [2, 0, 1]] -> [X, Z, Y]
    # * [-1, 1, 1] -> [-X, Z, Y]
    return load_exr(p)[..., [2, 0, 1]] * np.array([-1.0, 1.0, 1.0])


def load_seg_mask(p):
    return load_single_channel(p).astype(np.int64)


def load_uniq_inst(p):
    return load_exr(p).view(np.int32)


def colorize_flow(optical_flow):
    try:
        import flow_vis
    except ImportError:
        logger.warning(
            "Flow visualization requires the 'flow_vis' package. Please install via `pip install .[vis]."
        )
        return None

    flow_uv = optical_flow[..., :2]
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_color


def colorize_normals(surface_normals):
    assert surface_normals.max() < 1 + 1e-4
    assert surface_normals.min() > -1 - 1e-4
    norm = np.linalg.norm(surface_normals, axis=2)
    color = np.round((surface_normals + 1) * (255 / 2)).astype(np.uint8)
    color[norm < 1e-4] = 0
    return color


def colorize_depth(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[..., :3] * 255, dtype=np.uint8)


def colorize_int_array(data, color_seed=0):
    H, W, *_ = data.shape
    data = data.reshape((H * W, -1))
    uniq, indices = np.unique(data, return_inverse=True, axis=0)
    random_states = [
        np.random.RandomState(e[:2].astype(np.uint32) + color_seed) for e in uniq
    ]
    unique_colors = (
        np.asarray(
            [
                colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)
                for s in random_states
            ]
        )
        * 255
    ).astype(np.uint8)
    return unique_colors[indices].reshape((H, W, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_path", type=Path, default=None)
    parser.add_argument("--depth_path", type=Path, default=None)
    parser.add_argument("--seg_path", type=Path, default=None)
    parser.add_argument("--uniq_inst_path", type=Path, default=None)
    parser.add_argument("--normals_path", type=Path, default=None)
    args = parser.parse_args()

    if args.flow_path is not None:
        try:
            flow_color = colorize_flow(load_flow(args.flow_path))
            if flow_color is not None:
                output_path = args.flow_path.with_suffix(".png")
                imwrite(output_path, flow_color)
                print(f"Wrote {output_path}")
        except ModuleNotFoundError:
            print(
                "Flow visualization requires the 'flow_vis' package. Install it with 'pip install flow_vis'"
            )
            pass

    if args.normals_path is not None:
        normal_color = colorize_normals(load_normals(args.normals_path))
        output_path = args.normals_path.with_suffix(".png")
        imwrite(output_path, normal_color)
        print(f"Wrote {output_path}")

    if args.depth_path is not None:
        depth_color = colorize_depth(load_depth(args.depth_path))
        output_path = args.depth_path.with_suffix(".png")
        imwrite(output_path, depth_color)
        print(f"Wrote {output_path}")

    if args.uniq_inst_path is not None:
        mask_color = colorize_int_array(load_uniq_inst(args.uniq_inst_path))
        output_path = args.uniq_inst_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")

    if args.seg_path is not None:
        mask_color = colorize_int_array(load_seg_mask(args.seg_path))
        output_path = args.seg_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        print(f"Wrote {output_path}")
