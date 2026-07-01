#!/usr/bin/env python
"""Assemble per-scene multi-view panorama sets into GS/NeRF-style transforms.json.

The multi-view pipeline (configs_nature/multiview.gin + data_schema/multiview.gin)
renders each scene from N sub-cameras of one rig, every sub-camera a full 360 deg
equirectangular panorama. Infinigen writes, per sub-camera `s` (and rig `r`,
frame `f`), files named with the suffix `_<r>_<resample>_<f>_<s>`:

    frames/Image/camera_<s>/Image<suffix>.png            # RGB panorama
    frames/Depth/camera_<s>/Depth<suffix>.npy            # metric depth (m)
    frames/SurfaceNormal/camera_<s>/SurfaceNormal<suffix>.npy
    frames/camview/camera_<s>/camview<suffix>.npz        # K, T, HW

This script groups all sub-cameras of a rig into one manifest with each view's
camera-to-world pose, so a Gaussian-Splatting head can be trained: pick one view
as the monocular input, supervise the predicted Gaussians against the others.

Usage:
    # process every scene under an output root (dirs that contain a frames/ dir)
    python build_multiview_manifest.py --root outputs/outdoor

    # or a single scene folder
    python build_multiview_manifest.py --scene outputs/outdoor/12345_1
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from infinigen.tools.suffixes import get_suffix, parse_suffix

# Directories we don't keep in the final multi-view dataset.
PRUNE_DIRS = ["Objects", "UniqueInstances", "imu_tum"]

# channel dir -> (filename prefix, preferred extensions in priority order)
CHANNELS = {
    "Image": ("Image", (".png", ".jpg", ".exr")),
    "Depth": ("Depth", (".npy", ".png")),
    "SurfaceNormal": ("SurfaceNormal", (".npy", ".png")),
}


def _find_channel_file(frames: Path, channel: str, subcam: int, suffix: str):
    """Return the path (relative to `frames`) of channel `channel` for this view,
    or None if missing. Tries the expected name first, then a glob fallback."""
    prefix, exts = CHANNELS[channel]
    cam_dir = frames / channel / f"camera_{subcam}"
    for ext in exts:
        cand = cam_dir / f"{prefix}{suffix}{ext}"
        if cand.exists():
            return cand.relative_to(frames).as_posix()
    # fallback: any file with the matching suffix in that camera dir
    if cam_dir.is_dir():
        for ext in exts:
            hits = sorted(cam_dir.glob(f"{prefix}{suffix}*{ext}"))
            if hits:
                return hits[0].relative_to(frames).as_posix()
    return None


def build_scene_manifests(scene_dir: Path) -> list[Path]:
    """Build one transforms_camrig_<r>.json per camera rig in this scene.
    Returns the list of manifest paths written."""
    frames = scene_dir / "frames" if (scene_dir / "frames").is_dir() else scene_dir
    camview_files = sorted(frames.glob("camview/**/camview_*.npz"))
    if not camview_files:
        print(f"  [skip] no camview npz under {frames}")
        return []

    rigs: dict[int, dict] = {}
    for cv in camview_files:
        idxs = parse_suffix(cv.name)
        if idxs is None:
            print(f"  [warn] could not parse suffix from {cv.name}")
            continue
        r, f, s = idxs["cam_rig"], idxs["frame"], idxs["subcam"]
        suffix = get_suffix(idxs)  # canonical "_<r>_<res>_<f>_<s>"

        data = np.load(cv)
        T = np.asarray(data["T"], dtype=np.float64)      # camera-to-world (Blender)
        K = np.asarray(data["K"], dtype=np.float64)
        H, W = (int(x) for x in np.asarray(data["HW"]).reshape(-1)[:2])

        frame_entry = {
            "subcam": s,
            "frame": f,
            "is_anchor": s == 0,  # sub-camera 0 is the rig anchor at (0,0,0)
            "file_path": _find_channel_file(frames, "Image", s, suffix),
            "depth_path": _find_channel_file(frames, "Depth", s, suffix),
            "normal_path": _find_channel_file(frames, "SurfaceNormal", s, suffix),
            "transform_matrix": T.tolist(),
            "intrinsics": K.tolist(),
        }

        rig = rigs.setdefault(r, {"w": W, "h": H, "frames": []})
        rig["frames"].append(frame_entry)

    single_rig = len(rigs) == 1
    written = []
    for r, rig in sorted(rigs.items()):
        rig["frames"].sort(key=lambda e: (e["frame"], e["subcam"]))
        # Effective baseline for this scene = median anchor->neighbour distance.
        # Recorded so baseline-diverse datasets can be filtered / weighted by it.
        anchor = next(
            (np.array(f["transform_matrix"]) for f in rig["frames"] if f["is_anchor"]),
            None,
        )
        dists = (
            [
                float(np.linalg.norm(np.array(f["transform_matrix"])[:3, 3] - anchor[:3, 3]))
                for f in rig["frames"]
                if not f["is_anchor"]
            ]
            if anchor is not None
            else []
        )
        manifest = {
            "camera_model": "EQUIRECTANGULAR",
            # transform_matrix maps camera -> world (Blender axes: camera looks
            # along local -Z with +Y up). intrinsics are the pinhole K Blender
            # reports and are not meaningful for the equirect mapping; use
            # (w, h) + longitude/latitude for panorama rays.
            "convention": "blender_cam_to_world",
            "scene": scene_dir.name,
            "cam_rig": r,
            "w": rig["w"],
            "h": rig["h"],
            "baseline_m": round(float(np.median(dists)), 4) if dists else 0.0,
            "frames": rig["frames"],
        }
        # One rig per scene is the common case -> a single transforms.json. Only
        # disambiguate by rig id when a scene actually has multiple rigs.
        name = "transforms.json" if single_rig else f"transforms_camrig_{r}.json"
        out = frames / name
        out.write_text(json.dumps(manifest, indent=2))
        written.append(out)
        print(f"  wrote {out}  ({len(rig['frames'])} views)")

    return written


def prune_scene(scene_dir: Path):
    """Remove artifacts not needed for the multi-view dataset: the Objects /
    UniqueInstances / imu_tum folders and any leftover .exr files (the RGB and
    ground-truth passes are already saved as PNG/NPY)."""
    frames = scene_dir / "frames" if (scene_dir / "frames").is_dir() else scene_dir
    removed = []
    for d in PRUNE_DIRS:
        p = frames / d
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(d)
    n_exr = 0
    for exr in frames.rglob("*.exr"):
        exr.unlink()
        n_exr += 1
    if removed or n_exr:
        print(f"  pruned {removed} + {n_exr} .exr file(s)")


def iter_scene_dirs(root: Path):
    """Yield every scene directory (one that contains a frames/ subfolder)."""
    if (root / "frames").is_dir():
        yield root
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "frames").is_dir():
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=Path, help="output root; process every scene under it")
    g.add_argument("--scene", type=Path, help="a single scene directory")
    ap.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="delete Objects/UniqueInstances/imu_tum folders and .exr files "
        "after building manifests (default: on; use --no-prune to keep them)",
    )
    args = ap.parse_args()

    scenes = [args.scene] if args.scene else list(iter_scene_dirs(args.root))
    if not scenes:
        raise SystemExit(f"No scenes with a frames/ folder found under {args.root}")

    total = 0
    for scene in scenes:
        print(f"[scene] {scene}")
        total += len(build_scene_manifests(scene))
        if args.prune:
            prune_scene(scene)
    print(f"Done: wrote {total} manifest(s) across {len(scenes)} scene(s).")


if __name__ == "__main__":
    main()
