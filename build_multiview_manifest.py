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

Output layout (one folder per rig; the rig id is in the path, not the file name):

    frames/rig_<r>/Image/<subcam>.png
    frames/rig_<r>/Depth/<subcam>.npy   (+ <subcam>.png viz)
    frames/rig_<r>/SurfaceNormal/<subcam>.npy   (+ <subcam>.png viz)
    frames/rig_<r>/camview/<subcam>.npz
    frames/rig_<r>/transforms.json      (file_path etc. relative to rig_<r>/)

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
    or None if missing. Handles the flat layout (frames/<Channel>/<prefix><suffix>)
    that harvest_multiview writes, as well as the datagen's nested
    frames/<Channel>/camera_<subcam>/ layout. Exact name first, then a glob."""
    prefix, exts = CHANNELS[channel]
    search_dirs = [frames / channel, frames / channel / f"camera_{subcam}"]
    for d in search_dirs:
        for ext in exts:
            cand = d / f"{prefix}{suffix}{ext}"
            if cand.exists():
                return cand.relative_to(frames).as_posix()
    for d in search_dirs:  # fallback: any file with the matching suffix
        if d.is_dir():
            for ext in exts:
                hits = sorted(d.glob(f"{prefix}{suffix}*{ext}"))
                if hits:
                    return hits[0].relative_to(frames).as_posix()
    return None


def _relocate_view(frames: Path, rig_dir: Path, channel: str, subcam: int, primary_rel):
    """Move this view's files for `channel` into rig_dir/<channel>/<subcam>.<ext>,
    including sibling files with the same stem (e.g. Depth's .npy raw + .png viz).
    Returns the primary file's path relative to rig_dir (for the manifest), or None.
    Renaming to the zero-padded sub-camera index moves the rig id out of the file
    name and into the folder, so names are no longer overloaded."""
    if primary_rel is None:
        return None
    src_primary = frames / primary_rel
    if not src_primary.exists():
        return None
    dstdir = rig_dir / channel
    dstdir.mkdir(parents=True, exist_ok=True)
    prim_out = None
    for sib in sorted(src_primary.parent.glob(src_primary.stem + ".*")):
        dst = dstdir / f"{subcam:02d}{sib.suffix}"
        shutil.move(str(sib), str(dst))
        if sib.suffix == src_primary.suffix:
            prim_out = f"{channel}/{subcam:02d}{sib.suffix}"
    return prim_out


def _anchor_near_frac(frames: Path, anchor_frame, near_dist=8.0, band=0.09):
    """Fraction of the anchor equirect-depth's HORIZONTAL band (`band` of the
    rows, centred on the equator) that lies within `near_dist` m -- a proxy for
    how much parallax-giving near content the view has. None if unavailable."""
    if anchor_frame is None or not anchor_frame.get("depth_path"):
        return None
    p = frames / anchor_frame["depth_path"]
    if p.suffix != ".npy" or not p.exists():
        return None
    try:
        d = np.load(p).astype(np.float32)
    except Exception:
        return None
    H = d.shape[0]
    lo, hi = int(H * (0.5 - band / 2)), int(H * (0.5 + band / 2))
    strip = d[lo:hi]
    valid = np.isfinite(strip) & (strip > 1e-3) & (strip < 1e4)
    if not valid.any():
        return None
    return round(float(np.mean(valid & (strip < near_dist))), 4)


def flatten_frames(frames: Path):
    """Flatten frames/<Channel>/camera_<s>/<file> -> frames/<Channel>/<file> so
    every layout is uniform. The harvester already writes flat; the datagen paths
    (multiview / urban) nest by sub-camera via reorganize_old_framesfolder. Suffixes
    already encode rig+subcam, so filenames stay unique. No-op on flat input."""
    if not frames.is_dir():
        return
    for channel in frames.iterdir():
        if not channel.is_dir():
            continue
        for camdir in list(channel.iterdir()):
            if camdir.is_dir() and camdir.name.startswith("camera_"):
                for f in camdir.iterdir():
                    dst = channel / f.name
                    if not dst.exists():
                        shutil.move(str(f), str(dst))
                if not any(camdir.iterdir()):
                    camdir.rmdir()


def build_scene_manifests(scene_dir: Path) -> list[Path]:
    """Build one transforms_camrig_<r>.json per camera rig in this scene.
    Returns the list of manifest paths written."""
    frames = scene_dir / "frames" if (scene_dir / "frames").is_dir() else scene_dir
    flatten_frames(frames)  # unify to the flat one-folder-per-modality layout
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
            # source paths (relative to frames/); rewritten to rig-relative once the
            # files are relocated into rig_<r>/ below.
            "_src_image": _find_channel_file(frames, "Image", s, suffix),
            "_src_depth": _find_channel_file(frames, "Depth", s, suffix),
            "_src_normal": _find_channel_file(frames, "SurfaceNormal", s, suffix),
            "_src_camview": cv.relative_to(frames).as_posix(),
            "transform_matrix": T.tolist(),
            "intrinsics": K.tolist(),
        }

        rig = rigs.setdefault(r, {"w": W, "h": H, "frames": []})
        rig["frames"].append(frame_entry)

    written = []
    for r, rig in sorted(rigs.items()):
        rig["frames"].sort(key=lambda e: (e["frame"], e["subcam"]))
        # Each rig gets its own folder so the rig id lives in the path, not the file
        # name: rig_<r>/{Image,Depth,SurfaceNormal,camview}/<subcam>.<ext> + one
        # transforms.json inside it (paths relative to the rig folder). This is
        # uniform for single- and multi-rig scenes (no transforms_camrig_<r>.json vs
        # transforms.json split) and matches the NeRF/GS "one folder per capture".
        rig_dir = frames / f"rig_{r}"
        rig_dir.mkdir(parents=True, exist_ok=True)
        for e in rig["frames"]:
            s = e["subcam"]
            e["file_path"] = _relocate_view(frames, rig_dir, "Image", s, e.pop("_src_image"))
            e["depth_path"] = _relocate_view(frames, rig_dir, "Depth", s, e.pop("_src_depth"))
            e["normal_path"] = _relocate_view(frames, rig_dir, "SurfaceNormal", s, e.pop("_src_normal"))
            cv_rel = e.pop("_src_camview", None)
            if cv_rel and (frames / cv_rel).exists():
                (rig_dir / "camview").mkdir(parents=True, exist_ok=True)
                shutil.move(str(frames / cv_rel), str(rig_dir / "camview" / f"{s:02d}.npz"))

        # Effective baseline for this scene = median anchor->neighbour distance.
        # Recorded so baseline-diverse datasets can be filtered / weighted by it.
        anchor_frame = next((f for f in rig["frames"] if f["is_anchor"]), None)
        anchor = np.array(anchor_frame["transform_matrix"]) if anchor_frame else None
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
            # Parallax proxy: fraction of the anchor's horizontal band within ~8 m.
            # Low -> a distant/empty view (little parallax); filter/weight on it.
            # Paths are now rig-relative, so read depth from rig_dir.
            "near_frac": _anchor_near_frac(rig_dir, anchor_frame),
            "frames": rig["frames"],
        }
        out = rig_dir / "transforms.json"
        out.write_text(json.dumps(manifest, indent=2))
        written.append(out)
        print(f"  wrote {out}  ({len(rig['frames'])} views)")

    # Remove the now-empty flat channel dirs left behind after relocation.
    for ch in list(CHANNELS) + ["camview"]:
        d = frames / ch
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

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


def prune_low_parallax(scene_dir: Path, min_near: float) -> int:
    """Drop rigs whose RENDERED anchor has too little near content.

    The placement-time raycast near_frac (panoramic_stats) systematically
    OVER-reads how much parallax-giving content a pose has: on a high vista it
    counts the peak's own downward-sloping ground as 'near' (~0.3) while the
    rendered equirect band is almost all distant, hazed terrain (~0.05). The
    manifest's `near_frac` is measured from the ACTUAL rendered anchor depth, so
    it's the ground truth. Rigs below `min_near` are empty/washed-out vistas
    (atmospheric aerial perspective on far terrain, no near geometry) -- exactly
    the 'sparse biome / washed-out' rigs we want to discard -- so remove the whole
    rig_<r> folder. `min_near <= 0` disables the check.
    """
    if not min_near or min_near <= 0:
        return 0
    frames = scene_dir / "frames" if (scene_dir / "frames").is_dir() else scene_dir
    removed = 0
    for rig_dir in sorted(frames.glob("rig_*")):
        mf = rig_dir / "transforms.json"
        if not (rig_dir.is_dir() and mf.exists()):
            continue
        try:
            nf = json.loads(mf.read_text()).get("near_frac")
        except Exception:
            nf = None
        # None => depth unavailable; keep it rather than silently dropping.
        if nf is not None and nf < min_near:
            shutil.rmtree(rig_dir)
            removed += 1
            print(f"  dropped {rig_dir.name}: near_frac={nf:.3f} < {min_near} (sparse/washed-out)")
    if removed:
        print(f"  pruned {removed} low-parallax rig(s) (rendered near_frac < {min_near})")
    return removed


def finalize_scene(scene_dir: Path, keep_files=("aerial_rigs.png",)) -> int:
    """Collapse a scene to the unified layout, IDENTICAL across indoor/outdoor/urban:

        <scene_dir>/rig_<k>/{Image,Depth,SurfaceNormal,camview}/<subcam>.<ext>
        <scene_dir>/rig_<k>/transforms.json

    All the infinigen-generation residue (the <hash>/ and frames/ nesting,
    coarse/fine/logs/Objects/etc.) is removed, and the rigs are re-numbered 0..N
    so the id in the path is contiguous. `keep_files` (e.g. the aerial map) are
    preserved at the scene root. Returns the number of rigs kept."""
    scene = Path(scene_dir)
    rig_dirs = sorted(
        (d for d in scene.rglob("rig_*")
         if d.is_dir() and (d / "transforms.json").exists()),
        key=lambda p: (str(p.parent), int(p.name.rsplit("_", 1)[-1])),
    )
    if not rig_dirs:
        return 0
    clean = scene.with_name(scene.name + "__clean")
    if clean.exists():
        shutil.rmtree(clean)
    clean.mkdir(parents=True)
    for k, rd in enumerate(rig_dirs):
        dst = clean / f"rig_{k}"
        shutil.move(str(rd), str(dst))
        # Re-stamp the manifest's rig id / scene so they match the new path.
        mf = dst / "transforms.json"
        try:
            d = json.loads(mf.read_text())
            d["cam_rig"] = k
            d["scene"] = scene.name
            mf.write_text(json.dumps(d, indent=2))
        except Exception:
            pass
    for name in keep_files:
        src = scene / name
        if src.exists():
            shutil.move(str(src), str(clean / name))
    shutil.rmtree(scene)
    clean.rename(scene)
    print(f"  finalized {scene} -> {len(rig_dirs)} rig(s), residue removed")
    return len(rig_dirs)


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
    # Collapse to the unified layout (<root>/rig_<k>/...) shared by all envs,
    # removing the infinigen <hash>/frames nesting + residue.
    finalize_root = args.scene if args.scene else args.root
    n_rigs = finalize_scene(finalize_root)
    print(f"Done: wrote {total} manifest(s); finalized {n_rigs} rig(s) under {finalize_root}.")


if __name__ == "__main__":
    main()
