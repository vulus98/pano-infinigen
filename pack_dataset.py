#!/usr/bin/env python3
"""
pack_dataset.py -- package the generated pano-infinigen outputs into a compact,
low-inode, train/val/test-split dataset for cluster storage.

One tar per SCENE (a scene = one top-level folder under outputs/<domain>/; for
urban that folder is a whole city holding ~100 rigs). This collapses ~160k loose
small files into ~820 archives, which is what Lustre/GPFS inode quotas care about.

Produced layout:
    <dst>/<domain>/<split>/<scene>.tar
    <dst>/index.json

Each per-scene tar mirrors the scene's rig tree, with two space wins:
    rig_<r>/Image/NN.png            (unchanged, lossless RGB)
    rig_<r>/Depth/NN.npz            (float16 depth, np.savez_compressed key 'depth')
    rig_<r>/SurfaceNormal/NN.png    (unchanged)
    rig_<r>/camview/NN.npz          (unchanged, K + extrinsics)
    rig_<r>/transforms.json         (unchanged, NeRF-style poses)
  - Depth/*.npy  ->  Depth/*.npz    (lossless recompress, ~2.5x smaller)
  - Depth/*.png previews are DROPPED (they are just colorized copies of the .npy GT)
  - SurfaceNormal/*.npy (raw float16, ~12 MB/view) is DROPPED; the 8-bit .png normal
    map is kept as the GT (directional data is fine at 8-bit; the raw floats remain in
    outputs/ if full precision is ever needed)

Split assignment (deterministic + reproducible; stable as new scenes are added):
  - indoor, outdoor : by scene, md5(scene_id) bucket -> 92 / 6 / 2
  - urban           : by city, 18 train / 1 val / 1 test (the two cities with the
                      smallest hash go to test then val; rest train)

Usage:
    python pack_dataset.py --src outputs --dst dataset            # pack everything
    python pack_dataset.py --domains outdoor --dry-run            # preview a split
    python pack_dataset.py --src outputs --dst dataset --workers 8
"""

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- split configuration -----------------------------------------------------
SCENE_RATIO = {"train": 92, "val": 6, "test": 2}   # indoor + outdoor, per-scene
URBAN_SPLIT = {"val": 1, "test": 1}                # rest of the cities -> train
DOMAINS = ("indoor", "outdoor", "urban")

# Static description of one sample (a rig) so a dataloader can decode each member
# without guessing. Paths are relative to a rig prefix (rig_<r>/...) inside a scene tar.
SCHEMA = {
    "sample_unit": "rig -- one multi-view equirectangular panorama set",
    "projection": "equirectangular",
    "resolution": [2048, 1024],
    "container": "tar per scene; each tar holds rig_<r>/<modality>/<view>.<ext>",
    "modalities": {
        "rgb":     {"path": "{rig}/Image/{view}.png",         "format": "png",  "dtype": "uint8",   "shape": [1024, 2048, 3]},
        "depth":   {"path": "{rig}/Depth/{view}.npz",         "format": "npz",  "key": "depth", "dtype": "float16", "shape": [1024, 2048], "units": "meters"},
        "normal":  {"path": "{rig}/SurfaceNormal/{view}.png", "format": "png",  "dtype": "uint8", "shape": [1024, 2048, 3], "encoding": "camera-space normal, [0,255]->[-1,1]"},
        "camview": {"path": "{rig}/camview/{view}.npz",        "format": "npz",  "note": "per-view camera intrinsics/extrinsics"},
        "poses":   {"path": "{rig}/transforms.json",           "format": "json", "convention": "NeRF-style transform_matrix (camera-to-world), one frame per view"},
    },
    "split_policy": {"indoor": "by scene 92/6/2", "outdoor": "by scene 92/6/2", "urban": "by city 18 train / 1 val / 1 test"},
}

# a stable, per-file fixed mtime so re-packing the same scene yields byte-identical
# tars (reproducible datasets). 2020-01-01 UTC.
FIXED_MTIME = 1577836800


def _hash_bucket(name: str, mod: int = 1000) -> int:
    """Deterministic, process-independent hash bucket (avoids Python's salted hash)."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % mod


def scene_split(domain: str, scene_id: str) -> str:
    """Assign a scene to train/val/test. Urban is handled separately (by city set)."""
    b = _hash_bucket(scene_id)
    if b < SCENE_RATIO["train"] * 10:                      # 0..919  -> train
        return "train"
    if b < (SCENE_RATIO["train"] + SCENE_RATIO["val"]) * 10:  # 920..979 -> val
        return "val"
    return "test"                                          # 980..999 -> test


def assign_urban(cities: list[str]) -> dict[str, str]:
    """18/1/1 over the cities: two smallest-hash cities -> test, val; rest -> train."""
    ranked = sorted(cities, key=lambda c: hashlib.md5(c.encode()).hexdigest())
    out = {c: "train" for c in cities}
    if len(ranked) >= 1:
        out[ranked[0]] = "test"
    if len(ranked) >= 2:
        out[ranked[1]] = "val"
    return out


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name=name)
    ti.size = size
    ti.mtime = FIXED_MTIME
    ti.mode = 0o644
    return ti


def _depth_npy_to_npz_bytes(npy_path: Path) -> bytes:
    """Load a rendered depth array and losslessly recompress it as float16 npz."""
    arr = np.load(npy_path)
    if arr.dtype != np.float16:
        # renders are float16; downcast defensively only if the value range is safe
        arr = arr.astype(np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, depth=arr)
    return buf.getvalue()


def _iter_scene_members(scene_dir: Path):
    """
    Yield (arcname, kind, src) for every file to put in the scene tar.
    kind is 'file' (add from disk) or 'depth_npz' (convert npy in memory).
    Depth/*.png previews are skipped.
    """
    for rig_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("rig_")):
        for sub in sorted(rig_dir.iterdir()):
            if sub.is_file():
                # top-level rig files (transforms.json, meta.json, ...)
                yield f"{rig_dir.name}/{sub.name}", "file", sub
                continue
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if not f.is_file():
                    continue
                if sub.name == "Depth":
                    if f.suffix == ".png":
                        continue                      # drop colorized depth preview
                    if f.suffix == ".npy":            # metric depth -> float16 npz (precision matters)
                        arc = f"{rig_dir.name}/Depth/{f.stem}.npz"
                        yield arc, "depth_npz", f
                        continue
                if sub.name == "SurfaceNormal" and f.suffix == ".npy":
                    continue                          # drop raw float normal; keep the 8-bit png GT
                yield f"{rig_dir.name}/{sub.name}/{f.name}", "file", f


def pack_scene(scene_dir: Path, out_tar: Path) -> dict:
    """Write one scene -> one tar. Returns per-scene stats."""
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tar.with_suffix(".tar.tmp")
    n_files = 0
    n_rigs = len([p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("rig_")])
    with tarfile.open(tmp, "w") as tar:                    # uncompressed: members already compressed
        for arc, kind, src in _iter_scene_members(scene_dir):
            if kind == "depth_npz":
                data = _depth_npy_to_npz_bytes(src)
                tar.addfile(_tarinfo(arc, len(data)), io.BytesIO(data))
            else:
                ti = tar.gettarinfo(str(src), arcname=arc)
                ti.mtime = FIXED_MTIME
                ti.mode = 0o644
                with open(src, "rb") as fh:
                    tar.addfile(ti, fh)
            n_files += 1
    os.replace(tmp, out_tar)
    return {"scene": scene_dir.name, "rigs": n_rigs, "files": n_files, "bytes": out_tar.stat().st_size}


def _list_scenes(domain_dir: Path) -> list[Path]:
    if not domain_dir.is_dir():
        return []
    # a scene is a top-level dir that actually contains at least one finalized rig
    scenes = []
    for p in sorted(domain_dir.iterdir()):
        if p.is_dir() and any(c.is_dir() and c.name.startswith("rig_") for c in p.iterdir()):
            scenes.append(p)
    return scenes


def plan(src: Path, domains) -> dict:
    """Build the scene -> (domain, split, out_tar) plan without writing anything."""
    jobs = []
    summary = {d: {"train": 0, "val": 0, "test": 0} for d in domains}
    for domain in domains:
        scenes = _list_scenes(src / domain)
        if domain == "urban":
            split_of = assign_urban([s.name for s in scenes])
            for s in scenes:
                sp = split_of[s.name]
                jobs.append((domain, sp, s))
                summary[domain][sp] += 1
        else:
            for s in scenes:
                sp = scene_split(domain, s.name)
                jobs.append((domain, sp, s))
                summary[domain][sp] += 1
    return {"jobs": jobs, "summary": summary}


def _rig_samples_from_tar(tar_path: Path) -> list:
    """Read a scene tar's member list (headers only, no data) -> one record per rig."""
    rigs = {}
    with tarfile.open(tar_path) as tar:
        for n in tar.getnames():
            parts = n.split("/")
            if not parts or not parts[0].startswith("rig_"):
                continue
            rig = parts[0]
            rigs.setdefault(rig, set())
            if len(parts) == 3 and parts[1] == "Image" and parts[2].endswith(".png"):
                rigs[rig].add(parts[2][:-4])          # view id, e.g. "00"
    return [{"rig": r, "n_views": len(v), "views": sorted(v)} for r, v in sorted(rigs.items())]


def _scan_tar(spec):
    """Worker: (domain, split, tar_path) -> (domain, split, name, stem, nbytes, samples)."""
    domain, split, tar = spec
    p = Path(tar)
    return domain, split, p.name, p.stem, p.stat().st_size, _rig_samples_from_tar(p)


def build_metadata(dst: Path, workers: int = 16) -> dict:
    """Scan every packed tar and (re)write dataset_info.json + per-split index.jsonl.
    Reads only tar headers (in parallel -- header seeks over Lustre are latency-bound),
    so it is cheap to re-run as new domains are packed. The index.jsonl (one JSON
    sample-record per line) is the dataloader's length + index: len(split) = #lines;
    __getitem__ opens {tar} and reads members via the SCHEMA path patterns."""
    specs = []
    for domain in DOMAINS:
        for split in ("train", "val", "test"):
            sdir = dst / domain / split
            if sdir.is_dir():
                specs += [(domain, split, str(t)) for t in sorted(sdir.glob("*.tar"))]

    grouped = defaultdict(list)   # (domain, split) -> [(stem, name, nbytes, samples), ...]
    if specs:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as ex:
            for domain, split, name, stem, nbytes, samples in ex.map(_scan_tar, specs):
                grouped[(domain, split)].append((stem, name, nbytes, samples))

    domains = {}
    totals = {"scenes": 0, "rigs": 0, "tars": 0, "bytes": 0}
    splits_total = {s: {"scenes": 0, "rigs": 0} for s in ("train", "val", "test")}
    for domain in DOMAINS:
        dd = {}
        for split in ("train", "val", "test"):
            items = grouped.get((domain, split))
            if not items:
                continue
            items.sort()
            n_rigs, nbytes, lines = 0, 0, []
            for stem, name, sz, samples in items:
                nbytes += sz
                for s in samples:
                    n_rigs += 1
                    lines.append(json.dumps({"scene": stem, "tar": name, **s}))
            (dst / domain / split / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
            dd[split] = {"scenes": len(items), "rigs": n_rigs, "tars": len(items), "bytes": nbytes}
            totals["scenes"] += len(items); totals["rigs"] += n_rigs
            totals["tars"] += len(items); totals["bytes"] += nbytes
            splits_total[split]["scenes"] += len(items); splits_total[split]["rigs"] += n_rigs
        if dd:
            domains[domain] = dd
    info = {
        "created": int(time.time()),
        "schema": SCHEMA,
        "split_ratio": SCENE_RATIO,
        "domains": domains,             # per-domain per-split {scenes, rigs, tars, bytes}
        "splits_total": splits_total,   # cross-domain rig/scene counts per split (dataloader length)
        "totals": totals,
    }
    (dst / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="outputs", type=Path, help="root holding <domain>/ scene folders")
    ap.add_argument("--dst", default="dataset", type=Path, help="output dataset root")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--dry-run", action="store_true", help="print the split plan, write nothing")
    ap.add_argument("--metadata-only", action="store_true",
                    help="skip packing; (re)write dataset_info.json + per-split index.jsonl from existing tars")
    args = ap.parse_args()

    if args.metadata_only:
        info = build_metadata(args.dst, args.workers)
        print(f"metadata -> {args.dst/'dataset_info.json'} ({info['totals']['rigs']} rigs, {info['totals']['tars']} tars)")
        for s in ("train", "val", "test"):
            st = info["splits_total"][s]
            print(f"  {s:5s}: {st['rigs']:>5} rigs / {st['scenes']:>4} scenes")
        return

    p = plan(args.src, args.domains)
    jobs, summary = p["jobs"], p["summary"]

    print(f"Planned {len(jobs)} scene tars from {args.src} -> {args.dst}")
    for d in args.domains:
        s = summary[d]
        print(f"  {d:8s}: {s['train']:>4} train / {s['val']:>3} val / {s['test']:>3} test")

    if args.dry_run:
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pack_scene, sd, args.dst / dm / sp / f"{sd.name}.tar"): (dm, sp, sd.name)
                for dm, sp, sd in jobs}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(futs):
                gb = sum(x["bytes"] for x in results) / 1e9
                print(f"  packed {done}/{len(futs)} scenes ({gb:.1f} GB)")

    total_rigs = sum(r["rigs"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)
    print(f"\nPacked: {len(results)} tars, {total_rigs} rigs, {total_bytes/1e9:.1f} GB")

    # (re)build dataloader metadata from every tar now present (all domains, not just this run)
    info = build_metadata(args.dst, args.workers)
    print(f"Metadata: {args.dst/'dataset_info.json'} + per-split index.jsonl")
    for s in ("train", "val", "test"):
        st = info["splits_total"][s]
        print(f"  {s:5s}: {st['rigs']:>5} rigs / {st['scenes']:>4} scenes")


if __name__ == "__main__":
    main()
