#!/usr/bin/env python3
"""
pano_dataset.py -- reference dataloader for the packed panorama dataset (pack_dataset.py).

One sample = one RIG: a multi-view equirectangular panorama set (V views of the same
scene from a small camera constellation). Samples are enumerated from
<domain>/<split>/index.jsonl and read straight out of their scene tar.

    ds = PanoRigDataset("dataset", split="train", domains=["indoor", "urban"])
    len(ds)                       # number of rigs in the split
    s = ds[0]
      s["rgb"]        (V, H, W, 3) uint8
      s["depth"]      (V, H, W)    float32, metres
      s["normal"]     (V, H, W, 3) float32 in [-1, 1], camera-space
      s["pose"]       (V, 4, 4)    float32, camera-to-world (NeRF convention)
      s["K"]          (V, 3, 3)    float32 intrinsics
      s["baseline_m"] float, s["domain"], s["scene"], s["rig"], s["views"]

Works with torch DataLoader(num_workers>0): tar handles are opened lazily, cached
per process, and dropped across fork/pickle. torch is optional -- without it this is a
plain sequence returning numpy arrays.

NOTE: each frame in transforms.json carries depth_path/normal_path pointing at the
ORIGINAL render layout (.npy). The packed tars store depth as .npz and normals as
.png (the raw float normal .npy is dropped). Always resolve members via the patterns
in dataset_info.json["schema"], as this loader does -- not via those fields.
"""

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

try:  # torch is optional -- the class is a plain sequence without it
    from torch.utils.data import Dataset as _Base
except Exception:  # pragma: no cover
    class _Base:
        pass

SPLITS = ("train", "val", "test")


class PanoRigDataset(_Base):
    def __init__(self, root, split="train", domains=None,
                 modalities=("rgb", "depth", "normal", "pose"), max_views=None, min_views=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        info_path = self.root / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} not found -- run `python pack_dataset.py --dst {root} --metadata-only`")
        self.info = json.loads(info_path.read_text())
        self.schema = self.info["schema"]
        packed = list(self.info["domains"].keys())
        self.domains = list(domains) if domains else packed
        unknown = set(self.domains) - set(packed)
        if unknown:
            raise ValueError(f"domain(s) {sorted(unknown)} not packed; available: {packed}")
        self.split = split
        self.modalities = tuple(modalities)
        self.max_views = max_views

        self.samples = []
        for d in self.domains:
            idx = self.root / d / split / "index.jsonl"
            if not idx.exists():
                continue
            with open(idx) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["domain"] = d
                        self.samples.append(rec)

        # A few rigs finalized with <V views (their render task died mid-way). Keep
        # them by default (transparent, len == metadata); pass min_views=6 to train on
        # complete multi-view sets only.
        if min_views:
            self.samples = [s for s in self.samples if s.get("n_views", 0) >= min_views]

        self._tars = {}
        self._pid = os.getpid()

    def __len__(self):
        return len(self.samples)

    # -- tar access -----------------------------------------------------------
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_tars"] = {}          # never pickle open handles (spawn workers)
        return st

    def _tar(self, domain, tar_name):
        pid = os.getpid()
        if pid != self._pid:      # forked: inherited handles share file offsets -> drop
            self._tars = {}
            self._pid = pid
        key = (domain, tar_name)
        t = self._tars.get(key)
        if t is None:
            t = tarfile.open(self.root / domain / self.split / tar_name)
            self._tars[key] = t
        return t

    @staticmethod
    def _read(tar, name):
        f = tar.extractfile(name)
        if f is None:
            raise KeyError(f"{name} not in {tar.name}")
        return f.read()

    # -- sample ---------------------------------------------------------------
    def __getitem__(self, i):
        rec = self.samples[i]
        tar = self._tar(rec["domain"], rec["tar"])
        rig = rec["rig"]
        views = rec["views"]
        if self.max_views:
            views = views[: self.max_views]

        out = {"domain": rec["domain"], "scene": rec["scene"], "rig": rig, "views": list(views)}

        if "rgb" in self.modalities:
            out["rgb"] = np.stack([
                np.asarray(Image.open(io.BytesIO(self._read(tar, f"{rig}/Image/{v}.png"))).convert("RGB"))
                for v in views])

        if "depth" in self.modalities:
            out["depth"] = np.stack([
                np.load(io.BytesIO(self._read(tar, f"{rig}/Depth/{v}.npz")))["depth"].astype(np.float32)
                for v in views])

        if "normal" in self.modalities:
            n = np.stack([
                np.asarray(Image.open(io.BytesIO(self._read(tar, f"{rig}/SurfaceNormal/{v}.png"))).convert("RGB"))
                for v in views])
            out["normal"] = n.astype(np.float32) / 127.5 - 1.0

        if "pose" in self.modalities or "K" in self.modalities:
            tj = json.loads(self._read(tar, f"{rig}/transforms.json"))
            by_sub = {int(fr["subcam"]): fr for fr in tj["frames"]}
            frames = [by_sub[int(v)] for v in views]
            if "pose" in self.modalities:
                out["pose"] = np.stack([np.asarray(f["transform_matrix"], dtype=np.float32) for f in frames])
            out["K"] = np.stack([np.asarray(f["intrinsics"], dtype=np.float32) for f in frames])
            out["baseline_m"] = float(tj.get("baseline_m", 0.0))
            out["convention"] = tj.get("convention", "")

        return out


def _smoke(root="dataset"):
    """Quick self-test: counts match metadata, one sample decodes with sane shapes."""
    info = json.loads((Path(root) / "dataset_info.json").read_text())
    print("packed domains:", list(info["domains"].keys()))
    ok = True
    for split in SPLITS:
        ds = PanoRigDataset(root, split=split)
        expect = info["splits_total"][split]["rigs"]
        status = "OK" if len(ds) == expect else "MISMATCH"
        ok &= len(ds) == expect
        print(f"  {split:5s}: len={len(ds):>5}  metadata says {expect:>5}  [{status}]")

    ds = PanoRigDataset(root, split="train")
    s = ds[0]
    print(f"\nsample[0]: domain={s['domain']} scene={s['scene']} rig={s['rig']} views={s['views']}")
    for k in ("rgb", "depth", "normal", "pose", "K"):
        a = s[k]
        print(f"  {k:7s} shape={str(a.shape):22s} dtype={str(a.dtype):8s} "
              f"range=[{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]")
    d = s["depth"]
    finite = d[np.isfinite(d)]
    print(f"  depth finite: {finite.size}/{d.size}  median={np.median(finite):.2f} m  baseline={s['baseline_m']:.2f} m")
    assert s["rgb"].dtype == np.uint8 and s["rgb"].ndim == 4
    assert s["depth"].ndim == 3 and s["pose"].shape[1:] == (4, 4)
    assert -1.01 <= s["normal"].min() and s["normal"].max() <= 1.01
    print("\nsmoke test:", "PASS" if ok else "FAIL (count mismatch)")
    return ok


if __name__ == "__main__":
    import sys
    raise SystemExit(0 if _smoke(sys.argv[1] if len(sys.argv) > 1 else "dataset") else 1)
