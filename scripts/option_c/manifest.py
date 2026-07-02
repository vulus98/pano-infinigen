"""Generate / load the shard manifest used by all subsequent stages.

The manifest is a JSON list. Each entry describes one parquet shard:

    {
      "config":       "indoor",
      "split":        "train",
      "shard_name":   "train-00000.parquet",
      "src_path":     "data/indoor/train-00000.parquet",   # path on HF main branch
      "size":          1469310464,                          # bytes
      "depth_max_m":   75.0,
    }

Why precompute? Slurm array tasks need a deterministic ordering so task k
always handles the same shard, which makes the pipeline restartable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

from . import CONFIGS, DEPTH_MAX_M, SOURCE_REPO, SPLITS


def build_manifest() -> list[dict]:
    fs = HfFileSystem()
    out: list[dict] = []
    for cfg in CONFIGS:
        for split in SPLITS:
            base = f"datasets/{SOURCE_REPO}/data/{cfg}/"
            try:
                files = fs.ls(base, detail=True)
            except FileNotFoundError:
                continue
            files = [
                f for f in files
                if f["name"].endswith(".parquet") and f"/{split}-" in f["name"]
            ]
            files.sort(key=lambda f: f["name"])
            for f in files:
                rel = f["name"].split("/")[-1]
                out.append({
                    "config": cfg,
                    "split": split,
                    "shard_name": rel,
                    "src_path": f"data/{cfg}/{rel}",
                    "size": int(f.get("size", 0)),
                    "depth_max_m": DEPTH_MAX_M[cfg],
                })
    return out


def filter_manifest(
    manifest: list[dict],
    config: str | None = None,
    split: str | None = None,
) -> list[dict]:
    out = manifest
    if config:
        out = [e for e in out if e["config"] == config]
    if split:
        out = [e for e in out if e["split"] == split]
    return out


def slice_for_task(items: list[dict], task_index: int, num_tasks: int) -> list[dict]:
    """Round-robin slice so a quick task and a slow task each see balanced work."""
    return items[task_index::num_tasks]


def main():
    ap = argparse.ArgumentParser(description="Generate the PanoInfinigen Option-C shard manifest")
    ap.add_argument("--out", required=True, help="Path to write manifest.json")
    args = ap.parse_args()
    manifest = build_manifest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}  ({len(manifest)} shards)")
    # quick summary
    from collections import Counter
    by_split = Counter((e["config"], e["split"]) for e in manifest)
    total_bytes = sum(e["size"] for e in manifest)
    print(f"total: {total_bytes/1024**3:.1f} GB across {len(manifest)} shards")
    for k in sorted(by_split):
        print(f"  {k[0]}/{k[1]}: {by_split[k]} shards")


if __name__ == "__main__":
    main()
