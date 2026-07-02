"""Final README update for prs-eth/PanoInfinigen after the re-encode.

Run this AFTER every shard has been uploaded. It rewrites the YAML frontmatter
to:
- drop `dataset_info` (forces HF to auto-derive feature types from the
  parquet files themselves — this is what works around the stale-cache
  cast errors we hit on ZuriPano);
- leave the existing `configs` block untouched.

And rewrites the Data Structure / How to Use sections of the markdown body to
match the new column layout with per-config decode formulas.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

from . import DEPTH_MAX_M, SOURCE_REPO

CARD_NEW_TABLE = """## Data Structure
Three configurations: `indoor`, `nature`, `urban`. Each has `train`, `val`,
`test` splits. Every row carries:

| Feature     | Type    | Description |
| :---        | :---    | :---        |
| `image`     | `Image` | 8-bit RGB equirectangular panorama, JPEG (q=95) — transcoded from the source PNG. Visually indistinguishable from the source. |
| `depth`     | `Image` | 16-bit single-channel PNG. Decode to **metres** as `np.asarray(img, np.float32) * MAX_M / 65535.0`, where `MAX_M` is **75** for `indoor`, **75** for `nature`, **500** for `urban` (matching the renderer's hard clip). Invalid pixels would be `0.0`. |
| `depth_viz` | `Image` | 8-bit RGB Spectral-colormapped log-depth preview. **Preview only — do NOT use for metrics or training; decode `depth` instead.** |
| `normals`   | `Image` | 8-bit RGB JPEG (q=95) of `(n + 1) / 2 * 255`. Decode to **unit normals** in `[-1, 1]` as `np.asarray(img, np.float32) / 127.5 - 1.0`. JPG quantization introduces ~1° directional error after decode — smaller than the original float16 → uint8 cast already costs and below typical normal-estimation tolerance. |
"""

CARD_NEW_HOWTO = """## How to Use

```python
import numpy as np
from datasets import load_dataset

DEPTH_MAX_M = {"indoor": 75.0, "nature": 75.0, "urban": 500.0}

config = "indoor"  # or "nature" / "urban"
ds = load_dataset("prs-eth/PanoInfinigen", name=config, split="train")
sample = ds[0]

rgb     = sample["image"]                                                                  # PIL.Image, (W=4096, H=2048)
depth   = np.asarray(sample["depth"],   dtype=np.float32) * (DEPTH_MAX_M[config] / 65535.0)  # (H, W) float32, metres
normals = np.asarray(sample["normals"], dtype=np.float32) / 127.5 - 1.0                    # (H, W, 3) float32, unit vectors in [-1, 1]
```
"""


def rewrite_readme(text: str) -> str:
    # 1. YAML: drop the dataset_info block (HF will auto-derive from parquet).
    text = re.sub(
        r"\ndataset_info:\n(?:- [^\n]+\n(?:  [^\n]+\n)+)+",
        "\n",
        text,
        count=1,
    )

    # 2. Update the Modality bullet in Dataset Summary.
    text = re.sub(
        r"- \*\*Modality:\*\*[^\n]*\n",
        "- **Modality:** RGB (JPEG q=95), Depth (16-bit PNG, per-config scale factor), "
        "Surface Normals (8-bit RGB JPEG q=95), Depth Viz (8-bit Spectral RGB PNG, preview only).\n",
        text,
        count=1,
    )

    # 3. Replace the Data Structure section + the immediately-following table.
    text = re.sub(
        r"## Data Structure[\s\S]*?(?=\n##\s)",
        CARD_NEW_TABLE + "\n",
        text,
        count=1,
    )

    # 3. Replace the How to Use section + code block.
    text = re.sub(
        r"## How to Use[\s\S]*?```\n",
        CARD_NEW_HOWTO + "\n",
        text,
        count=1,
    )
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=SOURCE_REPO)
    ap.add_argument("--input", default=None, help="optional: read README from this local path instead of HF")
    ap.add_argument("--output", default=None, help="optional: also write the new README to this local path")
    ap.add_argument("--dry-run", action="store_true", help="don't push to HF")
    args = ap.parse_args()

    if args.input:
        text = Path(args.input).read_text()
    else:
        import requests
        text = requests.get(f"https://huggingface.co/datasets/{args.repo}/raw/main/README.md", timeout=30).text
    new_text = rewrite_readme(text)

    if args.output:
        Path(args.output).write_text(new_text)
        print(f"wrote {args.output}  ({len(new_text)} bytes)")

    if args.dry_run:
        print("--- new YAML preview ---")
        print(new_text.split("---", 2)[1])
        return

    # HfApi reads HF_TOKEN from env; avoid huggingface_hub.login() because
    # it writes to a shared cache file that races with concurrent jobs.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=new_text.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Option C re-encode: depth/normals as Image; drop stale dataset_info",
    )
    print(f"updated README on {args.repo}")


if __name__ == "__main__":
    main()
