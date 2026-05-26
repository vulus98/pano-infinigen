---
license: bsd-3-clause
pretty_name: PanoInfinigen Option C Validation
configs:
- config_name: default
  data_files:
  - split: val
    path: data/urban/val-*
---

# PanoInfinigen Option C Validation

End-to-end validation of the Option C re-encoding pipeline using all 18
shards of `urban/val` from `prs-eth/PanoInfinigen` (≈23 GB compressed).
After we sign off on the viewer here, the real run overwrites the
production PanoInfinigen dataset shards. This repo is temporary and will
be deleted.

Per-row columns (same as the production target):

- `image` : 8-bit RGB PNG (kept as-is from source).
- `depth` : 16-bit single-channel PNG. Decode to **metres** as
  `np.asarray(img, np.float32) * (500.0 / 65535.0)` (urban scale; source clip 500 m).
- `depth_viz` : 8-bit RGB PNG, Spectral-colormapped log-depth.
  **Preview only — decode `depth` for any metric use.**
- `normals` : 8-bit RGB PNG of `(n + 1) / 2 * 255`. Decode as
  `np.asarray(img, np.float32) / 127.5 - 1.0`.
