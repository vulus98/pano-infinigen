"""Option C re-encoding tooling for prs-eth/PanoInfinigen.

Re-encodes the dataset from its current `{image: PNG, depth: binary float16 NPY,
normals: binary float16 NPY}` layout into a viewer-friendly `{image: PNG,
depth: 16-bit PNG, depth_viz: 8-bit Spectral RGB PNG, normals: 8-bit RGB PNG}`
layout. The encoder pipeline matches what was validated on
`vulus98/panoinfinigen-option-c-test` and on `prs-eth/ZuriPano`.

Per-config depth scales (used by both the encoder and the dataset card).
These match the upstream renderer's hard clips exactly, so the 16-bit PNG
covers the full physical range with no wasted bits and no further clipping:
- indoor : 75 m max  (source clips at 75 m; precision 1.14 mm).
- nature : 75 m max  (source clips at 75 m; precision 1.14 mm).
- urban  : 500 m max (source clips at 500 m; precision 7.63 mm).
"""

# Per-config depth scale used by the 16-bit PNG depth encoding.
# Decode formula in user code: `np.asarray(img, np.float32) * DEPTH_MAX_M[config] / 65535.0`.
DEPTH_MAX_M = {
    "indoor": 75.0,
    "nature": 75.0,
    "urban": 500.0,
}

CONFIGS = ("indoor", "nature", "urban")
SPLITS = ("train", "val", "test")
SOURCE_REPO = "prs-eth/PanoInfinigen"
