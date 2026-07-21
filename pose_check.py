"""Validate saved camera poses geometrically.

(1) cycle consistency: pixel A -> 3D (depth_A) -> B -> 3D (depth_B) -> back to A.
    Round-trip pixel drift ~0 iff poses AND depth are mutually consistent.
(2) photometric reprojection: predict A's image by sampling B at the reprojected
    coords. Must beat the no-motion baseline (|A-B|) and a negated-translation
    control, otherwise the poses carry no real geometry.

Equirect (blender_cam_to_world): cam -Z forward, +Y up, +X right.
  dir = (cos(phi) sin(theta), sin(phi), -cos(phi) cos(theta))
theta sign convention is auto-detected (both tried, the consistent one reported).
"""
import json
import sys

import numpy as np

from pano_dataset import PanoRigDataset


def dirs_from_pix(H, W, sgn):
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    th = sgn * (((u + 0.5) / W) - 0.5) * 2 * np.pi
    ph = (0.5 - ((v + 0.5) / H)) * np.pi
    cp = np.cos(ph)
    return np.stack([cp * np.sin(th), np.sin(ph), -cp * np.cos(th)], -1)


def pix_from_dirs(d, H, W, sgn):
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
    ph = np.arcsin(np.clip(d[..., 1], -1, 1))
    th = np.arctan2(d[..., 0], -d[..., 2]) * sgn
    u = ((th / (2 * np.pi)) + 0.5) * W - 0.5
    v = (0.5 - ph / np.pi) * H - 0.5
    return u, v


def cam_to_world(P, T):
    return P @ T[:3, :3].T + T[:3, 3]


def world_to_cam(P, T):
    return (P - T[:3, 3]) @ T[:3, :3]


def project(depthA, TA, TB, H, W, sgn):
    """A's pixels -> pixel coords in B."""
    P = dirs_from_pix(H, W, sgn) * depthA[..., None]      # radial depth
    return pix_from_dirs(world_to_cam(cam_to_world(P, TA), TB), H, W, sgn)


def sample(img, u, v):
    H, W = img.shape[:2]
    ui = np.clip(np.rint(u), 0, W - 1).astype(np.int32)
    vi = np.clip(np.rint(v), 0, H - 1).astype(np.int32)
    return img[vi, ui]


def run(sample_rec, sgn, a=0, b=1):
    rgb, dep, pose = sample_rec["rgb"], sample_rec["depth"], sample_rec["pose"]
    H, W = dep.shape[1:]
    A, B = rgb[a].astype(np.float32), rgb[b].astype(np.float32)
    dA, dB, TA, TB = dep[a], dep[b], pose[a], pose[b]

    far = np.nanpercentile(dA, 99.5)
    valid = np.isfinite(dA) & (dA > 0) & (dA < far * 0.98)   # drop sky/far-clamp

    # ---- (1) cycle consistency A->B->A
    uB, vB = project(dA, TA, TB, H, W, sgn)
    dB_at = sample(dB[..., None], uB, vB)[..., 0]
    dirB = dirs_from_pix(H, W, sgn)
    dirB_at = sample(dirB, uB, vB)
    P_back = cam_to_world(dirB_at * dB_at[..., None], TB)
    uA2, vA2 = pix_from_dirs(world_to_cam(P_back, TA), H, W, sgn)
    v0, u0 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    drift = np.hypot(uA2 - u0, vA2 - v0)
    ok = valid & np.isfinite(drift)
    cyc = float(np.median(drift[ok])) if ok.any() else float("nan")
    cyc_frac = float(np.mean(drift[ok] < 2.0)) if ok.any() else float("nan")

    # ---- (2) photometric: true vs controls
    def photo(u, v):
        e = np.abs(sample(B, u, v) - A).mean(-1)
        return float(np.median(e[valid]))

    e_true = photo(uB, vB)
    e_still = float(np.median(np.abs(B - A).mean(-1)[valid]))       # no motion
    TB_neg = TB.copy()
    TB_neg[:3, 3] = TA[:3, 3] - (TB[:3, 3] - TA[:3, 3])            # flipped baseline
    un, vn = project(dA, TA, TB_neg, H, W, sgn)
    e_neg = photo(un, vn)
    return cyc, cyc_frac, e_true, e_still, e_neg


if __name__ == "__main__":
    specs = [("indoor", "train"), ("urban", "train"), ("outdoor", "train")]
    for dom, split in specs:
        ds = PanoRigDataset("dataset", split=split, domains=[dom], min_views=6)
        print(f"\n=== {dom} ({len(ds)} rigs) ===")
        for si in (0, len(ds) // 2):
            s = ds[si]
            best = None
            for sgn in (+1, -1):
                r = run(s, sgn, 0, 1)
                if best is None or r[0] < best[1][0]:
                    best = (sgn, r)
            sgn, (cyc, cfrac, e_t, e_s, e_n) = best
            print(f"  {s['scene']}/{s['rig']}  theta_sign={sgn:+d}")
            print(f"    cycle A->B->A: median drift {cyc:.3f}px | {100*cfrac:.1f}% under 2px")
            print(f"    photometric  : true {e_t:.1f} | no-motion {e_s:.1f} | neg-baseline {e_n:.1f}"
                  f"   -> {'PASS' if (e_t < e_s and e_t < e_n) else 'FAIL'}")
