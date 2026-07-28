"""
reproject_view.py -- re-render one panorama view of a rig from another, using the
saved camera poses + the target view's per-pixel depth. This is the core geometric
transform that pose_check.py validates and viz_gt.py visualizes, isolated for reuse.

Idea (inverse / image-based warping): to synthesize what the TARGET camera sees, walk
each TARGET pixel out along its viewing ray by that pixel's depth to a 3D point, move
that point into the SOURCE camera, find where it projects in the SOURCE image, and copy
that colour back. So we sample the SOURCE image to reconstruct the TARGET view.

Conventions (equirectangular; transforms.json "convention" = blender_cam_to_world;
all verified end-to-end by pose_check.py):
  - transform_matrix T is CAMERA -> WORLD (4x4). Camera axes: -Z forward, +Y up, +X right.
  - equirect pixel (u,v) <-> unit ray direction d in the camera frame:
        theta = ((u+0.5)/W - 0.5) * 2*pi          # yaw:  image centre = forward (-Z)
        phi   = (0.5 - (v+0.5)/H) * pi             # pitch: top row = +pi/2 (up)
        d     = (cos(phi)*sin(theta), sin(phi), -cos(phi)*cos(theta))
  - DEPTH is RADIAL distance along d, NOT planar z:   point_cam = d * depth.
  - the per-frame "intrinsics" matrix in transforms.json is a placeholder for the
    equirectangular model (focal ~ -2.2e-5) -- do NOT project with it; use the map above.

Run `python reproject_view.py [domain] [scene] [rig]` for a runnable example that
re-renders view 0 from view 1 and writes reprojected.png.
"""

import numpy as np


def equirect_rays(H, W):
    """Unit viewing-ray direction (camera frame) for every pixel of an HxW equirect image."""
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    theta = (((u + 0.5) / W) - 0.5) * 2.0 * np.pi
    phi = (0.5 - ((v + 0.5) / H)) * np.pi
    cph = np.cos(phi)
    return np.stack([cph * np.sin(theta), np.sin(phi), -cph * np.cos(theta)], axis=-1)


def rays_to_pixels(d, H, W):
    """Inverse of equirect_rays: camera-frame directions -> (u, v) float pixel coords."""
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
    phi = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    theta = np.arctan2(d[..., 0], -d[..., 2])
    u = (((theta / (2.0 * np.pi)) + 0.5) * W) - 0.5
    v = ((0.5 - (phi / np.pi)) * H) - 0.5
    return u, v


def sample(img, u, v, bilinear=False):
    """Look up img at float pixel coords (u, v). Longitude (u) wraps; latitude (v) clamps."""
    H, W = img.shape[:2]
    if not bilinear:
        ui = (np.rint(u).astype(np.int64)) % W
        vi = np.clip(np.rint(v).astype(np.int64), 0, H - 1)
        return img[vi, ui]
    u0 = np.floor(u).astype(np.int64); v0 = np.floor(v).astype(np.int64)
    fu = (u - u0)[..., None]; fv = (v - v0)[..., None]
    u0m, u1m = u0 % W, (u0 + 1) % W
    v0c, v1c = np.clip(v0, 0, H - 1), np.clip(v0 + 1, 0, H - 1)
    img = img.astype(np.float32)
    top = img[v0c, u0m] * (1 - fu) + img[v0c, u1m] * fu
    bot = img[v1c, u0m] * (1 - fu) + img[v1c, u1m] * fu
    return top * (1 - fv) + bot * fv


def reproject(src_rgb, dst_depth, dst_pose, src_pose, bilinear=False):
    """
    Re-render the DST view by sampling the SRC image (inverse warp).

        src_rgb   : (H, W, 3)  source-view image (the one we copy colour from)
        dst_depth : (H, W)     radial depth of the view we are synthesizing
        dst_pose  : (4, 4)     DST camera-to-world  (the pose we render at)
        src_pose  : (4, 4)     SRC camera-to-world  (where src_rgb was taken)
    returns:
        out   : (H, W, 3)      src_rgb resampled to align with the DST view
        valid : (H, W) bool    pixels with usable (finite, positive) depth

    If dst==src the output equals src_rgb (identity). The residual after warping is
    occlusion/disocclusion + sky (no depth), never global misalignment -- that is the
    signal that the relative pose (dst_pose^-1 @ src_pose) is correct.
    """
    H, W = dst_depth.shape

    # 1) DST pixel -> viewing ray -> 3D point in the DST camera frame (radial depth)
    pts_dst = equirect_rays(H, W) * np.nan_to_num(dst_depth)[..., None]

    # 2) DST camera -> world -> SRC camera.   (cam->world = R x + t;  world->cam = R^T (x - t))
    pts_world = pts_dst @ dst_pose[:3, :3].T + dst_pose[:3, 3]
    pts_src = (pts_world - src_pose[:3, 3]) @ src_pose[:3, :3]

    # 3) where those 3D points fall in the SRC equirect image, then sample
    u, v = rays_to_pixels(pts_src, H, W)
    out = sample(src_rgb, u, v, bilinear=bilinear)

    valid = np.isfinite(dst_depth) & (dst_depth > 0)
    return out, valid


def forward_render(src_rgb, src_depth, src_pose, dst_pose):
    """
    Forward warp: unproject the SOURCE view into a 3D point cloud (using the SOURCE
    depth), then RENDER that cloud at the DST pose. This is the "base RGB+depth ->
    point cloud -> novel view" test; unlike reproject() it uses the SOURCE depth and
    produces holes at disocclusions.

        src_rgb   : (H, W, 3)  base-view image (each pixel becomes a coloured 3D point)
        src_depth : (H, W)     radial depth of the base view
        src_pose  : (4, 4)     base camera-to-world
        dst_pose  : (4, 4)     novel camera-to-world (where we render the cloud)
    returns:
        out    : (H, W, 3)     rendered novel view (0 where nothing splats)
        filled : (H, W) bool   which novel pixels a point actually covered

    Occlusion is resolved with a z-buffer (nearest point to the DST camera wins).
    """
    H, W = src_depth.shape
    # 1) SOURCE pixels -> 3D points in world
    pts_cam = equirect_rays(H, W) * np.nan_to_num(src_depth)[..., None]
    pts_world = pts_cam @ src_pose[:3, :3].T + src_pose[:3, 3]
    # 2) project the cloud into the DST camera; range = distance to DST cam (z-buffer key)
    pts_dst = (pts_world - dst_pose[:3, 3]) @ dst_pose[:3, :3]
    rng = np.linalg.norm(pts_dst, axis=-1)
    u, v = rays_to_pixels(pts_dst, H, W)

    m = np.isfinite(src_depth) & (src_depth > 0) & np.isfinite(rng) & (rng > 0)
    lin = (np.clip(np.rint(v).astype(np.int64), 0, H - 1) * W
           + (np.rint(u).astype(np.int64) % W))[m]
    col = src_rgb[m].astype(np.float32)
    rr = rng[m]

    # 3) z-buffer splat: sort far->near so the nearest point is written last (wins)
    order = np.argsort(-rr)
    out = np.zeros((H * W, 3), np.float32)
    filled = np.zeros(H * W, bool)
    out[lin[order]] = col[order]
    filled[lin[order]] = True
    return out.reshape(H, W, 3), filled.reshape(H, W)


def relative_pose(dst_pose, src_pose):
    """4x4 transform taking SRC-camera coords into DST-camera coords (dst^-1 @ src)."""
    inv = np.eye(4)
    inv[:3, :3] = dst_pose[:3, :3].T
    inv[:3, 3] = -dst_pose[:3, :3].T @ dst_pose[:3, 3]
    return inv @ src_pose


if __name__ == "__main__":
    import sys
    from pano_dataset import PanoRigDataset

    dom = sys.argv[1] if len(sys.argv) > 1 else "indoor"
    ds = PanoRigDataset("dataset", split="train", domains=[dom], min_views=6)
    if len(sys.argv) > 3:
        i = next(k for k in range(len(ds))
                 if ds.samples[k]["scene"] == sys.argv[2] and ds.samples[k]["rig"] == sys.argv[3])
    else:
        i = 0
    s = ds[i]
    a, b = 0, 1                                    # synthesize view a (target) from view b (source)

    synth, valid = reproject(s["rgb"][b], s["depth"][a], s["pose"][a], s["pose"][b])

    A = s["rgb"][a].astype(np.float32)
    e_reproj = float(np.abs(synth.astype(np.float32) - A).mean(-1)[valid].mean())
    e_still = float(np.abs(s["rgb"][b].astype(np.float32) - A).mean(-1)[valid].mean())
    print(f"{s['domain']} {s['scene']}/{s['rig']}: re-render view {a} from view {b} "
          f"(baseline {s['baseline_m']:.2f} m)")
    print(f"  mean |reprojected - viewA| = {e_reproj:.1f}   vs no-motion |viewB - viewA| = {e_still:.1f}"
          f"   ({e_still / max(e_reproj, 1e-6):.1f}x better)")
    print("  relative pose (src->dst cam):\n", np.round(relative_pose(s["pose"][a], s["pose"][b]), 4))
    try:
        from PIL import Image
        Image.fromarray(synth.astype(np.uint8)).save("reprojected.png")
        Image.fromarray(s["rgb"][a]).save("target_viewA.png")
        print("  wrote reprojected.png (should match target_viewA.png)")
    except Exception:
        pass
