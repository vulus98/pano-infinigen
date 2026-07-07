#!/usr/bin/env python
"""Harvest multi-view panorama sets from already-generated scenes.

Scene generation (terrain + populate) is by far the most expensive step, and it
produces one saved ``fine/scene.blend`` per scene. Camera *placement* is cheap by
comparison (BVH raycasts), so this script decouples the two: it loads a saved
scene once and samples MANY multi-view camera rigs from it, rendering each as a
set of pose-registered 360 deg equirectangular panoramas. Failed placements are
free retries on an already-paid-for scene, and one scene yields several
multi-view sets.

Per scene it:
  1. loads fine/scene.blend and builds an INSTANCE-AWARE collision BVH (so cars,
     trees, grass etc. — which are geometry-node instances — are seen);
  2. samples `--rigs-per-scene` multi-view rigs (anchor + ring, see
     camera.multiview_rig_config), each with its own (optionally random) baseline,
     validating every sub-camera against the full populated scene via a
     panoramic sphere raycast (rejects underground / enclosed / prop-clipping);
  3. renders each sub-camera (RGB + metric depth + normals + pose) with the same
     passes the datagen uses, then writes a per-set transforms.json.

Example:
  python harvest_multiview.py --scenes-root outputs/mv_scenes \\
      --output outputs/mv_harvest --rigs-per-scene 4 --n-views 8 \\
      --baseline uniform,0.3,0.7 --resolution 1024,512
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("harvest")

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering import render as render_mod
from infinigen.core.util import blender as butil


# --------------------------------------------------------------------------- #
# Instance-aware collision BVH (matches what the renderer rasterizes).
# --------------------------------------------------------------------------- #
def build_instance_aware_bvh(exclude_prefix=("camera", "camrig", "Camera")):
    """World-space BVHTree including geometry-node / collection instances, built
    from the evaluated depsgraph so it matches the rendered geometry."""
    # Force full evaluation first so no geometry-node geometry is missing from the
    # depsgraph (a partial BVH would let enclosed poses pass validation).
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    cache = {}

    def _local(ob):
        key = ob.data.name
        if key in cache:
            return cache[key]
        me = bpy.data.meshes.new_from_object(ob)
        geom = None
        if me is not None:
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.to_mesh(me)
            bm.free()
            if len(me.vertices):
                geom = (
                    np.array([v.co[:] for v in me.vertices], dtype=np.float64),
                    [tuple(p.vertices) for p in me.polygons],
                )
            bpy.data.meshes.remove(me)
        cache[key] = geom
        return geom

    verts, faces, off = [], [], 0
    n_inst = n_base = n_hidden = 0
    for inst in deps.object_instances:
        ob = inst.object
        if ob is None or ob.type != "MESH" or ob.data is None:
            continue
        holder = inst.parent if inst.is_instance else ob
        if holder is not None and holder.name.startswith(tuple(exclude_prefix)):
            continue
        # Skip geometry hidden from the RENDER. Layout placeholders (Boulder/asset
        # bbox_placeholder & spawn_placeholder, etc.) and other helper meshes live
        # in the viewport depsgraph the BVH is built from but never render, so
        # raycasts hit them as "phantom" near-surfaces the rendered depth doesn't
        # have -- which made the placement parallax check (panoramic_stats) read
        # far more near content than the panorama actually shows. Checking the
        # HOLDER (instancer for instances, the object itself for base meshes) keeps
        # rendered geometry-node instances whose hidden SOURCE mesh is instanced.
        if holder is not None and holder.hide_render:
            n_hidden += 1
            continue
        geom = _local(ob)
        if geom is None:
            continue
        lv, lf = geom
        M = np.array(inst.matrix_world, dtype=np.float64)
        verts.append(lv @ M[:3, :3].T + M[:3, 3])
        faces.extend((a + off, b + off, c + off) for a, b, c in lf)
        off += len(lv)
        n_inst += inst.is_instance
        n_base += not inst.is_instance
    if not verts:
        raise ValueError("build_instance_aware_bvh found no geometry")
    allv = np.concatenate(verts, axis=0)
    logger.info(
        f"BVH: {n_base} base + {n_inst} instances "
        f"(skipped {n_hidden} render-hidden) -> {len(allv)} verts, {len(faces)} tris"
    )
    return BVHTree.FromPolygons(allv.tolist(), faces, all_triangles=True), allv


def panoramic_stats(origin, bvh, near_dist=8.0, horiz_deg=8.0,
                    n_theta=24, n_phi=48, sky_dist=1e4, n_band=16):
    """Cast rays from `origin` and return ``(sky_frac, near_frac)``:
      - sky_frac: fraction of a coarse full-sphere set of rays that miss geometry
        (open sky) -- yaw-independent, used to reject fully enclosed poses.
      - near_frac: fraction of the HORIZONTAL band (within `horiz_deg` of the
        equator) that hits geometry within `near_dist` m -- the parallax-giving
        content at eye level (the ground straight down is always 'near' but gives
        no parallax, and the sky straight up is empty).

    The band is sampled with its OWN dense set of `n_band` elevation rings across
    +/-horiz_deg so this raycast measure matches build_multiview_manifest's
    _anchor_near_frac, which averages the rendered depth over the same +/-8 deg
    band. Sampling only 2 coarse rings (as a shared n_theta grid would) sits right
    at the horizon and over-reads near content versus the rendered panorama.
    """
    origin = Vector(origin)
    phis = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi

    # sky_frac: coarse full sphere.
    thetas = np.pi * (np.arange(n_theta) + 0.5) / n_theta
    miss = 0
    for th in thetas:
        st, ct = np.sin(th), np.cos(th)
        for ph in phis:
            _, _, _, d = bvh.ray_cast(
                origin, Vector((st * np.cos(ph), st * np.sin(ph), ct))
            )
            if d is None or d > sky_dist:
                miss += 1
    sky_frac = miss / (n_theta * n_phi)

    # near_frac: dense horizontal band, matching the manifest's +/-horiz_deg band.
    band_th = np.radians(90.0 + np.linspace(-horiz_deg, horiz_deg, n_band))
    near = tot = 0
    for th in band_th:
        st, ct = np.sin(th), np.cos(th)
        for ph in phis:
            _, _, _, d = bvh.ray_cast(
                origin, Vector((st * np.cos(ph), st * np.sin(ph), ct))
            )
            tot += 1
            if d is not None and d <= sky_dist and d < near_dist:
                near += 1
    return sky_frac, (near / tot if tot else 0.0)


# --------------------------------------------------------------------------- #
# Placement.
# --------------------------------------------------------------------------- #
def sampling_bounds(all_verts, gen_cam_locs, radius, central):
    """XY box + z range to sample anchors in. Prefer a box of +/- `radius` m
    around the generation camera(s) -- where assets were populated -- and fall
    back to the central `central` fraction of the scene if there were none."""
    lo, hi = all_verts.min(0), all_verts.max(0)
    if gen_cam_locs:
        cx = float(np.mean([l[0] for l in gen_cam_locs]))
        cy = float(np.mean([l[1] for l in gen_cam_locs]))
        x0, x1 = max(lo[0], cx - radius), min(hi[0], cx + radius)
        y0, y1 = max(lo[1], cy - radius), min(hi[1], cy + radius)
    else:
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        hx, hy = (hi[0] - lo[0]) / 2 * central, (hi[1] - lo[1]) / 2 * central
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    return (float(x0), float(x1), float(y0), float(y1), float(lo[2]), float(hi[2]))


def find_and_place_anchor(rig, cams, bvh, bounds, altitude, min_clear, min_sky,
                          min_near, near_dist, tries):
    """Move `rig` (and its sub-cameras `cams`) to a valid anchor. Returns the
    anchor's near_frac (>= 0) on success, or None if no valid pose was found.
    A pose is valid when:
      - every sub-camera is >= min_clear m from any surface (not underground /
        not inside a prop),
      - the anchor sees >= min_sky open sky, AND
      - >= min_near of the horizontal band has content within near_dist m, so the
        panorama has close objects that actually produce parallax (rejects empty
        desert/ocean-horizon poses)."""
    x0, x1, y0, y1, zlo, zhi = bounds
    for _ in range(tries):
        x, y = np.random.uniform(x0, x1), np.random.uniform(y0, y1)
        hit, _, _, _ = bvh.ray_cast(Vector((x, y, zhi + 50.0)), Vector((0, 0, -1)))
        if hit is None:
            continue
        z = hit.z + cam_util.random_general(altitude)
        rig.location = Vector((x, y, z))
        rig.rotation_euler = (np.deg2rad(90.0), 0.0, np.random.uniform(-np.pi, np.pi))
        bpy.context.view_layer.update()

        clear = True
        for cam in cams:
            _, _, _, dist = bvh.find_nearest(cam.matrix_world.translation)
            if dist is not None and dist < min_clear:
                clear = False
                break
        if not clear:
            continue
        sky, near = panoramic_stats(cams[0].matrix_world.translation, bvh, near_dist)
        if sky < min_sky or near < min_near:
            continue
        return near  # placed; return the anchor's near_frac (>= 0) so the caller
        #              can account for it against the per-scene sparse budget
    return None


def spawn_rig(rig_id, offsets):
    """Create a rig empty `camrig.<id>` with cameras `camera_<id>_<j>`."""
    parent = butil.spawn_empty(f"camrig.{rig_id}")
    cams = []
    for j, off in enumerate(offsets):
        cam = cam_util.spawn_camera()
        cam.name = f"camera_{rig_id}_{j}"
        cam.parent = parent
        cam.location = off["loc"]
        cam.rotation_euler = off["rot_euler"]
        cams.append(cam)
    return parent, cams


# --------------------------------------------------------------------------- #
# Rendering (reuses infinigen's full/flat render_image passes).
# --------------------------------------------------------------------------- #
def _render_pass(cam, frames_dir, resolution, samples, passes, flat, ovr, keep):
    """Render ONE pass of one camera, then flatten the wanted channels into
    frames_dir/<Channel>/ (no per-camera subfolder -- suffixes are unique, so a
    whole scene's views for a modality share one folder for easy manipulation)."""
    scene = bpy.context.scene
    scene.cycles.samples = samples
    pass_root = frames_dir.parent / "_pass_stage"
    stage = pass_root / "frames_stage"
    stage.mkdir(parents=True, exist_ok=True)
    render_mod.render_image(
        camera=cam,
        frames_folder=stage,
        passes_to_save=passes,
        flat_shading=flat,
        render_resolution_override=resolution,
        override_num_samples=ovr,
    )  # render_image reorganizes `stage` into `stage.parent/frames/<Channel>/camera_<s>/`
    pass_frames = pass_root / "frames"
    for ch in keep:
        src = pass_frames / ch
        if not src.is_dir():
            continue
        dst = frames_dir / ch
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.rglob("*"):  # flatten: pull every file up into frames_dir/<ch>/
            if entry.is_file():
                shutil.move(str(entry), str(dst / entry.name))
    shutil.rmtree(pass_root, ignore_errors=True)


# The GT pass uses global_flat_shading(), which PERMANENTLY swaps the scene's
# materials to clay and never reverts them -- so every beauty render after the
# first GT pass would come out flat-shaded (looking like segmentation). Render is
# therefore split into two phases at the SCENE level: all beauty passes first
# (materials still original), then all GT passes.
_BEAUTY_VT_LOGGED = False


def _set_view_transform(*candidates):
    """Set the first available view transform from `candidates` and return it."""
    vs = bpy.context.scene.view_settings
    for vt in candidates:
        try:
            vs.view_transform = vt
            return vt
        except TypeError:  # not a valid enum in this Blender build
            continue
    return vs.view_transform


def render_beauty(cam, frames_dir, resolution, samples):
    """Beauty pass: textured RGB (Image) + camera pose (camview). Rendered with a
    filmic tone map (AgX, else Filmic) so bright outdoor scenes with a strong sun
    don't clip to white -- Standard has no highlight rolloff. Depth/normals are raw
    geometry passes, so this view transform doesn't affect the ground truth."""
    vt = _set_view_transform("AgX", "Filmic", "Standard")
    global _BEAUTY_VT_LOGGED
    if not _BEAUTY_VT_LOGGED:
        logger.info(f"beauty pass view transform: {vt}")
        _BEAUTY_VT_LOGGED = True
    _render_pass(cam, frames_dir, resolution, samples,
                 passes=[], flat=False, ovr=None, keep=("Image", "camview"))


def render_gt(cam, frames_dir, resolution, samples):
    """Ground-truth pass (flat-shaded): metric depth + surface normals."""
    _set_view_transform("Standard")  # data-safe (irrelevant to the raw passes)
    _render_pass(cam, frames_dir, resolution, samples,
                 passes=[("z", "Depth"), ("normal", "Normal")], flat=True, ovr=16,
                 keep=("Depth", "SurfaceNormal"))


def set_data_color_management():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.gamma = 1.0
    vs.exposure = 0.0


def harvest_scene(blend_path, out_root, args):
    scene_name = blend_path.parent.parent.name if blend_path.parent.name in ("fine", "coarse") else blend_path.stem
    out_dir = out_root / scene_name
    logger.info(f"=== scene {scene_name}: loading {blend_path} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    # Infinigen populates high-res assets only around the generation camera(s)
    # (within dist_cull). Remember where they were so we sample new anchors there
    # -- otherwise a harvested panorama would look out over bare terrain.
    gen_cam_locs = [
        o.matrix_world.translation.copy() for o in bpy.data.objects if o.type == "CAMERA"
    ]
    # Remove the generation rig so our harvested rigs get clean, collision-free
    # camera_<rig>_<subcam> names (Blender would otherwise suffix them ".001",
    # breaking get_id()).
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA" or obj.name.startswith(("camrig", "camera_")):
            bpy.data.objects.remove(obj, do_unlink=True)
    set_data_color_management()
    scene = bpy.context.scene
    scene.frame_start = scene.frame_end = 0
    scene.render.resolution_x, scene.render.resolution_y = args.resolution

    bvh, all_verts = build_instance_aware_bvh()
    bounds = sampling_bounds(all_verts, gen_cam_locs, args.sample_radius, args.central)

    # Placement PREFERS high-parallax anchors (near_frac >= min_near), but keeps a
    # small per-scene budget of low-parallax ("sparse") rigs so the dataset still
    # contains SOME open/parallax-free views without letting them dominate. On a
    # rich scene every rig is rich (0 sparse); on a sparse scene only the budgeted
    # rigs get placed, so sparse scenes contribute few views overall.
    n_sparse_allowed = int(np.ceil(args.sparse_frac * args.rigs_per_scene))
    n_sparse = 0
    placed = []
    for k in range(args.rigs_per_scene):
        cfg = cam_util.multiview_rig_config(
            n_views=args.n_views,
            baseline=args.baseline,
            pattern=args.pattern,
            z_amplitude=args.z_amplitude,
        )
        rig, cams = spawn_rig(k, cfg)
        # First insist on a rich anchor; only if none is found AND we still have
        # sparse budget, fall back to accepting any (min_near=0) anchor.
        near = find_and_place_anchor(
            rig, cams, bvh, bounds, args.altitude,
            args.min_clearance, args.min_sky, args.min_near, args.near_dist,
            args.place_tries,
        )
        if near is None and n_sparse < n_sparse_allowed:
            near = find_and_place_anchor(
                rig, cams, bvh, bounds, args.altitude,
                args.min_clearance, args.min_sky, 0.0, args.near_dist,
                args.place_tries,
            )
        if near is None:
            logger.warning(f"  rig {k}: no valid anchor in {args.place_tries} tries (skipped)")
            butil.delete([rig, *cams])
            continue
        sparse = near < args.min_near
        n_sparse += sparse
        b = np.linalg.norm(cams[1].matrix_world.translation - cams[0].matrix_world.translation)
        logger.info(f"  rig {k}: placed (baseline~{b:.2f} m, near_frac={near:.2f}"
                    f"{', SPARSE' if sparse else ''})")
        placed.append((rig, cams))

    if not placed:
        logger.warning(f"  scene {scene_name}: no rigs placed, nothing to render")
        return 0

    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    res = tuple(args.resolution)
    # Two phases (see render_beauty/render_gt): every beauty first while materials
    # are original, THEN every GT once flat-shading has been applied scene-wide.
    for rig, cams in placed:
        for cam in cams:
            logger.info(f"  beauty {cam.name}")
            render_beauty(cam, frames, res, args.samples)
    for rig, cams in placed:
        for cam in cams:
            logger.info(f"  gt {cam.name}")
            render_gt(cam, frames, res, args.samples)
    logger.info(f"  scene {scene_name}: rendered {len(placed)} rig(s), "
                f"{sum(len(c) for _, c in placed)} views ({n_sparse} sparse)")
    return len(placed)


def _spec(s):
    """Parse '0.5' -> 0.5 or 'uniform,0.3,0.7' -> ('uniform', 0.3, 0.7)."""
    parts = str(s).split(",")
    if len(parts) == 1:
        return float(parts[0])
    return (parts[0], *[float(p) for p in parts[1:]])


def iter_scene_blends(root):
    root = Path(root)
    if root.is_file() and root.suffix == ".blend":
        yield root
        return
    # prefer fine/scene.blend, else any scene.blend under the tree
    fines = sorted(root.glob("*/fine/scene.blend"))
    yield from (fines or sorted(root.glob("**/scene.blend")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-blend", type=Path, help="a single fine/scene.blend")
    src.add_argument("--scenes-root", type=Path, help="dir of generated scenes (uses */fine/scene.blend)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rigs-per-scene", type=int, default=4)
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--baseline", type=_spec, default=("uniform", 0.3, 0.7),
                    help="metres; float or 'uniform,lo,hi' (drawn per rig)")
    ap.add_argument("--pattern", default="ring", choices=["ring", "sphere"])
    ap.add_argument("--z-amplitude", type=float, default=0.2)
    ap.add_argument("--altitude", type=_spec, default=("uniform", 0.75, 1.75),
                    help="metres above ground; float or 'uniform,lo,hi'")
    ap.add_argument("--resolution", type=lambda s: [int(x) for x in s.split(",")], default=[4096, 2048])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--sample-radius", type=float, default=30.0,
                    help="sample anchors within this many metres of the generation "
                    "camera(s), where assets are populated (keep < dist_cull ~70 m)")
    ap.add_argument("--central", type=float, default=0.6,
                    help="fallback: central XY fraction if the scene had no cameras")
    ap.add_argument("--min-clearance", type=float, default=0.3, help="min metres from any surface per sub-camera")
    ap.add_argument("--min-sky", type=float, default=0.1, help="min open-sky fraction at the anchor")
    ap.add_argument("--min-near", type=float, default=0.2,
                    help="a rig is 'rich' when this fraction of the horizontal band "
                    "has content within --near-dist; rich anchors are always kept")
    ap.add_argument("--sparse-frac", type=float, default=0.2,
                    help="max fraction of a scene's rigs allowed to be low-parallax "
                    "(near_frac < --min-near). Keeps SOME sparse/open views in the "
                    "dataset without letting them dominate; 0 = rich only")
    ap.add_argument("--near-dist", type=float, default=8.0,
                    help="metres: content closer than this counts as parallax-giving")
    ap.add_argument("--place-tries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    blends = list(iter_scene_blends(args.scene_blend or args.scenes_root))
    if not blends:
        raise SystemExit(f"No scene.blend found under {args.scene_blend or args.scenes_root}")

    total = 0
    for blend in blends:
        try:
            total += harvest_scene(blend, args.output, args)
        except Exception as e:
            logger.error(f"scene {blend} failed: {e}")

    # Build manifests (+ prune) across everything we produced.
    if total:
        logger.info("Building per-set manifests...")
        from build_multiview_manifest import build_scene_manifests, prune_scene
        for scene_dir in sorted(p for p in args.output.iterdir() if (p / "frames").is_dir()):
            build_scene_manifests(scene_dir)
            prune_scene(scene_dir)
    logger.info(f"Done: {total} multi-view rig(s) across {len(blends)} scene(s).")


if __name__ == "__main__":
    main()
