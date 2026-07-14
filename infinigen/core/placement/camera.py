# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Lahav Lipson: Stationary camera selection
# - Alexander Raistrick: Refactor into proposal/validate, camera animation
# - Lingjie Mei: get_camera_trajectory


import logging
import typing
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from mathutils import Matrix
import bpy
import gin
import imageio
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from numpy.random import uniform as U
from tqdm import tqdm

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.rendering.post_render import colorize_depth
from infinigen.core.tagging import tag_system
from infinigen.core.util import blender as butil
from infinigen.core.util import camera
from infinigen.core.util.blender import SelectObjects, delete
from infinigen.core.util.logging import Timer
from infinigen.core.util.organization import SelectionCriterions
from infinigen.core.util.random import random_general
from infinigen.terrain.core import Terrain
from infinigen.tools.suffixes import get_suffix

from . import animation_policy

logger = logging.getLogger(__name__)

def adjust_camera_sensor(cam):
    scene = bpy.context.scene
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    sensor_width = 18 * (W / H)
    # assert sensor_width.is_integer(), (18, W, H)
    cam.data.sensor_height = 18
    cam.data.sensor_width = int(sensor_width)

@gin.configurable
def get_sensor_coords(cam, H, W, sparse=False):
    adjust_camera_sensor(cam)
    camd = cam.data
    f_in_m = camd.lens / 1000
    scene = bpy.context.scene
    resolution_x_in_px = W
    resolution_y_in_px = H

    scale = scene.render.resolution_percentage / 100
    sensor_width_in_m = camd.sensor_width / 1000
    sensor_height_in_m = camd.sensor_height / 1000

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = (
            resolution_x_in_px * scale / sensor_width_in_m / pixel_aspect_ratio
        )  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_m

    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_m
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_m

    u_0 = resolution_x_in_px * scale / 2  # cx (in pixels) Usually is just W/2
    v_0 = resolution_y_in_px * scale / 2  # cx (in pixels) Usually is just H/2
    xx, yy = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
    coords_x = (xx - u_0) / s_u  # relative, in mm
    coords_y = (yy - v_0 + 1) / s_v  # relative, in mm

    coords_z = np.full(coords_x.shape, -f_in_m)
    relative_cam_coords = np.stack((coords_x, coords_y, coords_z), axis=-1)

    cam_coords_vectors = np.empty((H, W), dtype=Vector)
    pixel_locs = np.stack((np.meshgrid(np.arange(W), np.arange(H))), axis=-1).reshape(
        (W * H, 2)
    )  # np.array(list(product(range(H), range(W))))
    if sparse:
        ii = np.random.choice(H * W, size=1000)
        pixel_locs = pixel_locs[ii]

    for x, y in tqdm(pixel_locs, desc="Building Camera Vectors", disable=True):
        pixelVector = Vector(relative_cam_coords[y, x])
        cam_coords_vectors[y, x] = cam.matrix_world @ pixelVector

    return cam_coords_vectors, pixel_locs

def spawn_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.view_layer.objects.active = cam
    cam.data.clip_end = 1e4
    adjust_camera_sensor(cam)
    return cam


def cam_name(cam_rig, subcam):
    return f"camera_{cam_rig}_{subcam}"


def get_id(camera: bpy.types.Object):
    _, rig, subcam = camera.name.split("_")
    return int(rig), int(subcam)


@gin.configurable
def multiview_rig_config(
    n_views=6,
    baseline=0.3,
    pattern="ring",
    z_amplitude=0.0,
    level_pitch_deg=90.0,
    converge_dist=0.0,
):
    """Build a ``camera_rig_config`` describing a small multi-view constellation
    of cameras for panoramic Gaussian-Splatting training data.

    One scene is captured by a single camera *rig* whose sub-cameras form the
    constellation: a central "anchor" camera plus surrounding cameras a short
    ``baseline`` (in metres) away. Because each sub-camera is rendered as a full
    360° equirectangular panorama, even a modest baseline yields useful parallax
    while every view still observes the same scene content — exactly what a GS
    head needs for multi-view supervision of a monocular input.

    All sub-cameras keep the rig orientation (``rot_euler`` offset of 0) so their
    panoramas are consistently oriented; only the positions differ, and each
    sub-camera's absolute pose is saved by :func:`save_camera_parameters`, so the
    relative extrinsics between views are recoverable.

    IMPORTANT — pitch compensation: ``spawn_camera_rigs`` applies these offsets in
    the rig's LOCAL frame, and the rig is rotated to look at the horizon
    (``camera.camera_pose_proposal.pitch`` ≈ 90°). A raw local offset would then
    be tilted, mapping the neighbour ring partly onto the world vertical axis —
    at a large baseline that would drive sub-cameras up to ±baseline metres
    vertically (i.e. underground). We therefore build the desired constellation
    in a level world frame and pre-rotate it by the inverse of the rig pitch, so
    that once the rig tilts, the spread ends up HORIZONTAL in world space
    (only ``z_amplitude`` contributes vertical parallax). ``level_pitch_deg`` must
    match ``camera.camera_pose_proposal.pitch``.

    NOTE: ``n_views`` here must equal ``iterate_scene_tasks.n_subcams`` in the
    datagen pipeline config, otherwise the render loop will not iterate every
    sub-camera. The launcher keeps the two in sync from a single variable.

    Args:
        n_views: total number of cameras (1 anchor + ``n_views-1`` neighbours).
        baseline: distance of the neighbour cameras from the anchor, in metres.
            Accepts a ``random_general`` spec, e.g. ``("uniform", 0.3, 0.7)``, in
            which case a FRESH baseline is drawn for each generated scene. Varying
            the baseline across the dataset stops the GS head from overfitting to
            a single parallax and helps it generalise to different view spacings;
            a plain float keeps it fixed.
        pattern: ``"ring"`` places neighbours on a (world) horizontal circle;
            ``"sphere"`` spreads them over a Fibonacci upper-hemisphere.
        z_amplitude: for ``"ring"``, alternate neighbour cameras up/down by this
            amount (metres) to add vertical parallax. Ignored for ``"sphere"``.
            Also accepts a ``random_general`` spec.
        level_pitch_deg: pitch (deg) the rig will be placed at; used to keep the
            constellation world-horizontal. Must match camera_pose_proposal.pitch.

    Returns:
        list[dict] with ``loc``/``rot_euler`` keys, consumable by
        :func:`spawn_camera_rigs`.
    """
    if n_views < 1:
        raise ValueError(f"multiview_rig_config needs n_views>=1, got {n_views}")

    # Draw the baseline / vertical amplitude once per scene (this runs inside the
    # per-scene worker with its seed set), so a ("uniform", lo, hi) spec yields a
    # different -- but reproducible -- baseline for every scene.
    baseline = random_general(baseline)
    z_amplitude = random_general(z_amplitude)

    # Desired offsets in a LEVEL world frame (x,y horizontal, z up).
    world_offsets = [(0.0, 0.0, 0.0)]  # anchor
    m = n_views - 1
    if pattern == "sphere":
        # Fibonacci hemisphere: even angular spread over the upper half sphere.
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(m):
            z = (i + 0.5) / m  # (0, 1) -> upper hemisphere only
            r = np.sqrt(max(0.0, 1.0 - z * z))
            theta = golden * i
            world_offsets.append(
                (baseline * r * np.cos(theta), baseline * r * np.sin(theta), baseline * z)
            )
    elif pattern == "ring":
        for i in range(m):
            theta = 2.0 * np.pi * i / m
            z = z_amplitude * (1.0 if i % 2 == 0 else -1.0) if z_amplitude else 0.0
            world_offsets.append((baseline * np.cos(theta), baseline * np.sin(theta), z))
    else:
        raise ValueError(f"multiview_rig_config: unknown pattern {pattern!r}")

    # Pre-rotate by Rx(-pitch) so that after the rig applies Rx(pitch) the spread
    # is world-horizontal. (Yaw about world-Z, applied on top, keeps it level.)
    p = np.deg2rad(level_pitch_deg)
    cp, sp = np.cos(p), np.sin(p)
    cd = random_general(converge_dist)
    # Anchor looks along its local -Z; a point converge_dist metres ahead of it is
    # the convergence target. Neighbours toe IN toward that target (rotate toward
    # the scene the anchor observes, never away), which also keeps the neighbour
    # panoramas' framing consistent with the anchor's. converge_dist=0 -> all
    # cameras share the anchor's orientation (no toe-in).
    target = Vector((0.0, 0.0, -cd)) if cd and cd > 0 else None
    cfg = []
    for idx, (wx, wy, wz) in enumerate(world_offsets):
        loc = (
            float(wx),
            float(cp * wy + sp * wz),
            float(-sp * wy + cp * wz),
        )
        if idx == 0 or target is None:
            rot = (0.0, 0.0, 0.0)
        else:
            # YAW-ONLY convergence. The rig's level pitch maps the camera's local +Y
            # to world up, so a rotation about local +Y is a pure YAW about the world
            # vertical -- the equirect HORIZON STAYS PERFECTLY LEVEL. We rotate the
            # neighbour's forward (-Z) toward only the HORIZONTAL bearing of the
            # convergence target and drop its vertical component. Using the full 3D
            # minimal rotation (rotation_difference) instead mixes in a local X/Z
            # component, i.e. a roll/pitch that tilts the panorama horizon -- which is
            # exactly the "weird rotation" we must avoid; a 360 view only needs yaw.
            direction = (target - Vector(loc)).normalized()
            yaw = float(np.arctan2(-direction.x, -direction.z))
            rot = (0.0, yaw, 0.0)
        cfg.append({"loc": loc, "rot_euler": rot})

    return cfg


@gin.configurable
def spawn_camera_rigs(
    camera_rig_config,
    n_camera_rigs,
) -> list[bpy.types.Object]:
    rigs_col = butil.get_collection("camera_rigs")
    cams_col = butil.get_collection("cameras")

    def spawn_rig(i):
        rig_parent = butil.spawn_empty(f"camrig.{i}")
        butil.put_in_collection(rig_parent, rigs_col)

        for j, config in enumerate(camera_rig_config):
            cam = spawn_camera()
            cam.name = cam_name(i, j)
            cam.parent = rig_parent
            cam.location = config["loc"]
            cam.rotation_euler = config["rot_euler"]

            butil.put_in_collection(cam, cams_col)

        return rig_parent

    return [spawn_rig(i) for i in range(n_camera_rigs)]


def get_camera_rigs() -> list[bpy.types.Object]:
    if "camera_rigs" not in bpy.data.collections:
        raise ValueError("No camera rigs found")

    result = list(bpy.data.collections["camera_rigs"].objects)

    for i, rig in enumerate(result):
        for j, child in enumerate(rig.children):
            expected = cam_name(i, j)
            if child.name != expected:
                raise ValueError(f"child {i=} {j}  was {child.name=}, {expected=}")

    return result


@node_utils.to_nodegroup(
    "nodegroup_camera_info", singleton=True, type="GeometryNodeTree"
)
def nodegroup_active_cam_info(nw: NodeWrangler):
    info = nw.new_node(Nodes.ObjectInfo, [bpy.context.scene.camera])
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={k: info.outputs[k] for k in info.outputs.keys()},
    )


def set_active_camera(camera: bpy.types.Object):
    bpy.context.scene.camera = camera

    ng = (
        nodegroup_active_cam_info()
    )  # does not create a new node group, retrieves singleton
    ng.nodes["Object Info"].inputs["Object"].default_value = camera

    return bpy.context.scene.camera


def terrain_camera_query(
    cam: bpy.types.Object,
    scene_bvh: BVHTree,
    terrain_tags_queries,
    vertexwise_min_dist,
    min_dist=0,
):
    dists = []
    H = bpy.context.scene.render.resolution_y
    W = bpy.context.scene.render.resolution_x
    sensor_coords, pix_it = get_sensor_coords(cam, H=H, W=W, sparse=True)
    terrain_tags_queries_counts = {q: 0 for q in terrain_tags_queries}

    for x, y in pix_it:
        direction = (sensor_coords[y, x] - cam.matrix_world.translation).normalized()
        _, _, index, dist = scene_bvh.ray_cast(cam.matrix_world.translation, direction)
        if dist is None:
            continue
        dists.append(dist)
        if dist < min_dist or (
            vertexwise_min_dist is not None and dist < vertexwise_min_dist[index]
        ):
            logger.debug(f"Found {dist=} < {min_dist=}")
            dists = None  # means dist < min
            break
        for q in terrain_tags_queries:
            terrain_tags_queries_counts[q] += terrain_tags_queries[q][index]

    n_pix = pix_it.shape[0]

    return dists, terrain_tags_queries_counts, n_pix


@dataclass
class CameraProposal:
    loc: np.array
    rot: np.array
    focal_length: float

    def apply(self, cam_rig):
        cam_rig.location = self.loc
        cam_rig.rotation_euler = self.rot

        if self.focal_length is not None:
            for cam in cam_rig.children:
                cam.data.lens = self.focal_length


@gin.configurable
def camera_pose_proposal(
    scene_bvh,
    location_sample: typing.Callable | tuple,
    center_coordinate=None,
    radius=None,
    bbox=None,
    altitude=("uniform", 1.5, 2.5),
    roll=0,
    yaw=("uniform", -180, 180),
    pitch=90,
    focal_length=50,
    override_loc=None,
):
    if isinstance(location_sample, tuple):
        location_sample = Vector(location_sample)

        def location_sample():
            return location_sample

    if override_loc is not None:
        loc = Vector(random_general(override_loc))
    elif center_coordinate:
        while True:
            # Define the radius of the circle
            random_angle = np.random.uniform(2 * np.math.pi)
            xoff = np.random.uniform(-radius / 10, radius / 10)
            yoff = np.random.uniform(-radius / 10, radius / 10)
            zoff = random_general(altitude)
            loc = Vector([0, 0, 0])
            loc[0] = center_coordinate[0] + radius * np.math.cos(random_angle) + xoff
            loc[1] = center_coordinate[1] + radius * np.math.sin(random_angle) + yoff
            loc[2] = center_coordinate[2] + zoff
            if bbox is not None:
                out_of_bbox = False
                for i in range(3):
                    if loc[i] < bbox[0][i] or loc[i] > bbox[1][i]:
                        out_of_bbox = True
                        break
                if out_of_bbox:
                    continue
            hit, *_ = scene_bvh.ray_cast(
                loc,
                Vector(center_coordinate) - loc,
                (Vector(center_coordinate) - loc).length,
            )
            if hit is None:
                break
    elif altitude is None:
        loc = location_sample()
    else:
        loc = location_sample()
        curr_alt = animation_policy.get_altitude(loc, scene_bvh)
        if curr_alt is None:
            logger.debug(f"camera_pose_proposal got {curr_alt=} for {loc=}")
            # butil.spawn_empty("fail")
            return None
        desired_alt = random_general(altitude)
        loc[2] = loc[2] + desired_alt - curr_alt

    if center_coordinate:
        direction = loc - Vector(center_coordinate)
        direction = Vector(direction)
        rotation_matrix = direction.to_track_quat("Z", "Y").to_matrix()
        rotation_euler = rotation_matrix.to_euler("XYZ")
        roll, pitch, yaw = rotation_euler
        noise_range = np.deg2rad(5.0)  # 5 degrees of noise in radians
        # Add random noise to roll, pitch, and yaw
        roll += np.random.uniform(-noise_range, noise_range)
        pitch += np.random.uniform(-noise_range, noise_range)
        yaw += np.random.uniform(-noise_range, noise_range)
        rot = np.array([roll, pitch, yaw])
    else:
        rot = np.deg2rad(
            [random_general(pitch), random_general(roll), random_general(yaw)]
        )
    focal_length = random_general(focal_length)
    return CameraProposal(loc, rot, focal_length)


@gin.configurable
def keep_cam_pose_proposal(
    cam: bpy.types.Object,
    terrain: Terrain,
    scene_bvh: BVHTree,
    placeholders_kd,
    camera_selection_answers,
    vertexwise_min_dist,
    camera_selection_ratio,
    min_placeholder_dist=0,
    min_terrain_distance=0,
    terrain_coverage_range=(0.5, 1),
):
    if terrain is not None:  # TODO refactor
        terrain_sdf = terrain.compute_camera_space_sdf(
            np.array(cam.matrix_world.translation).reshape((1, 3))
        )

    if not cam.type == "CAMERA":
        raise ValueError(f"{cam.name=} had {cam.type=}")

    bpy.context.view_layer.update()

    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(f"keep_cam_pose_proposal rejects {dist_to_placeholder=}, {v, i}")
        return "placeholder"

    dists, camera_selection_answers_counts, n_pix = terrain_camera_query(
        cam,
        scene_bvh,
        camera_selection_answers,
        vertexwise_min_dist,
        min_dist=min_terrain_distance,
    )

    if dists is None:
        logger.debug("keep_cam_pose_proposal rejects terrain dists")
        return "min_dist"

    coverage = len(dists) / n_pix
    if terrain_coverage_range is not None and (
        coverage < terrain_coverage_range[0]
        or coverage > terrain_coverage_range[1]
        or coverage == 0
    ):
        logger.debug(f"keep_cam_pose_proposal rejects {coverage=} for {terrain_coverage_range=}")
        return f"coverage={coverage:.3f}"

    if terrain is not None and terrain_sdf <= 0:
        logger.debug(f"keep_cam_pose_proposal rejects {terrain_sdf=}")
        return "terrain_sdf"

    if rparams := camera_selection_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == SelectionCriterions.CloseUp:
                closeup = len([d for d in dists if d < q[1]]) / n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    logger.debug(f"keep_cam_pose_proposal rejects {closeup=} for {q=}")
                    return f"closeup={closeup:.3f}"
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in camera_selection_answers_counts:
                    ratio = camera_selection_answers_counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        logger.debug(f"keep_cam_pose_proposal rejects {ratio=} for {q=}")
                        return f"selection_ratio={ratio:.3f}"

    try:
        res = np.std(dists) + 1.5 * np.min(dists)
    except ValueError:
        logger.debug("Dists empty.")
        res = 0

    return res


@gin.configurable
class AnimPolicyGoToProposals:
    def __init__(
        self, speed=("uniform", 1.5, 2.5), min_dist=4, max_dist=10, retries=30
    ):
        self.speed = speed
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.retries = retries

    def __call__(self, camera_rig, frame_curr, retry_pct, bvh):
        margin = Vector((self.max_dist, self.max_dist, self.max_dist))
        bbox = (camera_rig.location - margin, camera_rig.location + margin)

        for _ in range(self.retries):
            res = camera_pose_proposal(
                scene_bvh=bvh,
                location_sample=lambda: np.random.uniform(*bbox),
            )
            if res is None:
                continue
            dist = np.linalg.norm(np.array(res.loc) - np.array(camera_rig.location))
            if dist < self.min_dist:
                continue
            break
        else:
            raise animation_policy.PolicyError(
                f"{__name__} found no keyframe after {self.retries=}"
            )

        time = dist / random_general(self.speed)
        return Vector(res.loc), Vector(res.rot), time, "BEZIER"


@gin.configurable
def compute_base_views(
    camera_rig: bpy.types.Object,
    n_views: int,
    terrain,
    scene_bvh: BVHTree,
    location_sample: typing.Callable,
    center_coordinate=None,
    radius=None,
    bbox=None,
    placeholders_kd=None,
    min_candidates_ratio=5,
    max_tries=30000,
    visualize=False,
    panoramic_enclosure_check=False,
    min_pano_near_frac=0.0,
    sky_visibility_check=True,
    allow_fewer=False,
    **kwargs,
):
    import time as _time
    from collections import Counter as _Counter

    potential_views = []
    n_min_candidates = int(min_candidates_ratio * n_views)
    rejection_counts = _Counter()
    t_start = _time.perf_counter()

    with tqdm(total=n_min_candidates, desc="Searching for camera viewpoints") as pbar:
        for it in range(1, max_tries):
            if center_coordinate:
                props = camera_pose_proposal(
                    scene_bvh=scene_bvh,
                    location_sample=location_sample,
                    center_coordinate=center_coordinate,
                    radius=random_general(radius),
                    bbox=bbox,
                )
            else:
                props = camera_pose_proposal(
                    scene_bvh=scene_bvh, location_sample=location_sample
                )

            if props is None:
                logger.debug(f"{camera_pose_proposal.__name__} returned {props=} for {it=}")
                rejection_counts["pose_proposal=None"] += 1
                if it % 500 == 0:
                    elapsed = _time.perf_counter() - t_start
                    logger.info(
                        f"compute_base_views: {it}/{max_tries} tries, "
                        f"{len(potential_views)}/{n_min_candidates} candidates found "
                        f"({elapsed:.0f}s elapsed). Rejections: {dict(rejection_counts)}"
                    )
                continue

            props.apply(camera_rig)

            all_scores = []
            for cam in camera_rig.children:
                score = keep_cam_pose_proposal(
                    cam,
                    terrain,
                    scene_bvh,
                    placeholders_kd,
                    **kwargs,
                )
                all_scores.append(score)

            if any(isinstance(s, str) for s in all_scores):
                # keep_cam_pose_proposal returned a rejection reason string
                for s in all_scores:
                    if isinstance(s, str):
                        rejection_counts[s] += 1
                criterion = None
            elif any(s is None for s in all_scores):
                rejection_counts["score=None"] += 1
                criterion = None
            else:
                criterion = np.mean(all_scores)

            if it % 500 == 0:
                elapsed = _time.perf_counter() - t_start
                logger.info(
                    f"compute_base_views: {it}/{max_tries} tries, "
                    f"{len(potential_views)}/{n_min_candidates} candidates found "
                    f"({elapsed:.0f}s elapsed). Rejections: {dict(rejection_counts)}"
                )

            if visualize:
                criterion_str = f"{criterion:.2f}" if criterion is not None else "None"
                marker = butil.spawn_empty(f"attempt_{it}_{criterion_str}")
                marker.location = camera_rig.location
                marker.rotation_euler = camera_rig.rotation_euler

            if criterion is None:
                logger.debug(f"{it=} {criterion=}")
                continue

            # Sky-visibility check: raycast UPWARD only. Horizontal rays are
            # unreliable — wide buildings have walls 50-100m away, giving false
            # "open" readings. This catches OUTDOOR cameras stuck inside/under a
            # building, but indoors the ceiling is always overhead, so it would
            # reject every valid pose -- disable it for indoor (sky_visibility_check
            # = False) where the room itself is the intended enclosure.
            if sky_visibility_check:
                MAX_CEILING_DIST = 200.0  # no real building taller than this
                MIN_OPEN_FRAC = 0.3       # at least 30% of upward rays must see sky
                cam_loc = cam.matrix_world.translation
                sky_dirs = [Vector((0, 0, 1))]  # straight up
                for angle in range(0, 360, 30):  # 12 rays at ~27° from vertical
                    r = np.deg2rad(angle)
                    sky_dirs.append(Vector((0.5 * np.cos(r), 0.5 * np.sin(r), 1)).normalized())
                for angle in range(0, 360, 30):  # 12 rays at ~45° from vertical
                    r = np.deg2rad(angle)
                    sky_dirs.append(Vector((np.cos(r), np.sin(r), 1)).normalized())
                open_rays = 0
                for d in sky_dirs:
                    hit, _, _, dist = scene_bvh.ray_cast(cam_loc, d)
                    if hit is None or dist > MAX_CEILING_DIST:
                        open_rays += 1
                if open_rays / len(sky_dirs) < MIN_OPEN_FRAC:
                    rejection_counts["inside_building"] += 1
                    continue

            # Panoramic enclosure check (city path): the final render is a full
            # 360° equirectangular panorama, so a forward pinhole / upward-only
            # test misses props beside or below the camera. Cast a full sphere of
            # rays against the (instance-aware) BVH and apply the SAME thresholds
            # the post-render depth check used, making that check redundant.
            if panoramic_enclosure_check:
                max_depth, clip_frac, sky_frac, far_frac, near_frac = panoramic_depth_stats(
                    cam.matrix_world.translation, scene_bvh
                )
                if max_depth < 2.0:
                    rejection_counts["pano_enclosed"] += 1
                    continue
                if clip_frac > 0.05:
                    rejection_counts["pano_clip"] += 1
                    continue
                # Inside a building: almost no open sky AND almost no distant view.
                # (Requiring both low keeps valid narrow streets, which see little
                # sky but plenty of far geometry down the street.)
                if sky_frac < 0.12 and far_frac < 0.12:
                    rejection_counts["pano_inside_building"] += 1
                    continue
                # Low-parallax reject (urban multi-view): an open plaza / wide road
                # with buildings > near_dist away gives a small-baseline rig almost
                # no parallax, so it's poor GS supervision. Off by default
                # (min_pano_near_frac=0); the urban path sets it.
                if near_frac < min_pano_near_frac:
                    rejection_counts["pano_low_parallax"] += 1
                    continue

            # Compute focus distance
            destination = cam.matrix_world @ Vector((0.0, 0.0, -1.0))
            forward_dir = (destination - cam.location).normalized()
            *_, straight_ahead_dist = scene_bvh.ray_cast(cam.location, forward_dir)

            potential_views.append((criterion, deepcopy(props), straight_ahead_dist))
            pbar.update(1)

            if len(potential_views) >= n_min_candidates:
                break

    if len(potential_views) < n_views:
        if not allow_fewer:
            if visualize:
                butil.save_blend("compute_base_views-failed.blend")
            raise ValueError(f"Could not find {n_views} camera views")
        # allow_fewer: caller (configure_cameras placing many rigs at once) accepts
        # however many valid poses we found -- return them instead of failing.
        logger.warning(
            f"compute_base_views: found {len(potential_views)}/{n_views} views "
            f"in {max_tries} tries; returning fewer"
        )

    # Shuffle instead of sorting by openness score — we don't want to bias
    # toward wide-open areas, any outdoor camera that passed the sky check
    # is valid even if it's near walls.
    np.random.shuffle(potential_views)

    return potential_views[:n_views]


def build_instance_aware_bvh(exclude_prefix="Culling"):
    """World-space BVHTree that INCLUDES geometry-node / collection instances.

    ``meshes.new_from_object`` (used by :func:`build_bvh_and_attrs`) only bakes an
    object's own realized mesh — it is blind to Instance-on-Points / collection
    instances, so cars, trees, street furniture, façade greebles and windows
    (which iCity scatters as instances) never enter the BVH. Worse, the source
    template assets get baked at their authoring location instead of where they
    render.

    We instead walk the evaluated depsgraph's ``object_instances`` so the tree
    matches exactly what the renderer rasterizes. Objects/instancers whose name
    starts with ``exclude_prefix`` (the enclosing culling volumes) are skipped so
    they don't box in the sky raycast.
    """
    import bmesh as _bmesh
    import time as _time

    # Force a full depsgraph evaluation first. After loading / library-reloading a
    # scene (esp. iCity), geometry-node modifiers may not be evaluated yet, so a
    # bare evaluated_depsgraph_get() can return a PARTIAL scene -> the BVH would
    # miss buildings and enclosed camera poses would wrongly pass the pre-render
    # check (while the render, which fully evaluates, shows them enclosed).
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Cache local (untransformed) triangulated geometry per evaluated mesh so the
    # thousands of instances that share one asset are meshed only once.
    local_cache: dict = {}

    def _local_geom(ob):
        key = ob.data.name
        if key in local_cache:
            return local_cache[key]
        me = bpy.data.meshes.new_from_object(ob)
        geom = None
        if me is not None:
            bm = _bmesh.new()
            bm.from_mesh(me)
            _bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.to_mesh(me)
            bm.free()
            if len(me.vertices):
                verts = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
                # (F,3) int array -- faces are triangles after triangulate(). Keeping
                # faces/verts as numpy (never Python lists of tuples) is what lets a
                # 100M-tri iCity BVH build in ~25 GB instead of ~50 GB: the old
                # per-triangle tuple list + verts.tolist() at FromPolygons were the
                # spike. BVHTree.FromPolygons accepts numpy arrays directly.
                faces = (
                    np.array([p.vertices[:] for p in me.polygons], dtype=np.int32)
                    if len(me.polygons)
                    else np.empty((0, 3), dtype=np.int32)
                )
                geom = (verts, faces)
            bpy.data.meshes.remove(me)
        local_cache[key] = geom
        return geom

    all_verts = []
    all_faces = []
    voff = 0
    n_inst = n_base = n_hidden = 0
    t0 = _time.perf_counter()
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob is None or ob.type != "MESH" or ob.data is None:
            continue
        holder = inst.parent if inst.is_instance else ob
        if holder is not None and holder.name.startswith(exclude_prefix):
            continue
        # Skip geometry hidden from the render (layout placeholders, helper meshes):
        # it lives in the viewport depsgraph but never renders, so raycasts hit it
        # as phantom near-surfaces the rendered depth lacks -- which made the
        # panoramic near/enclosure check over-read content vs the actual panorama.
        if holder is not None and holder.hide_render:
            n_hidden += 1
            continue
        geom = _local_geom(ob)
        if geom is None:
            continue
        lv, lf = geom
        M = np.array(inst.matrix_world, dtype=np.float64)
        all_verts.append(lv @ M[:3, :3].T + M[:3, 3])
        all_faces.append(lf + voff)  # numpy offset, no per-triangle Python tuples
        voff += len(lv)
        if inst.is_instance:
            n_inst += 1
        else:
            n_base += 1

    if not all_verts:
        raise ValueError("build_instance_aware_bvh found no geometry")

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0) if all_faces else np.empty((0, 3), dtype=np.int32)
    logger.info(
        f"build_instance_aware_bvh: {n_base} base + {n_inst} instances "
        f"(skipped {n_hidden} render-hidden) -> "
        f"{len(verts)} verts, {len(faces)} tris "
        f"({_time.perf_counter() - t0:.1f}s)"
    )
    return BVHTree.FromPolygons(verts, faces, all_triangles=True)


def panoramic_depth_stats(origin, scene_bvh, n_theta=128, n_phi=256, sky_dist=1e4,
                          far_dist=30.0, near_dist=8.0, horiz_deg=8.0):
    """Raycast a full equirectangular sphere of directions from ``origin`` against
    ``scene_bvh`` and return ``(max_depth, clip_frac, sky_frac, far_frac, near_frac)``.

    Directions are sampled on a lat/long grid, matching how equirectangular pixels
    tile the sphere (denser toward the poles), so the returned fractions equal the
    fractions the rendered depth map would report. Because the statistics are over
    the whole sphere, they are independent of camera yaw. A missed ray (open sky)
    counts as ``sky_dist * 10`` so it registers as sky and as a large max depth.

    The grid is fairly dense (128x256): a coarse grid can thread rays through the
    small gaps in an iCity building shell and read a fully-enclosed courtyard as
    partly open. ``far_frac`` (fraction seeing beyond ``far_dist`` m) is returned
    alongside ``sky_frac`` because an inside-building view can have ~0 open sky yet
    a sliver of distant street; requiring BOTH to be low is a robust enclosure test.

    ``near_frac`` is the fraction of the HORIZONTAL band (rays within ``horiz_deg``
    of the equator/horizon) that hit geometry within ``near_dist`` m. It measures
    how much nearby, parallax-giving content the panorama has -- a low value means
    an open plaza / wide road where a small-baseline rig sees almost no motion, so
    the urban path rejects such poses. Only the horizontal band is used: the ground
    straight down is always 'near' but gives no useful parallax.
    """
    origin = Vector(origin)
    thetas = np.pi * (np.arange(n_theta) + 0.5) / n_theta
    phis = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    depths = np.empty((n_theta, n_phi), dtype=np.float64)
    miss = sky_dist * 10
    for i, th in enumerate(thetas):
        st, ct = np.sin(th), np.cos(th)
        for j, ph in enumerate(phis):
            _, _, _, dist = scene_bvh.ray_cast(
                origin, Vector((st * np.cos(ph), st * np.sin(ph), ct))
            )
            depths[i, j] = miss if dist is None else dist
    # Horizontal band: rows whose polar angle is within horiz_deg of the equator.
    band = np.abs(np.degrees(thetas) - 90.0) <= horiz_deg
    band_depths = depths[band]
    near_frac = float((band_depths < near_dist).mean()) if band_depths.size else 0.0
    return (
        float(depths.max()),
        float((depths < 0.5).mean()),
        float((depths > sky_dist).mean()),
        float((depths > far_dist).mean()),
        near_frac,
    )


def build_bvh_and_attrs(objs, tags_queries, include_instances=False):
    import bmesh as _bmesh
    import time as _time
    from infinigen.terrain.utils import Mesh

    # City path: build a BVH that matches the rendered geometry (includes GN /
    # collection instances). Tag/range selection queries are not supported here
    # (the urban pipeline passes none), so we return empty selection answers.
    if include_instances:
        if tags_queries:
            logger.warning(
                "build_bvh_and_attrs(include_instances=True) ignores selection "
                f"queries {list(tags_queries)}"
            )
        return build_instance_aware_bvh(), {}


    # Build a single triangulated world-space mesh using bmesh, avoiding all
    # bpy.ops calls that require viewport context (fails for hidden collections).
    combined_bm = _bmesh.new()

    mesh_objs = [o for o in objs if o.type == "MESH" and o.data is not None]
    logger.info(f"build_bvh_and_attrs: processing {len(mesh_objs)} mesh objects")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    t0 = _time.perf_counter()
    for i, obj in enumerate(mesh_objs):
        logger.info(f"  [{i+1}/{len(mesh_objs)}] Evaluating '{obj.name}' "
                    f"(verts={len(obj.data.vertices)}, faces={len(obj.data.polygons)})")
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_data = bpy.data.meshes.new_from_object(obj_eval)
        # Apply world transform so all geometry is in world space
        mesh_data.transform(obj.matrix_world)
        # Triangulate
        bm = _bmesh.new()
        bm.from_mesh(mesh_data)
        _bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
        bm.to_mesh(mesh_data)
        bm.free()
        combined_bm.from_mesh(mesh_data)
        bpy.data.meshes.remove(mesh_data)
    logger.info(f"build_bvh_and_attrs: mesh merging done in {_time.perf_counter()-t0:.1f}s, "
                f"combined verts={len(combined_bm.verts)}, faces={len(combined_bm.faces)}")

    # Create a temporary mesh object for BVH construction and Mesh reading
    temp_mesh = bpy.data.meshes.new("_bvh_temp")
    combined_bm.to_mesh(temp_mesh)
    combined_bm.free()
    temp_obj = bpy.data.objects.new("_bvh_temp", temp_mesh)
    bpy.context.scene.collection.objects.link(temp_obj)

    logger.info("build_bvh_and_attrs: building BVHTree...")
    t1 = _time.perf_counter()
    bvh = BVHTree.FromObject(temp_obj, bpy.context.evaluated_depsgraph_get())
    logger.info(f"build_bvh_and_attrs: BVHTree done in {_time.perf_counter()-t1:.1f}s")
    mesh = Mesh(obj=temp_obj)
    delete(temp_obj)

    camera_selection_answers = {}
    for q0 in tags_queries:
        if type(q0) is not tuple:
            q = (q0,)
        else:
            q = q0
        if q[0] in [SelectionCriterions.CloseUp]:
            continue
        if q[0] == SelectionCriterions.Altitude:
            min_altitude, max_altitude = q[1:3]
            altitude = mesh.vertices[:, 2]
            camera_selection_answers[q0] = mesh.facewise_mean(
                (altitude > min_altitude) & (altitude < max_altitude)
            )
        else:
            camera_selection_answers[q0] = np.zeros(len(mesh.faces), dtype=bool)
            for key in tag_system.tag_dict:
                if set(q).issubset(set(key.split("."))):
                    camera_selection_answers[q0] |= (
                        mesh.face_attributes["MaskTag"] == tag_system.tag_dict[key]
                    ).reshape(-1)
    return bvh, camera_selection_answers


def camera_selection_preprocessing(
    terrain,
    scene_objs,
    tags_ratio: dict = None,
    ranges_ratio: dict = None,
    anim_criterion_keys: dict = None,
    include_instances: bool = False,
):
    if tags_ratio is None:
        tags_ratio = {}
    if ranges_ratio is None:
        ranges_ratio = {}
    if anim_criterion_keys is None:
        anim_criterion_keys = {}

    # preprocessing code adapted from mazeyu's original gin-oriented solution
    tags_ratio = {
        k: (*v, anim_criterion_keys.get(k, False)) for k, v in tags_ratio.items()
    }
    ranges_ratio = {
        v[:-2]: (v[-2], v[-1], anim_criterion_keys.get(k, False))
        for k, v in ranges_ratio.items()
    }

    all_selection_ratios = {**tags_ratio, **ranges_ratio}

    with Timer("Building placeholders KDTree"):
        placeholders = list(
            chain.from_iterable(
                c.all_objects
                for c in bpy.data.collections
                if c.name.startswith("placeholders:")
            )
        )
        placeholders = [p for p in placeholders if p.type == "MESH"]
        logger.info(f"Building placeholder kd for {len(placeholders)} objects")
        placeholders_kd = butil.joined_kd(placeholders, include_origins=True)

    if terrain is None:
        scene_bvh, camera_selection_answers = build_bvh_and_attrs(
            scene_objs, all_selection_ratios.keys(), include_instances=include_instances
        )
        vertexwise_min_dist = None
    else:
        scene_bvh, camera_selection_answers, vertexwise_min_dist = (
            terrain.build_terrain_bvh_and_attrs(all_selection_ratios.keys())
        )

    return dict(
        terrain=terrain,
        scene_bvh=scene_bvh,
        camera_selection_ratio=all_selection_ratios,
        camera_selection_answers=camera_selection_answers,
        vertexwise_min_dist=vertexwise_min_dist,
        placeholders_kd=placeholders_kd,
    )


@node_utils.to_nodegroup("geo_distrib", singleton=True, type="GeometryNodeTree")
def geo_distrib_random_points(nw: NodeWrangler):
    input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )
    distribute = nw.new_node(
        Nodes.DistributePointsOnFaces,
        input_kwargs={"Mesh": input.outputs["Geometry"], "Density": 500},
    )
    verts = nw.new_node(Nodes.PointsToVertices, [distribute])
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": verts})


def sample_random_locs(surface: bpy.types.Object, eps=0.01):
    # HACK implementation - uses blender geonodes' uniform surface sample, im fairly sure theres a numpy impl somewhere in the repo
    surface = butil.copy(surface)
    butil.apply_transform(surface, loc=True, rot=True, scale=True)
    butil.modify_mesh(
        surface, "NODES", node_group=geo_distrib_random_points(), apply=True
    )
    locs = np.array([v.co for v in surface.data.vertices])
    locs[:, -1] += eps
    butil.delete(surface)
    return locs


@gin.configurable
def configure_cameras(
    cam_rigs,
    scene_preprocessed: dict,
    init_bounding_box: tuple[np.array, np.array] = None,
    init_surfaces: list[bpy.types.Object] = None,
    terrain_mesh=None,
    nonroom_objs=None,
    mvs_setting=False,
    mvs_radius=("uniform", 12, 18),
    allow_fewer_rigs=False,
    **kwargs,
):
    bpy.context.view_layer.update()

    if init_bounding_box is not None:

        def location_sample():
            return np.random.uniform(*init_bounding_box)
    elif init_surfaces is not None:
        random_locs = sample_random_locs(init_surfaces)

        def location_sample():
            loc = Vector(random_locs[np.random.randint(len(random_locs)), :])
            loc.z += 1e-3
            return loc
    else:
        raise ValueError("Either init_bounding_box or init_surfaces must be provided")

    if mvs_setting:
        if terrain_mesh:
            vertices = np.array([np.array(v.co) for v in terrain_mesh.data.vertices])
            sdfs = scene_preprocessed["terrain"].compute_camera_space_sdf(vertices)
            vertices = vertices[sdfs >= -1e-5]
            center_coordinate = list(
                vertices[np.random.choice(list(range(len(vertices))))]
            )
        elif nonroom_objs:

            def contain_keywords(name, keywords):
                for keyword in keywords:
                    if name == keyword or name.startswith(f"{keyword}."):
                        return True
                return False

            inside_objs = [
                x
                for x in nonroom_objs
                if not contain_keywords(x.name, ["window", "door", "entrance"])
            ]
            assert inside_objs != []
            obj = np.random.choice(inside_objs)
            vertices = [v.co for v in obj.data.vertices]
            center_coordinate = vertices[np.random.choice(list(range(len(vertices))))]
            center_coordinate = obj.matrix_world @ center_coordinate
            center_coordinate = list(np.array(center_coordinate))
    else:
        center_coordinate = None

    print("Cam rigs: ", len(cam_rigs))
    # One compute_base_views search per rig -- the simple, proven approach the
    # monocular pipeline used to place 300-500 cameras/city. With allow_fewer_rigs a
    # rig whose search exhausts its retries is skipped (kept rigs still render)
    # instead of failing the whole scene.
    placed = []
    for i, cam_rig in enumerate(cam_rigs):
        try:
            views = compute_base_views(
                cam_rig,
                n_views=1,
                location_sample=location_sample,
                center_coordinate=center_coordinate,
                radius=mvs_radius,
                bbox=init_bounding_box,
                **scene_preprocessed,
                **kwargs,
            )
        except ValueError:
            if not allow_fewer_rigs:
                raise
            logger.warning(f"configure_cameras: rig {i} unplaceable; skipping it")
            continue

        score, props, focus_dist = views[0]
        cam_rig.location = props.loc
        cam_rig.rotation_euler = props.rot
        for cam in cam_rig.children:
            cam.data.lens = props.focal_length
        if focus_dist is not None:
            for cam in cam_rig.children:
                if cam.type == "CAMERA":
                    cam.data.dof.focus_distance = focus_dist
        placed.append(cam_rig)

    if allow_fewer_rigs:
        unplaced = [r for r in cam_rigs if r not in placed]
        if unplaced:
            logger.warning(
                f"configure_cameras: placed {len(placed)}/{len(cam_rigs)} rigs; "
                f"removing {len(unplaced)} unplaceable rig(s) so they don't render"
            )
            butil.delete([o for r in unplaced for o in (list(r.children) + [r])])
        if isinstance(cam_rigs, list):
            cam_rigs[:] = placed  # trim the caller's list to the rigs that rendered
        if not placed:
            raise ValueError("configure_cameras: could not place ANY camera rig")

    return placed


@gin.configurable
def animate_cameras(
    cam_rigs,
    bounding_box,
    scene_preprocessed,
    pois=None,
    follow_poi_chance=0.0,
    policy_registry=None,
    **kwargs,
):
    animation_ratio = {}
    animation_answers = {}
    for k in scene_preprocessed["camera_selection_ratio"]:
        if scene_preprocessed["camera_selection_ratio"][k][2]:
            animation_ratio[k] = scene_preprocessed["camera_selection_ratio"][k]
            animation_answers[k] = scene_preprocessed["camera_selection_answers"][k]

    def anim_valid_camrig_pose_func(cam_rig: bpy.types.Object):
        assert len(cam_rig.children) > 0

        scores = []

        for cam in cam_rig.children:
            score = keep_cam_pose_proposal(
                cam,
                placeholders_kd=scene_preprocessed["placeholders_kd"],
                scene_bvh=scene_preprocessed["scene_bvh"],
                terrain=scene_preprocessed["terrain"],
                vertexwise_min_dist=scene_preprocessed["vertexwise_min_dist"],
                camera_selection_answers=animation_answers,
                camera_selection_ratio=animation_ratio,
                **kwargs,
            )

            frame = bpy.context.scene.frame_current
            logger.debug(f"Checking {cam.name=} {frame=} got {score=}")

            if score is None:
                return None

            scores.append(score)

        return np.min(scores)

    for cam_rig in cam_rigs:
        if policy_registry is None:
            if U() < follow_poi_chance and pois is not None and len(pois):
                policy = animation_policy.AnimPolicyFollowObject(
                    target_obj=cam_rig, pois=pois, bvh=scene_preprocessed["scene_bvh"]
                )
            else:
                policy = animation_policy.AnimPolicyRandomWalkLookaround()
        else:
            policy = policy_registry()

        logger.info(f"Animating {cam_rig=} using {policy=}")

        animation_policy.animate_trajectory(
            cam_rig,
            scene_preprocessed["scene_bvh"],
            policy_func=policy,
            validate_pose_func=anim_valid_camrig_pose_func,
            verbose=True,
            fatal=True,
            bounding_box=bounding_box,
        )


@gin.configurable
def save_camera_parameters(
    camera_obj: bpy.types.Object, output_folder: Path, frame: int, use_dof=False
):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if frame is not None:
        bpy.context.scene.frame_set(frame)

    camrig_id, subcam_id = get_id(camera_obj)

    if use_dof is not None:
        camera_obj.data.dof.use_dof = use_dof

    adjust_camera_sensor(camera_obj)

    # Saving camera parameters
    K = camera.get_calibration_matrix_K_from_blender(camera_obj)
    suffix = get_suffix(
        dict(cam_rig=camrig_id, resample=0, frame=frame, subcam=subcam_id)
    )
    output_file = output_folder / f"camview{suffix}.npz"

    height_width = np.array(
        (
            bpy.context.scene.render.resolution_y,
            bpy.context.scene.render.resolution_x,
        )
    )
    T = np.asarray(camera_obj.matrix_world, dtype=np.float64)
    # T = np.asarray(camera_obj.matrix_world, dtype=np.float64) @ np.diag(
    #     (1.0, -1.0, -1.0, 1.0)
    #)  # Y down Z forward (aka opencv)
    np.savez(output_file, K=np.asarray(K, dtype=np.float64), T=T, HW=height_width)


if __name__ == "__main__":
    """
    This interactive section generates a depth map by raycasting through each pixel. 
    It is very useful for debugging camera.py.
    """
    cam = bpy.context.scene.camera

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    adjust_camera_sensor(cam)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvhtree = BVHTree.FromObject(bpy.context.active_object, depsgraph)

    target_obj = bpy.context.active_object
    to_obj_coords = target_obj.matrix_world.inverted()
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=False)

    H, W = sensor_coords.shape
    depth_output = np.zeros((H, W), dtype=np.float64)

    for x, y in tqdm(pix_it):
        destination = sensor_coords[y, x]
        direction = (destination - cam.location).normalized()
        location, normal, index, dist = bvhtree.ray_cast(cam.location, direction)
        if dist is not None:
            dist_diff = (destination - cam.location).length
            assert dist > (location - destination).length, (
                dist,
                (location - destination).length,
            )
            assert dist > dist_diff
            depth_output[H - y - 1, x] = dist - dist_diff

    color_depth = colorize_depth(depth_output)
    imageio.imwrite("color_depth.png", color_depth)
