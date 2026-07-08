import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

# Use the environment variable for OpenCV EXR support BEFORE cv2 import
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _phase(name: str):
    logger.info(f"=== PHASE: {name} ===")
    return time.perf_counter()

def _phase_done(name: str, t0: float):
    logger.info(f"=== DONE: {name} ({time.perf_counter() - t0:.1f}s) ===")


@contextlib.contextmanager
def suppress_blender_output():
    """Redirect both stdout AND stderr at the fd level to silence Blender C-level noise.

    Blender's internal logging can go to either fd depending on the build.
    Python's print() is rerouted through a saved copy of the original stdout
    so our own messages still appear.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    # Redirect both fds to /dev/null (catches all C-level output)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    # Make Python's print() write to the saved original stdout
    py_out = os.fdopen(saved_stdout_fd, 'w', closefd=False)
    old_sys_stdout = sys.stdout
    sys.stdout = py_out

    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = old_sys_stdout
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

import bpy
import numpy as np
import gin
from mathutils import Vector
import imageio

# Add root to sys.path so we can import infinigen
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from infinigen.core import init
from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering import render
from infinigen.core.util import blender as butil

# Specific post-processing tools requested by user
from infinigen.core.rendering.post_render import load_normals, reorient_surface_normals_from_camview, colorize_normals, load_depth, colorize_depth, colorize_depth_viz, load_exr
from infinigen.tools.suffixes import get_suffix

def _register_icity_addon():
    """Register the iCity addon so its PropertyGroups are available before opening a city .blend file."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import iCity as _icity
    _icity.register()
    print("iCity addon registered.")
    return script_dir


def _remap_icity_libraries(script_dir: Path):
    """After opening a city .blend, fix any library paths that point to the
    original machine, then reload the linked data so it actually populates
    (without remapping + reload, only the first/main object tree appears)."""
    theme_dir = script_dir / "iCity - Default Theme"
    remapped_libs = []
    for lib in bpy.data.libraries:
        lib_abs = Path(bpy.path.abspath(lib.filepath))
        if not lib_abs.exists():
            matches = list(theme_dir.rglob(lib_abs.name))
            if matches:
                lib.filepath = str(matches[0])
                print(f"Remapped library: {lib_abs.name} -> {matches[0]}")
                remapped_libs.append(lib)
            else:
                print(f"Warning: Could not find library asset: {lib_abs.name}")

    if not remapped_libs:
        return

    # Disable Blender's library override auto-resync. On iCity .blend files
    # this resync runs for minutes and emits thousands of dependency-loop
    # warnings; we don't need overrides to be re-synced for headless rendering.
    try:
        prefs = bpy.context.preferences
        if hasattr(prefs, "experimental") and hasattr(prefs.experimental, "no_override_auto_resync"):
            prefs.experimental.no_override_auto_resync = True
            print("Disabled override auto-resync for fast library reload.")
    except Exception as e:
        print(f"Note: could not disable override auto-resync ({e}); reload may be slow.")

    # Reload each remapped library so its actual data (objects, meshes,
    # materials, ...) gets pulled into the scene. Without this, only the
    # main file's local data is visible — linked collections appear empty.
    for lib in remapped_libs:
        try:
            lib.reload()
            print(f"Reloaded library: {Path(lib.filepath).name}")
        except Exception as e:
            print(f"Warning: Could not reload {lib.filepath}: {e}")


def _force_lights_off():
    """Force daytime mode on all iCity objects.

    bpy 4.5 drops the saved boolean values from iCity's GeometryNodes modifier
    sockets when loading 4.4-saved files. The geometry node defaults leave
    building lights ON. We explicitly set both 'Light Mode' and 'Lights On'
    inputs to False on every relevant object — same mechanism iCity's own
    set_mode() uses internally.
    """
    # The file was saved with Blender 5.1 but we run bpy 4.5. The socket
    # type system changed between versions, so setting modifier properties
    # via mod[identifier] is silently ignored ("Property type does not match
    # input socket"). Instead we patch the node group's own default_value
    # for each socket — that IS what Blender falls back to when the modifier
    # property doesn't match.
    sockets_to_force_false = ["Light Mode", "Lights On"]
    patched = set()
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type != 'NODES' or mod.node_group is None:
                continue
            ng = mod.node_group
            if ng.name in patched:
                continue
            for socket_name in sockets_to_force_false:
                try:
                    item = ng.interface.items_tree[socket_name]
                    item.default_value = False
                    print(f"Patched node group '{ng.name}' socket '{socket_name}' default = False")
                    patched.add(ng.name)
                except (KeyError, AttributeError):
                    pass

    # 3. Disable all emission in materials (belt and suspenders)
    n_emission_zeroed = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'EMISSION':
                if 'Strength' in node.inputs:
                    node.inputs['Strength'].default_value = 0.0
                    n_emission_zeroed += 1
            elif node.type == 'BSDF_PRINCIPLED':
                if 'Emission Strength' in node.inputs:
                    node.inputs['Emission Strength'].default_value = 0.0
                    n_emission_zeroed += 1
    print(f"Zeroed emission on {n_emission_zeroed} shader nodes")

    # 4. Hide / zero all actual light objects
    n_lights = 0
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
            obj.hide_viewport = True
            try:
                obj.data.energy = 0.0
            except Exception:
                pass
            n_lights += 1
    print(f"Disabled {n_lights} light objects")

    # 5. Flip iCity's own property toggles for good measure
    main_props = getattr(bpy.context.scene, "parametra_icity_main", None)
    if main_props is not None:
        for prop in ("night_mode", "light_mode"):
            if hasattr(main_props, prop):
                try:
                    setattr(main_props, prop, False)
                except Exception:
                    pass


def _force_wetness_off():
    """Force dry city (no rain puddles).

    Wetness in iCity is controlled at many levels: a 'Rain density' int socket
    on most GeometryNodes groups (Road, Sidewalk, Towers, Twisted Towers,
    Podium Roof, Face to podium, Parks system.001, ...), a 'Wetness' float on
    Sidewalk materials / Down Town Buildings, 'dry_humid_wet' on front yard /
    ground materials, plus 'Wet'/'Wet Density' bools. Because the version
    mismatch between the 5.1-saved .blend and our bpy causes modifier[gid]
    assignment to be silently dropped, we patch every matching socket's
    interface default_value directly.

    We sweep ALL node groups (not just the ones attached to the iCity
    top-level objects) because linked asset groups like 'Twisted Towers.*'
    and 'Podium Roof.*' also have Rain density sockets that need zeroing.
    """
    # (socket_name_lower_pattern, "dry" value)
    # Patterns are matched case-insensitively against the socket name.
    wet_patterns = [
        ("rain density", 0),
        ("wetness", 0.0),
        ("dry_humid_wet", 0),
        ("roof material wet density", 0.0),
        ("roof material wet", False),
    ]

    n_patched = 0
    for ng in bpy.data.node_groups:
        if not hasattr(ng, "interface"):
            continue
        try:
            items = ng.interface.items_tree
        except Exception:
            continue
        for item in items:
            name = getattr(item, "name", None)
            if not name:
                continue
            name_l = name.strip().lower()
            # Match 'Wet'/'Wet ' sockets separately to avoid collisions with
            # e.g. 'Wetness' or 'dry_humid_wet' (handled in the pattern list).
            if name_l == "wet":
                try:
                    item.default_value = False
                    n_patched += 1
                except (AttributeError, TypeError):
                    pass
                continue
            for pattern, dry_value in wet_patterns:
                if pattern in name_l:
                    try:
                        item.default_value = dry_value
                        n_patched += 1
                    except (AttributeError, TypeError):
                        pass
                    break
    print(f"Patched {n_patched} wetness-related sockets")

    # Modifier-level values stored in the saved .blend override interface
    # defaults at evaluation time. Sweep every GeometryNodes modifier on
    # every object and force any wetness socket's stored value to the dry
    # equivalent (same matching rules as above).
    def _dry_value_for(socket_name_lower, socket):
        if socket_name_lower == "wet":
            return False
        for pattern, dry_value in wet_patterns:
            if pattern in socket_name_lower:
                return dry_value
        return None

    n_mod_patched = 0
    for obj in bpy.data.objects:
        if not hasattr(obj, "modifiers"):
            continue
        for mod in obj.modifiers:
            if mod.type != 'NODES' or mod.node_group is None:
                continue
            ng = mod.node_group
            try:
                items = ng.interface.items_tree
            except Exception:
                continue
            for item in items:
                name = getattr(item, "name", None)
                if not name:
                    continue
                name_l = name.strip().lower()
                dry = _dry_value_for(name_l, item)
                if dry is None:
                    continue
                try:
                    mod[item.identifier] = dry
                    n_mod_patched += 1
                except (KeyError, AttributeError, TypeError):
                    pass
    print(f"Overrode {n_mod_patched} modifier-level wetness values")

    # Globally remap every '* wet*' material to its dry equivalent. Wet
    # materials are referenced via GN Index Switch nodes (inside the 'wet'
    # sub-group of Road v3 and similar), NOT via object material slots.
    # user_remap walks all references in bpy.data — slots, shader nodes,
    # GN inputs — and rewrites them in one call.
    n_remapped = 0
    n_deleted = 0
    wet_mats = [m for m in bpy.data.materials if " wet" in m.name.lower() or "_wet" in m.name.lower()]
    for wet_mat in wet_mats:
        # "ICity_Road_Ashphalt_high wet.002" -> "ICity_Road_Ashphalt_high.002"
        # "Crack decal_wet.000"               -> "Crack decal.000"
        dry_name = wet_mat.name
        for token in (" wet", " Wet", "_wet", "_Wet"):
            dry_name = dry_name.replace(token, "")
        dry_mat = bpy.data.materials.get(dry_name)
        if dry_mat is not None and dry_mat is not wet_mat:
            wet_mat.user_remap(dry_mat)
            n_remapped += 1
        else:
            # No dry equivalent: make the wet material render as
            # transparent/invisible so any leftover reference contributes nothing.
            if wet_mat.use_nodes and wet_mat.node_tree:
                for node in list(wet_mat.node_tree.nodes):
                    if node.type == 'BSDF_PRINCIPLED':
                        try:
                            node.inputs['Alpha'].default_value = 0.0
                        except (KeyError, AttributeError):
                            pass
                        try:
                            node.inputs['Roughness'].default_value = 1.0
                        except (KeyError, AttributeError):
                            pass
                        try:
                            node.inputs['Metallic'].default_value = 0.0
                        except (KeyError, AttributeError):
                            pass
                n_deleted += 1
    print(f"Remapped {n_remapped} wet materials to dry; neutralized {n_deleted} without dry equivalent")

    # Flip the iCity property toggle too
    main_props = getattr(bpy.context.scene, "parametra_icity_main", None)
    if main_props is not None and hasattr(main_props, "road_moisture"):
        try:
            main_props.road_moisture = "0%"
        except Exception:
            pass


def _parse_spec(s):
    """Parse '0.5' -> 0.5 or 'uniform,0.3,0.7' -> ('uniform', 0.3, 0.7)."""
    parts = str(s).split(",")
    if len(parts) == 1:
        return float(parts[0])
    return (parts[0], *[float(p) for p in parts[1:]])


def main(args):
    # 1. Resolve city directory and discover files
    city_dir = args.city_dir.resolve()
    if not city_dir.is_dir():
        raise FileNotFoundError(f"City directory {city_dir} does not exist")

    # Find the .blend file
    blend_files = list(city_dir.glob("*.blend"))
    if not blend_files:
        raise FileNotFoundError(f"No .blend file found in {city_dir}")
    input_blend = blend_files[0]
    print(f"Using blend file: {input_blend}")

    # Output goes to outputs/urban/<city_dir_name>/
    output_folder = Path("outputs/urban") / city_dir.name
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Register iCity addon so its PropertyGroups exist when Blender deserializes the scene
    script_dir = _register_icity_addon()

    t0 = _phase("load blend file")
    bpy.ops.wm.open_mainfile(filepath=str(input_blend), load_ui=False)
    _remap_icity_libraries(script_dir)
    _phase_done("load blend file", t0)

    # Ensure any stuck material override is cleared immediately
    if "ViewLayer" in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers["ViewLayer"].material_override = None

    # Ensure we use Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # HDRI environment lighting is already embedded in the .blend file;
    # no need to override it from an external file.

    # Force iCity night-mode lights off (bpy 4.5 drops the saved False values)
    _force_lights_off()
    # Force dry city (rain puddles get re-enabled on version-mismatched load)
    _force_wetness_off()

    # 2. Apply configs
    # This sets up rendering settings, camera parameters etc on the loaded scene
    t0 = _phase("apply gin configs")
    init.apply_gin_configs(
        config_folders=[
            Path("infinigen/datagen/configs"),
            Path("infinigen_examples/configs_nature"),
            Path("infinigen_examples/configs_indoor")
        ],
        configs=args.configs,
        overrides=args.overrides,
        skip_unknown=True
    )
    _phase_done("apply gin configs", t0)

    # Explicitly set camera rotation to (90, 0, random_yaw)
    # This overrides any settings from the config files
    print("Overriding camera rotation to (90, 0, random_yaw)...")
    # pitch=90 creates a horizontal view (looking out at the horizon)
    gin.bind_parameter("infinigen.core.placement.camera.camera_pose_proposal.pitch", 90)
    # roll=0 ensures the camera is level
    gin.bind_parameter("infinigen.core.placement.camera.camera_pose_proposal.roll", 0)
    # yaw is randomized 360 degrees
    gin.bind_parameter("infinigen.core.placement.camera.camera_pose_proposal.yaw", ("uniform", -180, 180))
    # Altitude: 1-3m above ground
    gin.bind_parameter("infinigen.core.placement.camera.camera_pose_proposal.altitude", ("uniform", 2, 4))
    # Multi-view: require near, parallax-giving content in the horizontal band so a
    # small-baseline rig isn't placed in an open plaza / wide road where it sees no
    # motion. Off for single panoramas (baseline is moot there).
    if getattr(args, "n_views", 1) > 1 and args.min_near_frac > 0:
        gin.bind_parameter(
            "infinigen.core.placement.camera.compute_base_views.min_pano_near_frac",
            args.min_near_frac,
        )
        print(f"Multi-view: requiring near_frac >= {args.min_near_frac} at each anchor")

    # 3. Find floor level (min Z)
    # Only consider objects that are visible in render — hidden objects, culling
    # volumes, and system helpers can have huge bounding boxes that inflate the
    # scene bounds far beyond the actual city geometry.
    # Use only known iCity city objects for bounds — random leftover objects,
    # ground planes, and debug geometry inflate the bounds far beyond the city.
    _CITY_OBJECTS = {
        "Road", "Sidewalk", "Grid System", "Towers", "Parking",
        "Parks", "Green Area", "Terraced frontend new", "Terraced backend new",
    }
    min_z = float('inf')
    mesh_objs = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and o.name in _CITY_OBJECTS
    ]

    # Calculate scene bounds to sample from
    all_points = []

    for obj in mesh_objs:
        # Check world coordinates
        mw = obj.matrix_world
        # Bounding box corners in world space
        if obj.bound_box:
            bbox_corners = [mw @ Vector(corner) for corner in obj.bound_box]
            for corner in bbox_corners:
                if corner.z < min_z:
                    min_z = corner.z
                all_points.append(corner)
            
    if not all_points:
        # Fallback if empty scene
        print("Warning: No mesh objects found. Using default bounds.")
        min_z = 0
        scene_bounds = (Vector((-10, -10, 0)), Vector((10, 10, 10)))
    else:
        min_coords = np.min([v[:] for v in all_points], axis=0)
        max_coords = np.max([v[:] for v in all_points], axis=0)
        # Take the central 20-80% in X and Y to sample from the city core
        x_range = max_coords[0] - min_coords[0]
        y_range = max_coords[1] - min_coords[1]
        min_coords[0] += 0.2 * x_range
        max_coords[0] -= 0.2 * x_range
        min_coords[1] += 0.2 * y_range
        max_coords[1] -= 0.2 * y_range
        # Clamp Z to near ground level so altitude adjustment (1-3m) works correctly
        min_coords[2] = 2.0
        max_coords[2] = 4.0
        scene_bounds = (Vector(min_coords), Vector(max_coords))

    print(f"Scene bounds for sampling: {scene_bounds}")

    # DEBUG: find which objects define each extreme of the bounding box
    axis_names = ['X', 'Y', 'Z']
    for axis in range(3):
        for label, fn in [('min', min), ('max', max)]:
            worst_obj = None
            worst_val = float('inf') if label == 'min' else float('-inf')
            for obj in mesh_objs:
                if not obj.bound_box:
                    continue
                mw = obj.matrix_world
                corners = [mw @ Vector(c) for c in obj.bound_box]
                vals = [c[axis] for c in corners]
                v = fn(vals)
                if (label == 'min' and v < worst_val) or (label == 'max' and v > worst_val):
                    worst_val = v
                    worst_obj = obj.name
            print(f"  {axis_names[axis]} {label} = {worst_val:.1f} from '{worst_obj}'")

    # 4. Spawn Camera Rig(s)
    # Multi-view (--n-views > 1): each rig is a pitch-compensated constellation of
    # N sub-cameras (anchor + neighbours), so one placed rig yields N pose-
    # registered panoramas of the same content. n_views=1 keeps single-camera rigs.
    n_views = getattr(args, "n_views", 1)
    if n_views > 1:
        baseline = _parse_spec(args.baseline)
        rig_cfg = cam_util.multiview_rig_config(
            n_views=n_views,
            baseline=baseline,
            pattern="ring",
            z_amplitude=args.z_amplitude,
            level_pitch_deg=90,  # matches camera_pose_proposal.pitch bound above
        )
        print(f"Multi-view: {n_views} sub-cameras/rig, baseline={baseline}")
        camera_rigs = cam_util.spawn_camera_rigs(camera_rig_config=rig_cfg)
    else:
        camera_rigs = cam_util.spawn_camera_rigs()
    if not camera_rigs:
        raise RuntimeError("Failed to spawn camera rigs")

    # 5. Automatic Camera Placement
    # Use low resolution for the search phase (much faster raycasting)
    search_res = (256, 256)
    bpy.context.scene.render.resolution_x = search_res[0]
    bpy.context.scene.render.resolution_y = search_res[1]
    print(f"Using {search_res} for fast camera search...")

    # Only exclude enclosing culling volumes from the BVH — they form a box
    # around the city that breaks the sky raycast. Everything else (Terrain,
    # Islands, Plane, etc.) has real geometry that cameras can be inside.
    scene_objs = [
        o for o in bpy.data.objects
        if o.type == 'MESH'
        and not o.name.startswith("Culling")
    ]
    scene_preprocessed = cam_util.camera_selection_preprocessing(
        terrain=None,
        scene_objs=scene_objs,
        tags_ratio={},
        ranges_ratio={},
        # Build the collision BVH from the evaluated depsgraph so it includes GN /
        # collection instances (cars, trees, poles, façade greebles, windows) that
        # meshes.new_from_object() cannot see. This is what makes the pre-render
        # camera-validity tests match what the panorama actually renders.
        include_instances=True,
    )

    print("Searching for optimal camera views...")
    cam_util.configure_cameras(
        camera_rigs,
        scene_preprocessed=scene_preprocessed,
        init_bounding_box=scene_bounds,
        terrain_coverage_range=None,
        min_terrain_distance=2.0,
        # Reject poses that are enclosed / clipping through props using a full
        # 360° sphere raycast against the instance-aware BVH — same thresholds as
        # the old post-render depth check, but before we spend a render on them.
        panoramic_enclosure_check=True,
    )

    # Free the placement collision BVH (the instance-aware BVHTree over ~100M tris
    # is ~28 GB) before rendering -- the render doesn't need it, and holding it while
    # Cycles builds its own render BVH is what tips large cities over the memory
    # limit. Cameras are already placed on the rigs, so this is safe.
    import gc as _gc
    del scene_preprocessed
    _gc.collect()

    # Print found camera positions
    for i, rig in enumerate(camera_rigs):
        loc = rig.location
        rot = rig.rotation_euler
        print(f"Camera rig {i}: loc=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}), "
              f"rot_deg=({np.degrees(rot.x):.1f}, {np.degrees(rot.y):.1f}, {np.degrees(rot.z):.1f})")

    # Setup Resolution & Clipping
    render_res = tuple(args.resolution)
    clip_start = 0.001


    # 6. Render
    # We render into a 'frames' subdirectory to compatible with Infinigen's folder structure logic
    # which expects to reorganize files from 'frames_folder' into 'frames_folder/../frames' or 'frames_folder/Type/...'
    # By using a 'frames' subfolder, we ensure consistent behavior.
    frames_folder = output_folder / "frames"
    # Start from a clean frames dir. render_image -> reorganize_old_framesfolder
    # calls parse_suffix() on EVERY file directly in frames/, so a stale file from
    # a previous run (e.g. transforms.json) has no parseable suffix and crashes it
    # with "'NoneType' object is not subscriptable". A fresh render must own the dir.
    import shutil as _shutil
    if frames_folder.exists():
        _shutil.rmtree(frames_folder)
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Force single frame render
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    
    # 7. Set Color Management to Standard (Avoiding Filmic desaturation of data passes)
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.gamma = 1.0
    bpy.context.scene.view_settings.exposure = 0.0

    # Setup Resolution & Clipping
    render_res = tuple(args.resolution)
    clip_start = 0.001

    t0 = _phase("render")
    logger.info(f"Rendering to {frames_folder}...")

    # Ensure no material override is active
    if "ViewLayer" in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers["ViewLayer"].material_override = None

    # Restore full resolution for rendering
    bpy.context.scene.render.resolution_x = render_res[0]
    bpy.context.scene.render.resolution_y = render_res[1]
    print(f"Restored resolution to {render_res} for rendering...")

    # Bound render-time subdivision. iCity materials use REAL displacement, and the
    # blend ships with Cycles adaptive dicing at dicing_rate=1.0 / max_subdivisions
    # up to 12. In a 360 panorama every surface is "up close", so Cycles dices large
    # displaced facades to ~1px = billions of micro-triangles at RENDER time (unseen
    # in the ~99M viewport tri count) -> some cities OOM at >288 GB. Coarsen the
    # dicing, cap the subdivision depth, and drop real displacement to bump (shading
    # only) so no extra geometry is generated. Depth/normals come from the geometry
    # passes and stay consistent with the (now un-displaced) RGB. Keeps memory near
    # the ~40 GB the base geometry needs.
    sc = bpy.context.scene
    sc.cycles.dicing_rate = 8.0
    if hasattr(sc.cycles, "max_subdivisions"):
        sc.cycles.max_subdivisions = 2
    _disp_fixed = 0
    for _m in bpy.data.materials:
        if getattr(_m, "displacement_method", "BUMP") in ("DISPLACEMENT", "BOTH"):
            _m.displacement_method = "BUMP"
            _disp_fixed += 1
    print(f"Bounded render subdivision (dicing_rate=8, max_subdivisions=2); "
          f"displacement->bump on {_disp_fixed} material(s)")

    for rig_idx, cam_rig in enumerate(camera_rigs):
        for cam in cam_rig.children:
            cam.data.clip_start = clip_start
            print(f"\n=== Rendering camera rig {rig_idx}/{len(camera_rigs)} ({cam.name}) ===")

            render.render_image(
                frames_folder=frames_folder,
                camera=cam,
                passes_to_save=[('z', 'Depth'), ('normal', 'Normal')],
                render_resolution_override=render_res
            )

            # --- Post-process Normals to be in Camera Coordinate Frame ---
            # Reconstruct the filename suffix used by render.render_image
            # Suffix logic: get_suffix({cam_rig, resample, frame, subcam})
            camrig_id, subcam_id = cam_util.get_id(cam)
            # Use same indices dict as render.py uses
            suffix_indices = {
                "cam_rig": camrig_id,
                "resample": 0,
                "frame": bpy.context.scene.frame_start,
                "subcam": subcam_id,
            }
            suffix = get_suffix(suffix_indices)
            
            # Due to reorganize_old_framesfolder, the file is moved to:
            # frames_folder / ChannelName / camera_{subcam_id} / Filename
            
            # ------------------------------------------------------------------
            # NORMALS
            # ------------------------------------------------------------------
            normal_filename = f"Normal{suffix}.exr"
            normal_path = frames_folder / "Normal" / f"camera_{subcam_id}" / normal_filename
            
            if normal_path.exists():
                print(f"Post-processing normals: {normal_path}")
                try:
                    # Load World Space Normals (as saved by Blender)
                    normals_world = load_normals(str(normal_path))
                    
                    # Reorient to Camera Space using camera matrix
                    camview_T = np.array(cam.matrix_world)
                    normals_cam = reorient_surface_normals_from_camview(normals_world, camview_T)

                    # Save into a SurfaceNormal/ channel dir (matches the datagen
                    # layout the multi-view manifest expects), not the Normal/ dir
                    # that only held the raw world-space EXR.
                    sn_dir = frames_folder / "SurfaceNormal" / f"camera_{subcam_id}"
                    sn_dir.mkdir(parents=True, exist_ok=True)
                    np.save(sn_dir / f"SurfaceNormal{suffix}.npy", normals_cam.astype(np.float16))

                    # Save Visualization PNG
                    colored = colorize_normals(normals_cam)
                    imageio.imwrite(sn_dir / f"SurfaceNormal{suffix}.png", colored)
                    
                    # Clean up: Delete original EXR
                    normal_path.unlink()
                     
                    # Clean up: Delete original World Space Normal PNG
                    normal_png = normal_path.with_suffix(".png")
                    if normal_png.exists():
                        normal_png.unlink()
                        print(f"Processed normals and deleted {normal_path.name} and {normal_png.name}")
                    else:
                        print(f"Processed normals and deleted {normal_path.name}")
                except Exception as e:
                    print(f"Error processing normals {normal_path}: {e}")

            # ------------------------------------------------------------------
            # DEPTH
            # ------------------------------------------------------------------
            depth_filename = f"Depth{suffix}.exr"
            depth_path = frames_folder / "Depth" / f"camera_{subcam_id}" / depth_filename
            
            if depth_path.exists():
                print(f"Post-processing depth: {depth_path}")
                try:
                    depth_raw = load_depth(str(depth_path))

                    # Clamp max depth to 125m (sky and far structures) for saving
                    MAX_DEPTH = 500.0
                    depth_clamped = np.clip(depth_raw, None, MAX_DEPTH)

                    # Save clamped .npy (float32)
                    np.save(depth_path.with_name(f"Depth{suffix}.npy"), depth_clamped.astype(np.float32))

                    # Save log-depth visualization PNG with spectral colormap
                    import matplotlib.cm as cm
                    # near=red / far=blue, log-scaled, no white near-field artifacts.
                    depth_colored = colorize_depth_viz(depth_clamped)
                    imageio.imwrite(depth_path.with_name(f"Depth{suffix}.png"), depth_colored)

                    # Clean up: Delete original EXR
                    depth_path.unlink()

                    # Post-render enclosure check (SAFETY NET). The pre-render
                    # panoramic check uses the instance-aware BVH, which can miss
                    # some iCity building geometry, so a camera can still end up
                    # inside a building (dark, no open sky). Re-validate on the
                    # ACTUAL rendered depth (ground truth) and drop such cameras:
                    #   - max_depth < 2m            -> fully enclosed
                    #   - >5% of pixels < 0.5m      -> clipping through a prop
                    #   - <12% sky AND <12% far     -> inside a building / courtyard
                    max_depth = float(np.max(depth_raw))
                    clip_frac = float(np.mean(depth_raw < 0.5))
                    sky_frac = float(np.mean(depth_raw > 1e4))
                    far_frac = float(np.mean(depth_raw > 30.0))
                    reject_reason = None
                    if max_depth < 2.0:
                        reject_reason = f"max_depth={max_depth:.1f}m (enclosed)"
                    elif clip_frac > 0.05:
                        reject_reason = f"clip_frac={clip_frac:.2%} (inside prop)"
                    elif sky_frac < 0.12 and far_frac < 0.12:
                        reject_reason = (
                            f"sky_frac={sky_frac:.2%} far_frac={far_frac:.2%} "
                            "(inside building)"
                        )
                    if reject_reason:
                        print(f"Rejecting camera {suffix}: {reject_reason}")
                        for channel_dir in frames_folder.iterdir():
                            if not channel_dir.is_dir():
                                continue
                            for f in channel_dir.rglob(f"*{suffix}.*"):
                                f.unlink()
                        continue

                    print(f"Processed depth and deleted {depth_path}")
                except Exception as e:
                    print(f"Error processing depth {depth_path}: {e}")

            # ------------------------------------------------------------------
            # IMAGE / RGB
            # ------------------------------------------------------------------
            # In Blender 5.0 the compositor only saves multilayer EXR, so the
            # RGB beauty pass arrives as Image####.exr — convert it to PNG.
            image_filename_exr = f"Image{suffix}.exr"
            image_path_exr = frames_folder / "Image" / f"camera_{subcam_id}" / image_filename_exr

            if image_path_exr.exists():
                try:
                    rgb = load_exr(str(image_path_exr))  # returns BGR
                    if rgb is None:
                        raise RuntimeError("load_exr returned None")
                    # BGR -> RGB and tonemap to 8-bit sRGB
                    rgb = rgb[..., ::-1]
                    rgb = np.clip(rgb, 0.0, 1.0)
                    # Simple linear → sRGB approximation
                    rgb_srgb = np.where(
                        rgb <= 0.0031308,
                        12.92 * rgb,
                        1.055 * np.power(np.maximum(rgb, 0.0), 1 / 2.4) - 0.055,
                    )
                    rgb_uint8 = (np.clip(rgb_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                    imageio.imwrite(image_path_exr.with_suffix(".png"), rgb_uint8)
                    image_path_exr.unlink()
                    print(f"Converted Image EXR -> PNG and deleted {image_path_exr.name}")
                except Exception as e:
                    print(f"Error converting Image EXR {image_path_exr}: {e}")

    # --- Cleanup ---
    # The compositor accumulates file output nodes across renders, so earlier
    # cameras' EXRs get re-written by later renders. Delete all leftover EXRs
    # and camview files in one pass.
    import shutil
    for exr in frames_folder.rglob("*.exr"):
        exr.unlink()
    camview_dir = frames_folder / "camview"
    # Multi-view needs the per-camera poses to build transforms.json; keep camview
    # in that case (single-panorama mode has no use for it, so drop it).
    if camview_dir.exists() and n_views <= 1:
        shutil.rmtree(camview_dir)
    tmp_dir = output_folder / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)

    # 8. Multi-view: assemble one transforms.json per rig (anchor + neighbours)
    # from the saved camview poses, then prune leftover EXR / helper folders.
    if n_views > 1:
        t0 = _phase("build manifests")
        from build_multiview_manifest import build_scene_manifests, prune_scene
        build_scene_manifests(output_folder)
        prune_scene(output_folder)
        _phase_done("build manifests", t0)

    _phase_done("render", t0)

    # List final output files
    print("\n=== Output files ===")
    for p in sorted(frames_folder.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(frames_folder)}")

    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_dir', required=True, type=Path, help="Path to city directory (e.g. models/city1), containing a .blend file and HDRI .exr")
    parser.add_argument('-g', '--configs', nargs='+', default=[], help="Gin config files")
    parser.add_argument('-p', '--overrides', nargs='+', default=[], help="Gin config overrides")
    # Multi-view: >1 makes each camera rig a constellation of N sub-cameras (an
    # anchor + neighbours a short baseline apart), all rendered as 360 panoramas
    # and written as one pose-registered transforms.json per rig -- the multi-view
    # supervision a GS head needs. n_views=1 (default) keeps single-panorama rigs.
    parser.add_argument('--n-views', type=int, default=1,
                        help="sub-cameras per rig (1 = single panorama; >1 = multi-view set)")
    parser.add_argument('--baseline', default="0.3",
                        help="multi-view neighbour distance (m): float or 'uniform,lo,hi'")
    parser.add_argument('--z-amplitude', type=float, default=0.2,
                        help="multi-view: +/- vertical parallax for the ring (m)")
    parser.add_argument('--min-near-frac', type=float, default=0.12,
                        help="multi-view: reject anchors whose horizontal band has "
                        "less than this fraction of content within ~8 m (low parallax); 0 to disable")
    parser.add_argument('--resolution', type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(4096, 2048), help="render W,H (default 4096,2048)")
    # Ignored args that manage_jobs might pass
    parser.add_argument('--seed', default=0)
    parser.add_argument('--task', default='')
    parser.add_argument('--task_uniqname', default='')
    
    args = parser.parse_args()
    with suppress_blender_output():
        main(args)
