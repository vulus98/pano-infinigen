import argparse
import sys
from pathlib import Path

# Use the environment variable for OpenCV EXR support BEFORE cv2 import
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import bpy
import numpy as np
import gin
from mathutils import Vector
import cv2
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
from infinigen.core.rendering.post_render import load_normals, reorient_surface_normals_from_camview, colorize_normals, load_depth, colorize_depth
from infinigen.tools.suffixes import get_suffix

def main(args):
    # 1. Load the custom blend file
    # We do this BEFORE init/gin because gin might set scene properties
    if not args.input_blend.exists():
        raise FileNotFoundError(f"{args.input_blend} does not exist")
    
    print(f"Loading {args.input_blend}...")
    bpy.ops.wm.open_mainfile(filepath=str(args.input_blend))
    
    # Ensure any stuck material override is cleared immediately
    if "ViewLayer" in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers["ViewLayer"].material_override = None

    # Ensure we use Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # 2. Apply configs
    # This sets up rendering settings, camera parameters etc on the loaded scene
    print("Applying gin configs...")
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

    # 3. Find floor level (min Z)
    # min_z = float('inf')
    # mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    
    # # Calculate scene bounds to sample from
    # all_points = []
    
    # for obj in mesh_objs:
    #     # Check world coordinates
    #     mw = obj.matrix_world
    #     # Bounding box corners in world space
    #     if obj.bound_box:
    #         bbox_corners = [mw @ Vector(corner) for corner in obj.bound_box]
    #         for corner in bbox_corners:
    #             if corner.z < min_z:
    #                 min_z = corner.z
    #             all_points.append(corner)
            
    # if not all_points:
    #     # Fallback if empty scene
    #     print("Warning: No mesh objects found. Using default bounds.")
    #     min_z = 0
    #     scene_bounds = (Vector((-10, -10, 0)), Vector((10, 10, 0)))
    # else:
    #     min_coords = np.min([v[:] for v in all_points], axis=0)
    #     max_coords = np.max([v[:] for v in all_points], axis=0)
    #     scene_bounds = (Vector(min_coords), Vector(max_coords))

    floor_z = -0.1
    print(f"Detected floor Z level: {floor_z}")

    # 4. Spawn Camera Rig
    camera_rigs = cam_util.spawn_camera_rigs()
    if not camera_rigs:
        raise RuntimeError("Failed to spawn camera rigs")

    # 5. Manual Camera Placement (for testing)
    print("Using manual camera placement...")
    
    # Set your desired location and rotation here
    # Example: 2m up, looking along Y axis (adjust based on your scene)
    manual_loc = (-2.67, -0.3, 0.000673)  
    manual_rot = (np.deg2rad(90), np.deg2rad(0), np.deg2rad(178))  # (pitch, roll, yaw) in radians

    for rig in camera_rigs:
        rig.location = manual_loc
        rig.rotation_euler = manual_rot
        print(f"Placed rig {rig.name} at {manual_loc} with rotation {manual_rot}")

    # The automatic camera Configuration is commented out below:
    """
    # 5. Configure Cameras using Infinigen's search logic
    print("Pre-processing scene for camera placement...")
    
    # Treat all mesh objects as potential obstacles/terrain for camera selection
    scene_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    
    scene_preprocessed = cam_util.camera_selection_preprocessing(
        terrain=None, 
        scene_objs=scene_objs,
        tags_ratio={}, # Relax all tag requirements
        ranges_ratio={}
    )
    
    print("Searching for optimal camera views...")
    # NOTE: 'altitude' and 'pitch' are controlled by gin configs passed in args.overrides
    # Default altitude is 1.5-2.5m, pitch is 90 deg (horizontal)
    
    cam_util.configure_cameras(
        camera_rigs,
        scene_preprocessed=scene_preprocessed,
        init_bounding_box=scene_bounds, # Search within the scene bounds
        terrain_coverage_range=None, # DISABLE terrain coverage check entirely
        min_terrain_distance=0.1, # Reduce min distance to avoid false positives
    )
    """
    
    # 6. Render
    # We render into a 'frames' subdirectory to compatible with Infinigen's folder structure logic
    # which expects to reorganize files from 'frames_folder' into 'frames_folder/../frames' or 'frames_folder/Type/...'
    # By using a 'frames' subfolder, we ensure consistent behavior.
    frames_folder = args.output_folder / "frames"
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
    render_res = (4096, 2048) 
    clip_start = 0.001

    print(f"Rendering to {frames_folder}...")

    # Ensure no material override is active
    if "ViewLayer" in bpy.context.scene.view_layers:
        bpy.context.scene.view_layers["ViewLayer"].material_override = None

    for cam_rig in camera_rigs:
        for cam in cam_rig.children:
            cam.data.clip_start = clip_start
            
            # This renders Beauty RGB + Depth + Normals in one go
            # NOTE: render_image() calls reorganize_old_framesfolder() at the end, 
            # which moves files into subdirectories (e.g. Normal/camera_0/)
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
                    
                    # Save Raw .npy
                    np.save(normal_path.with_name(f"SurfaceNormal{suffix}.npy"), normals_cam.astype(np.float16))
                    
                    # Save Visualization PNG
                    colored = colorize_normals(normals_cam)
                    imageio.imwrite(normal_path.with_name(f"SurfaceNormal{suffix}.png"), colored)
                    
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
                    depth_arr = load_depth(str(depth_path))
                    
                    # Save Raw .npy (float32)
                    np.save(depth_path.with_name(f"Depth{suffix}.npy"), depth_arr.astype(np.float32))
                    
                    # Save Visualization PNG (colorized or normalized)
                    # colorize_depth maps 0-inf to color scale
                    colored_depth = colorize_depth(depth_arr)
                    imageio.imwrite(depth_path.with_name(f"Depth{suffix}.png"), colored_depth)
                    
                    # Clean up: Delete original EXR
                    depth_path.unlink()
                    print(f"Processed depth and deleted {depth_path}")
                except Exception as e:
                    print(f"Error processing depth {depth_path}: {e}")

            # ------------------------------------------------------------------
            # IMAGE / RGB
            # ------------------------------------------------------------------
            # Check if there is an Image EXR (rendering pipeline sometimes outputs both)
            image_filename_exr = f"Image{suffix}.exr"
            image_path_exr = frames_folder / "Image" / f"camera_{subcam_id}" / image_filename_exr
            
            if image_path_exr.exists():
                # We only want PNG for RGB image. 
                # PNG is likely already generated by render pipeline if configured correctly as default.
                print(f"Deleting extra Image EXR: {image_path_exr}")
                image_path_exr.unlink()

    # Cleanup tmp folder
    tmp_dir = args.output_folder / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        import shutil
        print(f"Cleaning up tmp directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    print("Render complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_blend', required=True, type=Path, help="Path to input .blend file")
    parser.add_argument('--output_folder', required=True, type=Path, help="Output folder")
    parser.add_argument('-g', '--configs', nargs='+', default=[], help="Gin config files")
    parser.add_argument('-p', '--overrides', nargs='+', default=[], help="Gin config overrides")
    # Ignored args that manage_jobs might pass
    parser.add_argument('--seed', default=0)
    parser.add_argument('--task', default='')
    parser.add_argument('--task_uniqname', default='')
    
    args = parser.parse_args()
    main(args)
