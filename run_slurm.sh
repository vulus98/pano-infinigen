#!/usr/bin/env bash

#SBATCH --job-name=pano-infinigen-array
#SBATCH --array=1-10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=50:00:00
#SBATCH --tmp=408000
#SBATCH -o logs/%A_%a.out
#SBATCH --gpus=pro_6000:1
# alternative gpu: rtx_4090

# ---------------------------------------------------------------------------
# Single entry point for BOTH generators, selected by one variable SCENE_TYPE:
#   outdoor | indoor | multiview | harvest  -> native infinigen  (bpy 4.2,   env "infinigen")
#   urban                                    -> iCity .blend      (bpy 5.0.1, env "infinigen_city")
# iCity blends are Blender-5.0 files, so they need the bpy-5.0.1 env; the rest
# of infinigen runs on bpy 4.2. The correct conda env is activated automatically.
# ---------------------------------------------------------------------------
SCENE_TYPE=${SCENE_TYPE:-outdoor}

source  ~/miniconda3/etc/profile.d/conda.sh
if [ "$SCENE_TYPE" == "urban" ]; then
    conda activate infinigen_city
else
    conda activate infinigen
fi

echo "$(date) start ${SLURM_JOB_ID} - Mode: ${SCENE_TYPE}"

module load eth_proxy

num_scenes=1
num_concurrent=32
folder_name="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
base_output="outputs/${SCENE_TYPE}/${folder_name}"


if [ "$SCENE_TYPE" == "outdoor" ]; then
    echo "Running Outdoor Generation..."
    python -m infinigen.datagen.manage_jobs --output_folder "$base_output" --num_scenes $num_scenes \
        --configs simple.gin \
        --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin \
        --pipeline_overrides LocalScheduleHandler.use_gpu=True manage_datagen_jobs.num_concurrent=$num_concurrent iterate_scene_tasks.n_camera_rigs=2\
        --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 \
        --wandb_mode disabled

elif [ "$SCENE_TYPE" == "indoor" ]; then
    echo "Running Indoor Generation..."
    python -m infinigen.datagen.manage_jobs --output_folder "$base_output" --num_scenes $num_scenes \
        --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
        --configs singleroom.gin fast_solve.gin \
        --pipeline_overrides LocalScheduleHandler.use_gpu=True get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=$num_concurrent iterate_scene_tasks.n_camera_rigs=6 \
        --overrides compose_indoors.restrict_single_supported_roomtype=True camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 compose_indoors.lights_off_chance=0.0 \
        --wandb_mode disabled

elif [ "$SCENE_TYPE" == "multiview" ]; then
    # Multi-view panoramas for Gaussian-Splatting training: one anchor per scene,
    # N sub-cameras (a small constellation) each rendered as a 360 panorama with
    # its pose saved. N_VIEWS must be passed to BOTH config layers, hence the two
    # overrides (iterate_scene_tasks.n_subcams and multiview_rig_config.n_views).
    # MV_DOMAIN selects the scene family: outdoor (nature) or indoor.
    N_VIEWS=${N_VIEWS:-8}
    MV_DOMAIN=${MV_DOMAIN:-outdoor}
    if [ "$MV_DOMAIN" == "indoor" ]; then
        SCENE_CONFIGS="singleroom.gin fast_solve.gin multiview.gin"
        PIPE_CONFIGS="local_256GB.gin multiview.gin blender_gt.gin indoor_background_configs.gin"
        DOMAIN_PIPE_OVR="get_cmd.driver_script='infinigen_examples.generate_indoors'"
        DOMAIN_OVR="compose_indoors.restrict_single_supported_roomtype=True compose_indoors.lights_off_chance=0.0"
    else
        SCENE_CONFIGS="simple.gin multiview.gin"
        PIPE_CONFIGS="local_256GB.gin multiview.gin blender_gt.gin"
        DOMAIN_PIPE_OVR=""
        DOMAIN_OVR=""
    fi
    # By default the baseline is drawn per-scene from the config's uniform range
    # (baseline diversity). Set BASELINE=<metres> to force a single fixed value.
    if [ -n "$BASELINE" ]; then
        BASELINE_OVR="camera.multiview_rig_config.baseline=$BASELINE"; BL_LABEL="${BASELINE} (fixed)"
    else
        BASELINE_OVR=""; BL_LABEL="config range (variable)"
    fi
    echo "Running Multi-view Panorama Generation (${MV_DOMAIN}, N_VIEWS=${N_VIEWS}, BASELINE=${BL_LABEL})..."
    python -m infinigen.datagen.manage_jobs --output_folder "$base_output" --num_scenes $num_scenes \
        --configs $SCENE_CONFIGS \
        --pipeline_configs $PIPE_CONFIGS \
        --pipeline_overrides LocalScheduleHandler.use_gpu=True manage_datagen_jobs.num_concurrent=$num_concurrent iterate_scene_tasks.n_subcams=$N_VIEWS $DOMAIN_PIPE_OVR \
        --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 camera.multiview_rig_config.n_views=$N_VIEWS $BASELINE_OVR $DOMAIN_OVR \
        --wandb_mode disabled

    # Build the per-scene pose manifests (and prune Objects/UniqueInstances/imu_tum/.exr).
    echo "Building multi-view pose manifests..."
    python build_multiview_manifest.py --root "$base_output" || echo "manifest build failed (non-fatal)"

elif [ "$SCENE_TYPE" == "harvest" ]; then
    # Decoupled workflow (one array task = one scene): generate ONE scene once
    # (the expensive part), then harvest several multi-view panorama sets from it
    # with harvest_multiview.py. Failed placements are free retries on the saved
    # scene; each rig gets its own (optionally random) baseline.
    #   Time scales with RIGS_PER_SCENE * N_VIEWS * resolution -- tune to fit the
    #   SBATCH --time budget (full-res renders are minutes each).
    N_VIEWS=${N_VIEWS:-8}
    RIGS_PER_SCENE=${RIGS_PER_SCENE:-4}
    BASELINE=${BASELINE:-uniform,0.3,0.7}
    RES=${RES:-4096,2048}
    SAMPLE_RADIUS=${SAMPLE_RADIUS:-30}
    # Min fraction of the horizontal band with content within --near-dist for an
    # anchor to be accepted -- rejects sparse, no-parallax poses. Sparse scenes
    # (open terrain) may yield 0 rigs at the default; lower MIN_NEAR to keep them.
    MIN_NEAR=${MIN_NEAR:-0.2}
    scene_dir="${base_output}/scene"
    mv_dir="${base_output}/multiview"

    # Generate the scene with a THROWAWAY tiny render: manage_jobs only reports a
    # scene "done" once its render tasks succeed, so we keep a 64x32 / 1-sample
    # render (a few seconds) rather than dropping renders entirely. The real
    # panoramas are rendered by harvest_multiview.py at full RES below.
    echo "Generating one scene (tiny throwaway render)..."
    python -m infinigen.datagen.manage_jobs --output_folder "$scene_dir" --num_scenes 1 \
        --configs simple.gin \
        --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin \
        --pipeline_overrides LocalScheduleHandler.use_gpu=True manage_datagen_jobs.num_concurrent=1 \
        --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 \
            "render_image.render_resolution_override=(64, 32)" "execute_tasks.generate_resolution=(64, 32)" \
            "configure_render_cycles.num_samples=1" \
        --wandb_mode disabled

    blend=$(find "$scene_dir" -path '*fine/scene.blend' 2>/dev/null | head -1)
    if [ -z "$blend" ]; then
        echo "ERROR: scene generation produced no fine/scene.blend"; exit 1
    fi
    echo "Harvesting ${RIGS_PER_SCENE} multi-view set(s) from $blend ..."
    python harvest_multiview.py --scene-blend "$blend" --output "$mv_dir" \
        --rigs-per-scene "$RIGS_PER_SCENE" --n-views "$N_VIEWS" --baseline "$BASELINE" \
        --resolution "$RES" --sample-radius "$SAMPLE_RADIUS" --min-near "$MIN_NEAR" \
        --seed "${SLURM_ARRAY_TASK_ID:-0}"

    # The multi-view sets are in $mv_dir; the source scene (incl. the ~1-2 GB
    # scene.blend) is no longer needed.
    rm -rf "$scene_dir"

elif [ "$SCENE_TYPE" == "urban" ]; then
    # iCity multi-view panoramas from a pre-made iCity .blend in models/city<N>.
    # process_custom_blend.py registers the iCity addon, remaps libraries, forces
    # daytime / dry city, then places N_RIGS validated rigs -- each a constellation
    # of N_VIEWS sub-cameras (anchor + neighbours a short baseline apart) rendered
    # as 360 panoramas -- and writes one pose-registered transforms.json per rig.
    # Set N_VIEWS=1 to fall back to single independent panoramas (no manifest).
    #   Time scales with N_RIGS * N_VIEWS * resolution (full-res ~3.4 min/camera).
    N_VIEWS=${N_VIEWS:-8}
    N_RIGS=${N_RIGS:-40}
    BASELINE=${BASELINE:-uniform,0.3,0.6}
    RES=${RES:-4096,2048}
    city_dir="models/city${SLURM_ARRAY_TASK_ID}"
    echo "Running Urban (iCity) multi-view Generation for ${city_dir} (N_VIEWS=${N_VIEWS}, N_RIGS=${N_RIGS})..."
    if [ ! -d "$city_dir" ]; then
        echo "Error: ${city_dir} does not exist, skipping."
        exit 1
    fi
    # n_views=1 needs the single-camera rig config; n_views>1 builds its own
    # multi-view constellation inside process_custom_blend.py.
    if [ "$N_VIEWS" -gt 1 ]; then RIG_OVR=""; else
        RIG_OVR="camera.spawn_camera_rigs.camera_rig_config=[{'loc':(0,0,0),'rot_euler':(0,0,0)}]"; fi
    python process_custom_blend.py --city_dir "$city_dir" \
        -g local_256GB.gin monocular.gin blender_gt.gin \
        -p "camera.spawn_camera_rigs.n_camera_rigs=$N_RIGS" \
           "camera.compute_base_views.max_tries=100000" \
           $RIG_OVR \
        --n-views "$N_VIEWS" --baseline "$BASELINE" --resolution "$RES" \
        --seed 0

else
    echo "Error: Unknown SCENE_TYPE '$SCENE_TYPE'. Use outdoor | indoor | multiview | harvest | urban."
    exit 1
fi

# Cleanup (urban handles its own cleanup inside process_custom_blend.py).
if [ "$SCENE_TYPE" != "urban" ]; then
    echo "Cleaning up $base_output..."
    find "$base_output" -type f -name "*.exr" -delete
    find "$base_output" -type d \( -name "coarse" -o -name "fine" -o -name "Objects" -o -name "UniqueInstances" -o -name "imu_tum" -o -name "logs" -o -name "tmp" -o -name "frames_2_0_0048_0" -o -name "frames_1_0_0048_0" \) -exec rm -rf {} +
fi

echo "$(date) finished ${SLURM_JOB_ID}"


# Usage (single variable SCENE_TYPE selects the generator + env):
#   Native infinigen (bpy 4.2, env "infinigen"):
#     sbatch --export=ALL,SCENE_TYPE=outdoor run_slurm.sh
#     sbatch --export=ALL,SCENE_TYPE=indoor  run_slurm.sh
#     sbatch --array=1-500 --export=ALL,SCENE_TYPE=multiview,N_VIEWS=8 run_slurm.sh
#     sbatch --array=1-500 --export=ALL,SCENE_TYPE=harvest,RIGS_PER_SCENE=4,N_VIEWS=8 run_slurm.sh
#   iCity urban multi-view (bpy 5.0.1, env "infinigen_city"; one array index per models/city<N>):
#     sbatch --array=1-50 --export=ALL,SCENE_TYPE=urban,N_VIEWS=8,N_RIGS=40 run_slurm.sh
