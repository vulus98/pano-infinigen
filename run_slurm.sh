#!/usr/bin/env bash

#SBATCH --job-name=pano-infinigen-array
#SBATCH --array=1-10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --tmp=208000
#SBATCH -o logs/%A_%a.out
#SBATCH --gpus=rtx_4090:1

# Default to outdoor if SCENE_TYPE is not provided
SCENE_TYPE=${SCENE_TYPE:-outdoor}

source  ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen

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

else
    echo "Error: Unknown SCENE_TYPE '$SCENE_TYPE'. Use 'indoor', 'outdoor', or 'multiview'."
    exit 1
fi

---

# Cleanup logic (now uses the dynamic $base_output)
echo "Cleaning up $base_output..."
find "$base_output" -type f -name "*.exr" -delete
find "$base_output" -type d \( -name "coarse" -o -name "fine" -o -name "Objects" -o -name "UniqueInstances" -o -name "imu_tum" -o -name "logs" -o -name "tmp" -o -name "frames_2_0_0048_0" -o -name "frames_1_0_0048_0" \) -exec rm -rf {} +

echo "$(date) finished ${SLURM_JOB_ID}"


# Usage
# sbatch --export=ALL,SCENE_TYPE=type run_slurm.sh
# Multi-view outdoor (8 panoramas/scene, 1 m baseline):
#   sbatch --export=ALL,SCENE_TYPE=multiview,N_VIEWS=8 run_slurm.sh
# Multi-view indoor (0.3 m baseline):
#   sbatch --export=ALL,SCENE_TYPE=multiview,MV_DOMAIN=indoor,N_VIEWS=8 run_slurm.sh