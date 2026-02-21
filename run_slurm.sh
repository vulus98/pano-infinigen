#!/usr/bin/env bash

#SBATCH --job-name=pano-infinigen-array
#SBATCH --array=1-8`000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --tmp=208000
#SBATCH -o logs/%A_%a.out
#SBATCH --gpus=rtx_4090:1

# Default to outdoor if SCENE_TYPE is not provided
SCENE_TYPE=${SCENE_TYPE:-outdoor}

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

else
    echo "Error: Unknown SCENE_TYPE '$SCENE_TYPE'. Use 'indoor' or 'outdoor'."
    exit 1
fi

---

# Cleanup logic (now uses the dynamic $base_output)
echo "Cleaning up $base_output..."
find "$base_output" -type f -name "*.exr" -delete
find "$base_output" -type d \( -name "coarse" -o -name "fine" -o -name "Objects" -o -name "camview" -o -name "UniqueInstances" -o -name "logs" -o -name "tmp" -o -name "frames_2_0_0048_0" -o -name "frames_1_0_0048_0" \) -exec rm -rf {} +

echo "$(date) finished ${SLURM_JOB_ID}"


# Usage
# sbatch --export=ALL,SCENE_TYPE=type run_slurm.sh