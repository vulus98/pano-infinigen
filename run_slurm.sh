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

#alternative gpu: pro_6000
# Default to urban if SCENE_TYPE is not provided
SCENE_TYPE=${SCENE_TYPE:-urban}
    
source  ~/miniconda3/etc/profile.d/conda.sh
conda activate infinigen_city

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

elif [ "$SCENE_TYPE" == "urban" ]; then
    city_dir="models/city${SLURM_ARRAY_TASK_ID}"
    echo "Running Urban Generation for ${city_dir}..."
    if [ ! -d "$city_dir" ]; then
        echo "Error: ${city_dir} does not exist, skipping."
        exit 1
    fi
    python process_custom_blend.py --city_dir "$city_dir" \
        -g local_256GB.gin monocular.gin blender_gt.gin \
        -p "camera.spawn_camera_rigs.n_camera_rigs=500" \
           "camera.compute_base_views.max_tries=100000" \
           "camera.spawn_camera_rigs.camera_rig_config=[{'loc':(0,0,0),'rot_euler':(0,0,0)}]" \
        --seed 0

else
    echo "Error: Unknown SCENE_TYPE '$SCENE_TYPE'. Use 'indoor', 'outdoor', or 'urban'."
    exit 1
fi

# Cleanup logic (skip for urban — process_custom_blend.py handles its own cleanup)
if [ "$SCENE_TYPE" != "urban" ]; then
    echo "Cleaning up $base_output..."
    find "$base_output" -type f -name "*.exr" -delete
    find "$base_output" -type d \( -name "coarse" -o -name "fine" -o -name "Objects" -o -name "camview" -o -name "UniqueInstances" -o -name "logs" -o -name "tmp" -o -name "frames_2_0_0048_0" -o -name "frames_1_0_0048_0" \) -exec rm -rf {} +
fi

echo "$(date) finished ${SLURM_JOB_ID}"


# Usage
# sbatch --export=ALL,SCENE_TYPE=outdoor run_slurm.sh
# sbatch --export=ALL,SCENE_TYPE=indoor run_slurm.sh
# sbatch --array=1-50 --export=ALL,SCENE_TYPE=urban run_slurm.sh