#!/usr/bin/env bash

#SBATCH --job-name=pano-infinigen-array
#SBATCH --array=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4:00:00
#SBATCH --tmp=208000
#SBATCH -o logs/%A_%a.out
#SBATCH --gpus=rtx_4090:1

echo "$(date) start ${SLURM_JOB_ID}"

module load eth_proxy

name=pano-infinigen-array
num_scenes=1
num_concurrent=32
folder_name="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Generate outdoor
python -m infinigen.datagen.manage_jobs --output_folder outputs/outdoor/$folder_name --num_scenes $num_scenes \
    --configs simple.gin \
    --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin \
    --pipeline_overrides LocalScheduleHandler.use_gpu=True \
    --wandb_mode disabled 

# Generate indoor
# python -m infinigen.datagen.manage_jobs --output_folder outputs/indoor/$folder_name --num_scenes $num_scenes --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
#     --configs singleroom.gin fast_solve.gin \
#     --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=$num_concurrent \
#     --overrides compose_indoors.restrict_single_supported_roomtype=True \
#     --wandb_mode online

# Delete unecessary files after generating depth and normals.
find outputs/outdoor/$folder_name/  -type f -name "*.exr" -delete
find outputs/outdoor/$folder_name/  -type d -name "coarse" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "fine" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "Objects" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "camview" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "UniqueInstances" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "logs" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "tmp" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "frames_2_0_0048_0" -exec rm -rf {} +
find outputs/outdoor/$folder_name/  -type d -name "frames_1_0_0048_0" -exec rm -rf {} +