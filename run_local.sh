# Generate outdoor 
python -m infinigen.datagen.manage_jobs --output_folder outputs/outdoor --num_scenes 1 \
--configs simple.gin --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin --pipeline_overrides LocalScheduleHandler.use_gpu=True

# python -m infinigen.datagen.manage_jobs --output_folder outputs/my_outdoor_dataset --num_scenes 1 \
# --configs simple.gin fast_solve.gin --overwrite --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin cuda_terrain.gin --cleanup big_files --warmup_sec 1200 \
# --pipeline_overrides LocalScheduleHandler.use_gpu=True \
# --config no_particles no_creatures

# Generate indoor 
# python -m infinigen.datagen.manage_jobs --output_folder outputs/my_indoor_dataset4 --num_scenes 1 --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
#     --configs singleroom.gin fast_solve.gin \
#     --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=16 \
#     --overrides compose_indoors.restrict_single_supported_roomtype=True #\
    # --wandb_mode online