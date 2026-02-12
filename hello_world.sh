# python -m infinigen.datagen.manage_jobs --output_folder outputs/hello_world --num_scenes 1 --specific_seed 0 \
# --configs simple.gin --pipeline_configs local_16GB.gin monocular.gin blender_gt.gin --pipeline_overrides LocalScheduleHandler.use_gpu=True

python -m infinigen.datagen.manage_jobs --output_folder outputs/my_dataset --num_scenes 1 --pipeline_configs \
local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin --configs singleroom.gin \
--pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=16 \
--overrides compose_indoors.restrict_single_supported_roomtype=True 