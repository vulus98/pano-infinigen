# Indoor
# python -m infinigen.datagen.manage_jobs --output_folder outputs/indoor --num_scenes 2 \
# --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin \
# --configs singleroom.gin fast_solve.gin \
# --pipeline_overrides iterate_scene_tasks.n_camera_rigs=6 get_cmd.driver_script='infinigen_examples.generate_indoors' manage_datagen_jobs.num_concurrent=16 LocalScheduleHandler.use_gpu=True \
# --overrides compose_indoors.restrict_single_supported_roomtype=True camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 compose_indoors.lights_off_chance=0.0

# Outdoor
python -m infinigen.datagen.manage_jobs --output_folder outputs/outdoor --num_scenes 2 \
    --configs simple.gin \
    --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin \
    --pipeline_overrides manage_datagen_jobs.num_concurrent=16 LocalScheduleHandler.use_gpu=True \
    --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0

# Multi-view panoramas (for training a GS head): N pose-registered 360 panoramas
# per scene from one anchor. Keep n_subcams == multiview_rig_config.n_views.
# Outdoor (1 m baseline):
# N_VIEWS=8
# python -m infinigen.datagen.manage_jobs --output_folder outputs/multiview --num_scenes 2 \
#     --configs simple.gin multiview.gin \
#     --pipeline_configs local_256GB.gin multiview.gin blender_gt.gin \
#     --pipeline_overrides manage_datagen_jobs.num_concurrent=16 LocalScheduleHandler.use_gpu=True iterate_scene_tasks.n_subcams=$N_VIEWS \
#     --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 camera.multiview_rig_config.n_views=$N_VIEWS camera.multiview_rig_config.baseline=1.0
# python build_multiview_manifest.py --root outputs/multiview
#
# Indoor (0.3 m baseline): add get_cmd.driver_script='infinigen_examples.generate_indoors',
#   use --configs singleroom.gin fast_solve.gin multiview.gin and
#   --pipeline_configs local_256GB.gin multiview.gin blender_gt.gin indoor_background_configs.gin,
#   with camera.multiview_rig_config.baseline=0.3  (or just: SCENE_TYPE=multiview MV_DOMAIN=indoor run_slurm.sh)
