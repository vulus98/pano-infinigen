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
# Top output folder is the ENVIRONMENT (indoor/outdoor/urban), not the generator
# mode -- so all three datasets live under a consistent outputs/<env>/ tree.
case "$SCENE_TYPE" in
    harvest)   ENV=outdoor ;;
    multiview) ENV="${MV_DOMAIN:-outdoor}" ;;
    *)         ENV="$SCENE_TYPE" ;;   # indoor / outdoor / urban already correct
esac
base_output="outputs/${ENV}/${folder_name}"


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
    # N_RIGS = independent multi-view rigs placed per scene (each an anchor + N_VIEWS
    # sub-cameras). >1 amortizes scene-gen over several rigs -- the practical way to
    # get multiple rigs from an INDOOR scene (the decoupled harvest workflow is
    # outdoor-only, as it needs open sky to place anchors).
    N_VIEWS=${N_VIEWS:-6}
    MV_DOMAIN=${MV_DOMAIN:-outdoor}
    # Terrain-mesh (OcMesher) resolution for native outdoor generation. Match the
    # render resolution so the camera-optimized mesh is ~1 facet/pixel and the ground
    # doesn't show low-poly triangles up close.
    GEN_RES_MV=${GEN_RES_MV:-2048,1024}
    # N_RIGS default is domain-aware: INDOOR amortizes the (expensive) room scene-gen
    # over several rigs -- 5 per room, matching urban's 5 rigs/city -- while OUTDOOR
    # multiview defaults to 1 (the decoupled harvest workflow is the outdoor multi-rig
    # path). Override with N_RIGS=<n>.
    if [ "$MV_DOMAIN" == "indoor" ]; then
        N_RIGS=${N_RIGS:-5}
    else
        # NATIVE outdoor: infinigen places the panorama rig(s) DURING generation and
        # builds terrain + vegetation FOR them (360 deg populated automatically), so a
        # few rigs per scene amortize the gen -- no re-placement (that was the harvest
        # workflow, which left the panoramas barren/blocky). A test placed 2 rigs; 3 is
        # a reasonable default.
        N_RIGS=${N_RIGS:-3}
    fi
    if [ "$MV_DOMAIN" == "indoor" ]; then
        SCENE_CONFIGS="singleroom.gin fast_solve.gin multiview.gin"
        PIPE_CONFIGS="local_256GB.gin multiview.gin blender_gt.gin indoor_background_configs.gin"
        DOMAIN_PIPE_OVR="get_cmd.driver_script='infinigen_examples.generate_indoors'"
        DOMAIN_OVR="compose_indoors.restrict_single_supported_roomtype=True compose_indoors.lights_off_chance=0.0"
    else
        # Native outdoor (nature): pick a biome per scene -- ~30% SPARSE open
        # landscapes, ~70% DENSE (trees/water/rock) -- instead of infinigen's random
        # roll. Same mix the harvest path used; here the scene is built for the rig.
        DENSE_BIOMES=(canyon forest river cliff coast mountain)
        SPARSE_BIOMES=(desert plain arctic snowy_mountain)
        SPARSE_EVERY=${SPARSE_EVERY:-3}   # ~30% sparse
        _idx=${SLURM_ARRAY_TASK_ID:-1}
        if [ "$BIOME" = "random" ]; then _BIOME_CFG=""
        elif [ -n "$BIOME" ]; then _BIOME_CFG="${BIOME}.gin"
        elif [ $(( _idx % SPARSE_EVERY )) -eq 0 ]; then
            _BIOME_CFG="${SPARSE_BIOMES[$(( (_idx / SPARSE_EVERY) % ${#SPARSE_BIOMES[@]} ))]}.gin"
        else
            _BIOME_CFG="${DENSE_BIOMES[$(( _idx % ${#DENSE_BIOMES[@]} ))]}.gin"
        fi
        echo "Native outdoor biome: ${_BIOME_CFG:-<infinigen random>}"
        SCENE_CONFIGS="simple.gin ${_BIOME_CFG} multiview.gin"
        PIPE_CONFIGS="local_256GB.gin multiview.gin blender_gt.gin"
        DOMAIN_PIPE_OVR=""
        # Cap render-phase concurrency for native outdoor. A 2k, multi-rig scene has a
        # huge per-render-process host-RAM footprint; manage_jobs' default of 32
        # concurrent render/GT subprocesses => peak = 32 x footprint => >150 GB OOM.
        # The node's single GPU serializes actual rendering anyway, so high host-side
        # concurrency only multiplies RAM. A small cap keeps GT/CPU work overlapped
        # while bounding peak. Override with MV_NUM_CONCURRENT=<n>.
        num_concurrent=${MV_NUM_CONCURRENT:-6}
        # Terrain-mesh resolution for the native, camera-optimized OcMesher, plus a
        # bigger outdoor baseline range (config default 0.3-0.7 m is small for open
        # scenes) as a gin tuple -- unless the user forced a fixed BASELINE. Cameras
        # are already placed >= min_terrain_distance (=2 m) from terrain.
        DOMAIN_OVR="execute_tasks.generate_resolution=($GEN_RES_MV)"
        [ -z "$BASELINE" ] && DOMAIN_OVR="$DOMAIN_OVR camera.multiview_rig_config.baseline=(\"uniform\", 0.7, 1.4)"
    fi
    # By default the baseline is drawn per-scene from the config's uniform range
    # (baseline diversity). Set BASELINE=<metres> to force a single fixed value.
    if [ -n "$BASELINE" ]; then
        BASELINE_OVR="camera.multiview_rig_config.baseline=$BASELINE"; BL_LABEL="${BASELINE} (fixed)"
    else
        BASELINE_OVR=""; BL_LABEL="config range (variable)"
    fi
    # Optional render resolution override, e.g. RES=1024,512 (default: config res).
    # Render at 2k by default (was 4096x2048 from the scene config) for a manageable
    # dataset, consistent with outdoor/urban. Override with RES=W,H.
    RES=${RES:-2048,1024}
    RES_OVR="render_image.render_resolution_override=($RES)"
    echo "Running Multi-view Panorama Generation (${MV_DOMAIN}, N_RIGS=${N_RIGS}, N_VIEWS=${N_VIEWS}, BASELINE=${BL_LABEL})..."
    python -m infinigen.datagen.manage_jobs --output_folder "$base_output" --num_scenes $num_scenes \
        --configs $SCENE_CONFIGS \
        --pipeline_configs $PIPE_CONFIGS \
        --pipeline_overrides LocalScheduleHandler.use_gpu=True manage_datagen_jobs.num_concurrent=$num_concurrent iterate_scene_tasks.n_subcams=$N_VIEWS iterate_scene_tasks.n_camera_rigs=$N_RIGS $DOMAIN_PIPE_OVR \
        --overrides camera.camera_pose_proposal.pitch=90 camera.camera_pose_proposal.roll=0 camera.multiview_rig_config.n_views=$N_VIEWS camera.spawn_camera_rigs.n_camera_rigs=$N_RIGS $RES_OVR $BASELINE_OVR $DOMAIN_OVR \
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
    N_VIEWS=${N_VIEWS:-6}
    RIGS_PER_SCENE=${RIGS_PER_SCENE:-4}
    # Bigger baseline for stronger parallax (outdoor scenes are open / deep). Each
    # sub-camera still must clear geometry (min_clearance), so very dense scenes may
    # place fewer rigs. Override with BASELINE=<m> or 'uniform,lo,hi'.
    BASELINE=${BASELINE:-uniform,1.0,2.0}
    RES=${RES:-2048,1024}
    # Terrain-mesh resolution for scene GENERATION (independent of the throwaway
    # render size below). infinigen's SphericalMesher meshes terrain + atmosphere
    # 360 deg around the gen camera, sizing facets by this camera resolution. It
    # MUST be near the final render resolution -- (64,32) meshes huge coarse facets
    # (low-poly terrain + blocky atmosphere = the fog/dome sky artifacts). Default
    # to the final RES; override with GEN_RES for speed.
    GEN_RES=${GEN_RES:-$RES}
    SAMPLE_RADIUS=${SAMPLE_RADIUS:-30}

    # Biome-controlled density. infinigen's scene_types is mandatory-exclusive, so
    # if we don't name a biome it picks one at RANDOM -- landing on open
    # deserts/plains ~half the time (sparse, low parallax), which then yields only
    # ~1 rig/scene through the parallax gate. Instead we choose the biome and its
    # frequency: ~2 in 10 scenes SPARSE (open), the rest DENSE (trees/rock/water =
    # near content). With the mix controlled here we DON'T need the per-rig parallax
    # gate, so MIN_NEAR defaults to 0 -> every scene reliably yields RIGS_PER_SCENE
    # rigs, and ~20% of the dataset is (deliberately) sparse.
    #   Override: BIOME=<name> forces one biome; BIOME=random restores infinigen's
    #   random roll; SPARSE_EVERY=<n> makes 1-in-n scenes sparse (default 3 ~= 30%).
    # Ordered so the lushest biomes land on the low array indices (idx%6 = 1,2,4,5 =>
    # forest,river,coast,mountain) -- the rockier canyon/cliff sit at 0,3 -- so small
    # sample runs show nice near-content dense scenes; full production still cycles all.
    DENSE_BIOMES=(canyon forest river cliff coast mountain)
    SPARSE_BIOMES=(desert plain arctic snowy_mountain)
    SPARSE_EVERY=${SPARSE_EVERY:-3}   # ~30% sparse, ~70% dense
    _idx=${SLURM_ARRAY_TASK_ID:-1}
    _is_sparse=0
    if [ "$BIOME" = "random" ]; then BIOME_CFG=""
    elif [ -n "$BIOME" ]; then BIOME_CFG="${BIOME}.gin"
    elif [ $(( _idx % SPARSE_EVERY )) -eq 0 ]; then
        BIOME_CFG="${SPARSE_BIOMES[$(( (_idx / SPARSE_EVERY) % ${#SPARSE_BIOMES[@]} ))]}.gin"; _is_sparse=1
    else
        BIOME_CFG="${DENSE_BIOMES[$(( _idx % ${#DENSE_BIOMES[@]} ))]}.gin"
    fi
    echo "Biome for this scene: ${BIOME_CFG:-<infinigen random>} (sparse=${_is_sparse})"
    # Near-content gate. Even DENSE biomes have clearings, and with the gate off a
    # camera can land in one and shoot an empty, washed-out panorama (no near
    # geometry). So require a small fraction of near content at each anchor for
    # DENSE scenes; keep it OFF for the ~20% SPARSE scenes (deliberately open).
    # find_and_place_anchor resamples up to its try budget, then harvest keeps
    # however many rigs passed (fewer, not a failed scene).
    # Near-content gates are DOMAIN-AWARE:
    #   SPARSE biomes (desert/plain/...) are INTENTIONALLY open -> place any anchor
    #     (MIN_NEAR=0) and NEVER post-render-prune them (RENDER_MIN_NEAR=0); they are
    #     the deliberate ~30% open-landscape share of the dataset.
    #   DENSE biomes should have near content, but the placement raycast over-reads
    #     it, so keep the placement gate modest (0.12; instrumentation showed a real
    #     forest's best anchor is ~0.19) and let the RENDERED-depth prune (0.10) be the
    #     reliable backstop that drops only genuinely empty / washed-out dense rigs.
    if [ "$_is_sparse" = "1" ]; then
        MIN_NEAR=${MIN_NEAR:-0}
        RENDER_MIN_NEAR=${RENDER_MIN_NEAR:-0}
    else
        MIN_NEAR=${MIN_NEAR:-0.12}
        RENDER_MIN_NEAR=${RENDER_MIN_NEAR:-0.10}
    fi
    SPARSE_FRAC=${SPARSE_FRAC:-0}
    # Placement robustness. Nature scenes vary a lot: a DENSE forest has a canopy
    # overhead (little open sky) and trees close on every side, so the default gates
    # (>=10% sky AND all 6 ring-cameras >=0.3 m clear) reject every anchor and the
    # scene yields 0 rigs even though it's rich. Relax the sky floor (a forest floor
    # legitimately sees only a few % sky through the canopy) and give more tries so
    # dense scenes reliably place. The post-render near_frac gate still guards quality.
    MIN_SKY=${MIN_SKY:-0.04}
    MIN_CLEAR=${MIN_CLEAR:-0.3}
    PLACE_TRIES=${PLACE_TRIES:-8000}
    scene_dir="${base_output}/scene"
    mv_dir="${base_output}/multiview"

    # Generate the scene with a THROWAWAY tiny render: manage_jobs only reports a
    # scene "done" once its render tasks succeed, so we keep a 64x32 / 1-sample
    # render (a few seconds) rather than dropping renders entirely. The real
    # panoramas are rendered by harvest_multiview.py at full RES below.
    #
    # NOTE: we deliberately do NOT force camera_pose_proposal.pitch=90 here. This
    # scene's generation camera is throwaway -- it only defines where assets get
    # populated; harvest_multiview.py re-places its own pitch=90 panorama cameras
    # afterward. Forcing the gen camera dead-horizontal over-constrains infinigen's
    # placement and is a frequent "Could not find 1 camera views" scene-gen crash,
    # so we let it use the default (clip_gaussian) pitch distribution instead.
    # Scene-gen is the flaky part (infinigen asset-factory hangs, occasional
    # placement crashes). Retry up to GEN_TRIES times with a fresh scene, each
    # attempt hard-capped at GEN_TIMEOUT so an indefinite asset hang is killed and
    # retried instead of burning the whole job. A fresh scene_dir => new random
    # seed => a different scene, so a retry dodges a scene-specific hang/crash.
    GEN_TRIES=${GEN_TRIES:-3}
    GEN_TIMEOUT=${GEN_TIMEOUT:-50m}
    # Terrain mesher. infinigen's default (OcMesher, from base.gin) meshes the terrain
    # VIEW-DEPENDENTLY for the throwaway generation camera -- fine in its cone, coarse
    # elsewhere. harvest_multiview then re-places 360 deg cameras at DIFFERENT spots,
    # so they see the coarsely-meshed side => big blocky facets. UniformMesher meshes
    # the whole terrain uniformly (view-independent), so it looks good from any
    # harvested camera. Costs more gen time/memory; MESHER_SUBDIV controls density.
    MESHER=${MESHER:-UniformMesher}
    MESHER_SUBDIV=${MESHER_SUBDIV:-448}   # finer terrain (kills near-camera facets); ~5min mesh
    if [ "$MESHER" != "OcMesher" ] && [ "$MESHER" != "SphericalMesher" ]; then
        MESHER_OVR=("fine_terrain.mesher_backend=\"$MESHER\"" "UniformMesher.subdivisions=($MESHER_SUBDIV, -1, -1)")
    else
        MESHER_OVR=("fine_terrain.mesher_backend=\"$MESHER\"")
    fi
    blend=""
    for attempt in $(seq 1 "$GEN_TRIES"); do
        echo "Generating one scene (tiny throwaway render), attempt ${attempt}/${GEN_TRIES}, mesher=${MESHER}..."
        rm -rf "$scene_dir"
        timeout "$GEN_TIMEOUT" python -m infinigen.datagen.manage_jobs --output_folder "$scene_dir" --num_scenes 1 \
            --configs simple.gin $BIOME_CFG \
            --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin \
            --pipeline_overrides LocalScheduleHandler.use_gpu=True manage_datagen_jobs.num_concurrent=1 \
            --overrides "render_image.render_resolution_override=(64, 32)" "execute_tasks.generate_resolution=($GEN_RES)" \
                "configure_render_cycles.num_samples=1" "${MESHER_OVR[@]}" \
            --wandb_mode disabled
        blend=$(find "$scene_dir" -path '*fine/scene.blend' 2>/dev/null | head -1)
        [ -n "$blend" ] && break
        echo "  attempt ${attempt} produced no scene (crash/timeout); retrying with a fresh scene..."
    done
    if [ -z "$blend" ]; then
        echo "ERROR: scene generation produced no fine/scene.blend after ${GEN_TRIES} attempts"; exit 1
    fi
    echo "Harvesting ${RIGS_PER_SCENE} multi-view set(s) from $blend ..."
    python harvest_multiview.py --scene-blend "$blend" --output "$mv_dir" \
        --rigs-per-scene "$RIGS_PER_SCENE" --n-views "$N_VIEWS" --baseline "$BASELINE" \
        --resolution "$RES" --sample-radius "$SAMPLE_RADIUS" \
        --min-near "$MIN_NEAR" --sparse-frac "$SPARSE_FRAC" \
        --render-min-near "$RENDER_MIN_NEAR" \
        --min-sky "$MIN_SKY" --min-clearance "$MIN_CLEAR" --place-tries "$PLACE_TRIES" \
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
    N_VIEWS=${N_VIEWS:-6}
    N_RIGS=${N_RIGS:-40}
    # Bigger baseline: cities are large/open with deep sightlines, so a ~0.4 m ring
    # gave weak parallax. 1-2 m gives strong parallax while sub-cameras stay on the
    # street. Override with BASELINE=<m> or 'uniform,lo,hi'.
    BASELINE=${BASELINE:-uniform,1.0,2.0}
    RES=${RES:-2048,1024}
    # Resampling: dense downtowns put ~2/3 of ring sub-cameras inside a building, so
    # placing exactly N_RIGS yields only ~1/3 complete rigs. Over-place OVERSAMPLE x
    # N_RIGS candidates, cheaply depth-probe each, and keep the first N_RIGS whose
    # cameras all clear geometry -- only those get the full 2k render.
    OVERSAMPLE=${OVERSAMPLE:-3.0}
    city_dir="models/city${SLURM_ARRAY_TASK_ID}"
    echo "Running Urban (iCity) multi-view Generation for ${city_dir} (N_VIEWS=${N_VIEWS}, N_RIGS=${N_RIGS}, OVERSAMPLE=${OVERSAMPLE})..."
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
           "camera.compute_base_views.max_tries=30000" \
           $RIG_OVR \
        --n-views "$N_VIEWS" --baseline "$BASELINE" --resolution "$RES" \
        --oversample "$OVERSAMPLE" \
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
