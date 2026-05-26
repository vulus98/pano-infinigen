# PanoInfinigen Option C Re-encode

Re-encodes `prs-eth/PanoInfinigen` (1.89 TB, 78,362 rows across indoor/nature/urban × train/val/test) from its current `{image: PNG, depth: binary float16 NPY, normals: binary float16 NPY}` layout into the viewer-friendly Option C layout:

| Column      | Source                  | New encoding                       |
| :---        | :---                    | :---                               |
| `image`     | PNG (kept as-is)        | PNG (8-bit RGB)                    |
| `depth`     | binary NPY float16      | 16-bit single-channel PNG (per-config scale) |
| `depth_viz` | *(new)*                 | 8-bit Spectral RGB PNG, log-depth preview |
| `normals`   | binary NPY float16, (H,W,3) | 8-bit RGB PNG, `(n + 1) / 2 * 255` |

Per-config depth max (defined in [`__init__.py`](__init__.py); matches the
upstream renderer's hard clip so no precision is wasted on unreachable values):
- indoor : `75 m`  (source clip 75 m;  precision 1.14 mm)
- nature : `75 m`  (source clip 75 m;  precision 1.14 mm)
- urban  : `500 m` (source clip 500 m; precision 7.63 mm)

This pipeline is the exact recipe validated on
[`vulus98/panoinfinigen-option-c-test`](https://huggingface.co/datasets/vulus98/panoinfinigen-option-c-test)
(20-row preview, since deleted) and on
[`prs-eth/ZuriPano`](https://huggingface.co/datasets/prs-eth/ZuriPano)
(100-row outdoor LiDAR set, currently live).

## Files

| Path                                            | Role                                            |
| :---                                            | :---                                            |
| `scripts/option_c/__init__.py`                  | Per-config constants (`DEPTH_MAX_M`)            |
| `scripts/option_c/encoder.py`                   | Pure encoders: depth / normals / depth_viz PNG. |
| `scripts/option_c/manifest.py`                  | Build the shard manifest by querying HF.        |
| `scripts/option_c/shard_worker.py`              | `backup → reencode → upload` for one shard.     |
| `scripts/option_c/process_batch.py`             | Slurm-array entry: handles a slice of shards.   |
| `scripts/option_c/update_readme.py`             | Final dataset-card YAML/markdown rewrite.       |
| `scripts/option_c/test_repo_stub_README.md`     | README stub for the validation test repo.       |
| `scripts/launch/option_c_setup_validation.sh`   | **0. Setup**: generate manifest + create the validation repo + push its stub README. |
| `scripts/launch/option_c_test_pipeline.sh`      | **A. Test**: end-to-end validation on urban/val. |
| `scripts/launch/option_c_backup.sh`             | **B. Phase 1**: mirror every shard to scratch. |
| `scripts/launch/option_c_reencode.sh`           | **C. Phase 2**: re-encode + upload (overwrites). |
| `scripts/launch/option_c_finalize_readme.sh`    | **D. Phase 3**: rewrite the dataset card.       |
| `scripts/launch/option_c_teardown_validation.sh`| **E. Cleanup** (optional): delete the validation repo. |

## Workflow

Everything is launched through `sbatch`. **No interactive Python required.**
Set `HF_TOKEN` in your shell once, then submit the scripts in order.

```bash
export HF_TOKEN='hf_...'   # write scope on prs-eth/PanoInfinigen
                           # and on vulus98/panoinfinigen-option-c-validation
cd /path/to/pano-infinigen
```

### 0. Setup the validation repo + manifest (~1 min)

```bash
sbatch scripts/launch/option_c_setup_validation.sh
```

Generates the shard manifest, creates `vulus98/panoinfinigen-option-c-validation` on the Hub as a public dataset (no-op if it already exists), and uploads the stub README.

### A. Small-scale validation (~30 min wall)

```bash
sbatch scripts/launch/option_c_test_pipeline.sh
```

Backs up + re-encodes + uploads only the 18 shards of `urban/val` (~23 GB), targeting the validation repo. Production is untouched.

When the job finishes, refresh https://huggingface.co/datasets/vulus98/panoinfinigen-option-c-validation/viewer and confirm:
1. All four columns render — `image`, `depth`, `depth_viz`, `normals`.
2. `depth_viz` colors span the full Spectral gradient (per-image stretch).
3. Decoded `depth` in a notebook returns metric values consistent with the source.

### B. Phase 1 — Backup every shard (~few hours)

```bash
sbatch --export=ALL,NUM_TASKS=32 scripts/launch/option_c_backup.sh
```

Mirrors all 1,515 parquet shards to `/cluster/scratch/$USER/panoinfinigen_backup/`. Idempotent — restartable jobs skip files that already exist with matching sizes. **Don't proceed until this is fully green.**

### C. Phase 2 — Re-encode + upload (~10–24 h wall with NUM_TASKS=64)

```bash
sbatch --export=ALL,NUM_TASKS=64 scripts/launch/option_c_reencode.sh
```

Each task: reads its slice of `manifest[task::NUM_TASKS]` from the local backup, encodes Option C, uploads to `prs-eth/PanoInfinigen` at `data/<config>/<shard>.parquet` (overwriting). Local re-encoded shards are deleted after a successful upload (use `--keep-new-after-upload` to keep them around for debugging).

Tune `NUM_TASKS` higher (e.g. 128) for faster wall time at the cost of more concurrent HF uploads. HF appears to handle 64 parallel uploads from one account without rate-limiting in practice.

Multiple back-to-back submissions are safe — every step is idempotent.

### D. Phase 3 — Rewrite the dataset card (~1 min)

```bash
sbatch scripts/launch/option_c_finalize_readme.sh
```

Drops the stale `dataset_info` YAML block (forces HF to re-derive features from the new parquets — this is what fixed the cast errors on ZuriPano), and rewrites the *Data Structure* and *How to Use* sections to match the new schema.

### E. Tear down the validation repo (~1 min, optional)

```bash
sbatch scripts/launch/option_c_teardown_validation.sh
```

### Chained one-shot submission (optional)

If you'd rather submit everything in one go and let Slurm dependencies serialize the phases:

```bash
J0=$(sbatch --parsable scripts/launch/option_c_setup_validation.sh)
JT=$(sbatch --parsable --dependency=afterok:$J0   scripts/launch/option_c_test_pipeline.sh)
# (pause to inspect the viewer here — break the chain by re-running from JB if happy)
JB=$(sbatch --parsable --dependency=afterok:$JT   scripts/launch/option_c_backup.sh)
JR=$(sbatch --parsable --dependency=afterok:$JB   scripts/launch/option_c_reencode.sh)
JF=$(sbatch --parsable --dependency=afterok:$JR   scripts/launch/option_c_finalize_readme.sh)
echo "setup=$J0 test=$JT backup=$JB reencode=$JR finalize=$JF"
```

## Operational notes

- **Resumability**. Every phase is idempotent:
  - `backup_shard` skips files that already exist with the expected size;
  - `reencode_shard` skips if the local file already has the four Option-C columns;
  - `upload_shard` skips if `repo_info().siblings` shows the target path at the matching size.
  Restart any failed Slurm job and it picks up cleanly.

- **Transitional viewer state**. During Phase 2, the dataset on HF will have a mix of old-schema and new-schema shards. The viewer will report cast errors until every shard for at least one config is replaced. This is unavoidable when overwriting in place; if you need zero-downtime, swap `TARGET_PREFIX=data/` for `TARGET_PREFIX=data_v2/`, then flip the README's `data_files:` paths atomically and delete `data/` afterwards. *(Discussed but not chosen for this run.)*

- **Disk budget on `/cluster/scratch/$USER/`**:
  - Backup: ~1.76 TB.
  - Re-encoded (transient, deleted after each upload): ≤ ~10 GB per concurrent task.
  - Recommended free space: ≥ 2 TB.

- **Per-shard timing** (measured on the test dataset, indoor/val shard, 88 rows, 8 CPUs):
  - Backup: 15-30 s (network bound).
  - Re-encode: 60-90 s (CPU bound, decode + 4 × PNG encode per row).
  - Upload: 20-40 s (depends on HF upstream).
  - With 64 parallel tasks: ~14 hours expected total wall time for Phase 2.
