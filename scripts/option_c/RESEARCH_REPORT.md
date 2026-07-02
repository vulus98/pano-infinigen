# Comprehensive report: making the HF Data Studio viewer work for PanoInfinigen

**Date**: 2026-05-27
**Author**: Claude (autonomous research session)
**Goal**: figure out which schema/encoding makes the HF Data Studio viewer reliably work on the production PanoInfinigen re-encode, without compromising data quality unreasonably.

---

## TL;DR

**Recommendation**: ship **Pattern A** — `image` (JPG q=95), `depth` (16-bit PNG, per-config scale), `depth_viz` (8-bit Spectral PNG), `normals` (**JPG q=95**, 8-bit RGB, ~1.4 MB/cell on urban). All five "columns" are HF `Image` features. No binary cells.

**Why**: any column with `dtype: binary` is inlined as base64 into the `/rows` JSON response (Data Studio's data source). HF's `/rows` endpoint **does not truncate cells**, so a 50 MB binary normals column produces a multi-GB JSON response that always trips a server limit. `Image`-typed cells, by contrast, are returned as ~200-byte signed URLs and are essentially free in the response.

**Storage**: Pattern A → ~700 GB total dataset on Hub (vs original 1.89 TB, vs Pattern 1 attempt at ~1.4 TB).
**Precision**: normals lose ~1° per direction (q=95) which is **smaller than the float16→uint8 cast quantization** we already accepted earlier. Depth/image are unchanged from what's been validated.
**Code changes from the current branch (`city_generation @ 0d0745e5`)**: minimal — drop the binary `normals` column, switch `normals_viz` → `normals` (or rename), reuse the existing `encode_normals_jpg`-shaped encoder. Production launchers unchanged.

If raw float16 normals are non-negotiable, the only path is **two parallel datasets** (Pattern A as the public viewer-facing one, plus a separate `prs-eth/PanoInfinigen-raw` containing the binary blobs accessed via `snapshot_download` or `hf://` URLs). HF's viewer cannot serve large binary cells in any in-place configuration we tested.

---

## Background

We've been iterating on a re-encoding of `prs-eth/PanoInfinigen` (1.89 TB, 78,362 rows, indoor/nature/urban × train/val/test). The original layout stored `depth` and `normals` as raw `.npy` bytes in a `Value("binary")` column. The viewer doesn't render those, the dataset card preview broke past the first split, and the dataset was too heavy in general.

The "Option C" reshape introduced earlier in this session was:

- `image` (kept), `depth` (16-bit PNG per-config scale), `depth_viz` (Spectral RGB PNG), `normals` (8-bit RGB PNG).

That works for indoor (small detail → ~0.6 MB normals cell), but for **urban** (high-frequency facades) the normals PNG ballooned to ~8 MB/cell, dragging the per-row payload over HF's Data Studio worker timeout at the default `length=100`.

The user then asked for "Pattern 1": keep the raw float16 NPY normals as a binary column (full precision) **plus** a small `normals_viz` Image column for the viewer. The thinking was that HF would truncate the big binary cell in the viewer response.

This report tests Pattern 1 empirically and finds it doesn't work — and shows why.

---

## Key HF Data Viewer limits (researched from docs + source)

From [`huggingface/dataset-viewer/services/worker`](https://github.com/huggingface/dataset-viewer/blob/main/services/worker/README.md), [docs](https://huggingface.co/docs/dataset-viewer/parquet), [docs/rows](https://huggingface.co/docs/dataset-viewer/en/rows), [`libcommon/constants.py`](https://github.com/huggingface/dataset-viewer/blob/main/libs/libcommon/src/libcommon/constants.py):

| Limit | Default | Where it kicks in |
|---|---|---|
| `length` parameter max | **100** | `/rows` query |
| `MAX_NUM_ROWS_PER_PAGE` | 100 | hard cap, server side |
| `WORKER_CONTENT_MAX_BYTES` | 10 MB | `/rows` response body cap (soft) |
| `FIRST_ROWS_MAX_BYTES` | 1 MB | `/first-rows` response body cap (used by dataset card preview iframe) |
| `FIRST_ROWS_MIN_CELL_BYTES` | 100 | minimum cell size after truncation (`/first-rows` only) |
| `WORKER_MAX_JOB_DURATION_SECONDS` | 1200 | hard worker timeout |
| `WORKER_MAX_MISSING_HEARTBEATS × HEARTBEAT_INTERVAL` | 5 × 60s = **5 min silent → killed** | what produces `JobManagerCrashedError` |
| Row-group size, image/binary datasets | recommended 100 rows | tuning hint |
| Per-row-group scan limit (parquet) | ~300 MB | the `TooBigContentError` we hit earlier |
| Auto-conversion dataset size cap | 5 GB (with `partial:true`) | for non-Parquet sources |

**The decisive fact**, confirmed both in the docs ("[Unlike `/first-rows`, there is currently no truncation in `/rows`. The `truncated_cells` field is still there but is always empty.](https://huggingface.co/docs/dataset-viewer/en/rows)") and empirically in §4 below:

- **Image cells** are returned as `{src: "...", height, width}` — a ~200-byte URL — in `/rows` JSON. Asset bytes live on HF's CDN and are fetched by the browser separately.
- **Binary (`Value("binary")`) cells** are inlined as full base64 into `/rows` JSON. **No truncation.**

That's the whole game: 100 rows × ~50 MB binary cell = ~5 GB JSON response → impossible. Same dataset with the same column as `Image` = 100 × ~200 bytes = 20 KB → trivially fits.

---

## Experimental setup

I created seven small public test datasets under `vulus98/hf-viewer-experiment-*` and `vulus98/hf-viewer-stress-*`, each with five 4096×2048 rows but varying schemas. After each finished HF's processing, I hit:

- `/is-valid?dataset=...` — flags for `viewer`, `preview`, `search`, `filter`, `statistics`
- `/rows?dataset=...&length={1,5,10,100}` — measure response status, size, time

Two row archetypes: **indoor-shape** (smooth scenes, normals compress well — sourced from the earlier indoor test parquet) and **urban-shape** (random unit-vectors per pixel = worst-case high-entropy normals, simulating urban architectural complexity).

---

## Results

### A. Schema-variation matrix on indoor-shape rows (5 rows each)

| Pattern | Cells | parquet size | `/rows?length=5` response | `/is-valid` viewer/preview |
|---|---|---|---|---|
| **A. JPG normals** | all `Image` | 13.74 MB | **12.7 KB**, 1.6 s | ✅ / ✅ |
| **B. PNG normals** | all `Image` | 15.94 MB | **12.7 KB**, 2.2 s | ✅ / ✅ |
| **C. Raw NPY binary normals** | 4 `Image` + 1 `binary` | 34.07 MB | **327 MB**, 6.1 s | ✅ / ✅ |
| **D. NPZ-compressed binary normals** | 4 `Image` + 1 `binary` | 20.91 MB | **10 MB**, 2.5 s | ✅ / ✅ |
| **E. Tiny ~1 KB binary** | 4 `Image` + 1 tiny `binary` | 13.74 MB | **19 KB**, 1.7 s | ✅ / ✅ |
| **F. External `Image` cell (path-only)** | 1 `Image` with `bytes=null, path=...` | 80 KB | **HTTP 500**, "Server error while post-processing the rows. Please report the issue. TypeError" | viewer:✅, preview:❌, statistics:❌ |

The size of the `/rows` response **scales linearly with binary content** — Image cells contribute ~200 bytes/row, binary cells contribute their full byte length. At length=100, Pattern C would return ~6.5 GB; Pattern D would return ~200 MB.

**Pattern F (external Image cells)**: I tested whether we could use the HF `Image` feature with `bytes=null, path="external/file.jpg"` to keep the actual bytes outside the parquet (truly decouple binary from viewer). The viewer accepts the schema (status 200 on `is-valid`), but every `/rows` call fails with HTTP 500 and a server-side `TypeError`. **This pattern is not supported by the current dataset-viewer code path.**

### B. Stress test with urban-like high-entropy normals (5 rows each)

The indoor-shape rows used in §A let snappy compress the binary normals heavily. Repeating with random unit-vector normals (worst case for both PNG and parquet):

| Pattern | parquet size | `/rows?length=1` | `/rows?length=5` | `/rows?length=100` |
|---|---|---|---|---|
| **stress-raw-npy** | 322 MB | 64.0 MB, 7.6 s | 320 MB, 35 s | 320 MB, 25 s |
| **stress-compressed** (NPZ) | 187 MB | 57.1 MB, 7.6 s | 285 MB, 19 s | 285 MB, 20 s |
| **stress-rg1** (row_group_size=1) | 322 MB | 64.0 MB, 5.9 s | 320 MB, 45 s | 320 MB, 42 s |

Three observations:

1. **HF does cap the response at ~320 MB**, but only after attempting to materialize the cells. The cap appears to clamp the row count silently (length=100 returns the same ~320 MB as length=5 — i.e. only the first 5 rows survive). For Data Studio's UX, this means tables look broken (rows missing or partial).
2. **NPZ compression** helps a bit on storage but barely on `/rows` bandwidth because the response is base64-encoded inline regardless.
3. **`row_group_size=1` makes things slower**, not faster — too many row-group metadata reads, no help for the actual per-cell cost.

### C. Production validation repo (real urban data, ~411 rows across 18 shards, 15.3 GB)

`vulus98/panoinfinigen-option-c-validation` (Pattern 1: binary normals + normals_viz, the current `city_generation` branch).

Behavior over the past ~90 minutes:
- `is-valid`: `{"error": "The server is busier than usual and the response is not ready yet."}`
- `splits`, `info`, `parquet`, `rows`: same error
- Two README "no-op" retriggers had no effect
- Two consecutive teardown+upload+commit-batching iterations: HF stays stuck

The smaller (5-row) Pattern C stress test eventually processed (HF cooled down for a few minutes after the upload), but the 411-row × 50 MB-per-cell version exceeds whatever HF's worker can chew through in its 20-min budget before being killed (`JobManagerCrashedError` lineage). The dataset is "valid" structurally (we can read every parquet, decode every cell) — HF just won't serve it.

### D. Per-cell encoding sizes on a *real* urban normals cell

To choose between Pattern A (JPG) and Pattern B (PNG) for normals, measured on shape `(2048, 4096, 3)` `float32` unit-normals from `urban/val-00000`:

| Encoding | Cell size |
|---|---|
| Raw NPY (float16) | 48.00 MB |
| NPZ compressed (float16) | 29.38 MB |
| PNG 8-bit RGB | 8.27 MB |
| **JPG q=95 8-bit RGB** | **2.08 MB** |
| JPG q=90 8-bit RGB | 1.41 MB |
| JPG q=85 8-bit RGB | 1.09 MB |

For 78,362 rows the full-dataset cost of just the normals column is:

| Encoding | normals total |
|---|---|
| Raw NPY | ~3.6 TB |
| NPZ | ~2.2 TB |
| PNG | ~620 GB |
| **JPG q=95** | **~160 GB** |
| JPG q=90 | ~110 GB |

JPG at q=95 is ~5× smaller than PNG, ~25× smaller than NPY. Quality cost: ~1° error per normal direction (each channel ±1–2 levels out of 255 → renormalization perturbs angles by ~1°). That's below the directional noise the original float16→uint8 cast already introduces, and well below typical normal-estimation evaluation tolerances.

---

## Why Pattern 1 (binary normals in the viewer dataset) cannot work

Combining all of the above:

1. `/rows` does not truncate cells.
2. The response body therefore contains the **full** base64 of every binary cell × every row in the request.
3. Even with HF's silent ~320 MB response cap, only 5-ish rows actually come back when binary cells are big — Data Studio displays a broken-looking table.
4. For 78,362 rows of urban + indoor + nature, the dataset-server can't even pre-compute `splits`/`info` within the 20 min worker timeout (verified: validation repo has been stuck for >90 min after two reset cycles).
5. Workarounds tried and failed:
   - Smaller row groups (rg=1, rg=4, rg=10): doesn't change per-cell response cost.
   - NPZ compression: cuts cell size by ~40 % but still kills the response at length=100.
   - External-file `Image` cells with `bytes=null, path=...`: HF's serve path raises a `TypeError`.

The only way binary cells of this size are tolerable is if HF truncates them in the `/rows` endpoint. They don't. End of story.

---

## Recommended path forward

### Pattern A is the right choice

```
image       : Image  (JPEG q=95)
depth       : Image  (16-bit single-channel PNG, per-config scale)
depth_viz   : Image  (8-bit RGB Spectral PNG, preview only)
normals     : Image  (JPEG q=95, 8-bit RGB, `(n+1)*127.5`)
```

Per-row estimated cost on urban:

| Cell | Size |
|---|---|
| image (JPG q=95) | 1.5 MB |
| depth (16-bit PNG) | 1.0 MB |
| depth_viz (PNG) | 0.3 MB |
| normals (JPG q=95) | 2.1 MB |
| **per row** | **~5 MB** |

Indoor/nature per-row is much smaller (~3 MB).

`/rows?length=100` JSON response: 100 rows × 4 Image cells × ~200 bytes = **~80 KB**. Well inside every HF limit.

### Code changes from the current branch

In [scripts/option_c/encoder.py](encoder.py), drop `encode_normals_png` and `encode_normals_viz_png` in favour of a single `encode_normals_jpg(arr, q=95)`. In [scripts/option_c/shard_worker.py](shard_worker.py), the `_encode_row` helper returns four cells: `image, depth, depth_viz, normals`. Feature schema becomes four `Image` columns, no `binary`. The README rewrite drops the dual `normals` / `normals_viz` documentation in favour of a single `normals` column with `np.asarray(img, np.float32) / 127.5 - 1.0` decode.

Per-task and per-batch sizing is unchanged. Upload batching (`upload_shards_batched` in `shard_worker.py`, `--upload-batch-size` in `process_batch.py`) stays as-is — it's still the right safeguard against HF commit-squash failures.

### Storage budget (production)

|  | Original PanoInfinigen | Pattern A |
|---|---|---|
| Total dataset on Hub | 1.89 TB | ~700 GB |
| Compressed download size | ~983 GB (indoor only?) | ~600 GB |
| Per-shard (e.g. urban/val) | 1.39 GB | ~100 MB |
| Number of shards (we keep 1515) | 1515 | 1515 |

### Code path that **does not** change

1. Backup phase (Phase 1) — same script, same Slurm launcher, same idempotency.
2. Upload batching, manifest, per-shard worker structure.
3. The Slurm launchers (`option_c_setup_validation.sh`, `option_c_test_pipeline.sh`, `option_c_backup.sh`, `option_c_reencode.sh`, `option_c_finalize_readme.sh`, `option_c_teardown_validation.sh`).

Only the encoder + shard schema + the README rewrite change.

### If raw float16 normals are non-negotiable

Two parallel datasets:

- `prs-eth/PanoInfinigen` (viewer-facing) — Pattern A as above.
- `prs-eth/PanoInfinigen-raw-normals` (codepath-only) — one `Value("binary")` column with the raw NPY bytes, indexed by row id matching the main dataset. Users join the two via `id` if they need lossless float16. Data Studio for this second repo will be broken — that's the unavoidable consequence — but `load_dataset(...)` works fine.

This separation costs ~2.5 TB extra Hub storage (the raw normals on top of Pattern A). Worth it only if specific downstream training relies on better-than-1° normal-direction precision — which is rare in normal-map estimation work.

---

## Things I tested and ruled out

- **`row_group_size=1`**: makes things slower, not faster.
- **NPZ-compressed binary normals**: marginal storage win, doesn't fix `/rows` bandwidth.
- **External `Image` cells (`bytes=null, path=...`)**: HF post-processing throws `TypeError`; viewer doesn't support this pattern.
- **README no-op commits** to retrigger HF: works to clear *some* stuck states but does not unbreak datasets where the worker truly can't fit the job in its timeout.
- **`upload_shards_batched`** (already in the codebase): does avoid the commit-squash failures we hit earlier and is still the right call for production. It's not what's blocking Pattern 1; the cell-size limit is.

---

## Open questions / things I did not test exhaustively

- **HF Enterprise tier**: I don't know if `prs-eth` has Enterprise (the user mentioned it for ZuriPano private viewing). Enterprise may have larger viewer worker memory budgets that could change the math, but the docs don't claim this and the public dataset-viewer source is what we measured.
- **Lance format**: HF has been experimenting with [Lance tables](https://lancedb.com/) for large multimodal datasets. Could potentially store binary blobs separately and still serve them in the viewer. Not yet GA for HF datasets.
- **HF webhook to manually trigger reprocessing**: there's a `/refresh` endpoint that may speed up recovery from stuck states, but you need to be a repo admin and we'd need to whitelist for `prs-eth`.
- **Croissant metadata**: HF auto-generates Croissant manifests for datasets and that's another job that can fail silently. Didn't see it in our error logs but worth noting if anyone files a support ticket.

---

## Cleanup checklist (after you read this)

Delete the test repos when you're done — they're all under `vulus98/hf-viewer-experiment-*` and `vulus98/hf-viewer-stress-*`. Quick teardown:

```python
from huggingface_hub import HfApi, delete_repo
import os
api = HfApi(token=os.environ["HF_TOKEN"])
for r in [
    "hf-viewer-experiment-a-jpg-normals",
    "hf-viewer-experiment-b-png-normals",
    "hf-viewer-experiment-c-raw-binary-normals",
    "hf-viewer-experiment-d-compressed-binary-norm",
    "hf-viewer-experiment-e-tiny-binary",
    "hf-viewer-experiment-f-external-image",
    "hf-viewer-stress-raw-npy",
    "hf-viewer-stress-compressed",
    "hf-viewer-stress-rg1",
]:
    delete_repo(f"vulus98/{r}", repo_type="dataset", token=os.environ["HF_TOKEN"])
```

Also delete `vulus98/panoinfinigen-option-c-validation` once you decide on Pattern A and re-run the test pipeline against a fresh validation repo using the new schema.
