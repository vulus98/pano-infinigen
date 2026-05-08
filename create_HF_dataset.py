import os
import random
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import Dataset, Features, Image as HfImage, Value
from huggingface_hub import login, HfApi, list_repo_files, CommitOperationAdd

# --- Configuration ---
BASE_DIR = "outputs"
REPO_ID = "prs-eth/PanoInfinigen"
HF_TOKEN = os.environ.get("HF_TOKEN")
CHUNK_SIZE = 15

# NOTE: The existing "indoor" and "nature" configs are already on the Hub and
# must stay untouched. We intentionally do NOT include them in CONFIG_MAPPING
# so this run will only ever read/upload the new "urban" category.
#
# Previously uploaded (DO NOT add back unless re-uploading on purpose):
#   "indoor":  "indoor",
#   "outdoor": "nature",
CONFIG_MAPPING = {
    "urban": "urban",
}

def get_args():
    parser = argparse.ArgumentParser(description="Resumable HF Upload")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument(
        "--mini-batch",
        type=int,
        default=8,
        help="Rows loaded+written per streaming iteration. Peak RSS ≈ mini_batch × sample_size. "
             "Drop to 4 or 2 if you still see OOMs.",
    )
    parser.add_argument(
        "--target-part-size-gb",
        type=float,
        default=1.5,
        help="Approximate size of each parquet shard. The writer rolls over to a new file "
             "as soon as the current one exceeds this size.",
    )
    return parser.parse_args()

def get_id1_splits(scene_dir, train_pct=0.85, val_pct=0.075, debug=False):
    if not scene_dir.exists(): return [], [], []
    id1_names = sorted([d.name for d in scene_dir.iterdir() if d.is_dir()])
    random.seed(42)
    random.shuffle(id1_names)
    if debug: return id1_names[:2], id1_names[2:3], id1_names[3:4]
    n_total = len(id1_names)
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)
    return id1_names[:n_train], id1_names[n_train:n_train+n_val], id1_names[n_train+n_val:]

def _paths_legacy(base_dir, target_scene, ids):
    """Two-level layout used by indoor/outdoor:
        <base>/<scene>/<id1>/<id2>/frames/{Image,Depth,SurfaceNormal}/camera_0/*
    Kept for reference / possible re-runs; not used in the current upload.
    """
    paths = []
    scene_path = Path(base_dir) / target_scene
    for id1_name in ids:
        id1_dir = scene_path / id1_name
        for id2 in id1_dir.iterdir():
            if not id2.is_dir(): continue
            img_dir = id2 / "frames" / "Image" / "camera_0"
            if not img_dir.exists(): continue
            for img_p in img_dir.glob("Image_*.png"):
                suffix = img_p.stem.replace("Image_", "")
                depth_p = id2 / "frames" / "Depth" / "camera_0" / f"Depth_{suffix}.npy"
                norm_p = id2 / "frames" / "SurfaceNormal" / "camera_0" / f"SurfaceNormal_{suffix}.npy"
                if depth_p.exists() and norm_p.exists():
                    paths.append({"image_path": str(img_p), "depth_path": str(depth_p), "normals_path": str(norm_p)})
    return paths


def _paths_urban(base_dir, target_scene, ids):
    """One-level layout used by urban:
        <base>/<scene>/<city>/frames/{Image,Depth,Normal}/camera_0/*
    File-naming convention (note the asymmetry — folder is "Normal" but the
    filename prefix is still "SurfaceNormal"):
        Image_<suffix>.png, Depth_<suffix>.npy, SurfaceNormal_<suffix>.npy
    where <suffix> is the full stem after the modality prefix, e.g.
    "0_0_0001_0". The suffix derived from Image_*.png is reused verbatim for
    the depth and normal files, so the three modalities stay matched per frame.
    """
    paths = []
    scene_path = Path(base_dir) / target_scene
    for city_name in ids:
        city_dir = scene_path / city_name
        img_dir = city_dir / "frames" / "Image" / "camera_0"
        if not img_dir.exists(): continue
        for img_p in img_dir.glob("Image_*.png"):
            suffix = img_p.stem.replace("Image_", "")
            depth_p = city_dir / "frames" / "Depth"  / "camera_0" / f"Depth_{suffix}.npy"
            norm_p  = city_dir / "frames" / "Normal" / "camera_0" / f"SurfaceNormal_{suffix}.npy"
            if depth_p.exists() and norm_p.exists():
                paths.append({"image_path": str(img_p), "depth_path": str(depth_p), "normals_path": str(norm_p)})
    return paths


# Dispatch per local folder — picked by name so we never ambiguously apply
# the wrong layout to a category.
_LAYOUT_RESOLVERS = {
    "indoor":  _paths_legacy,
    "outdoor": _paths_legacy,
    "urban":   _paths_urban,
}


def get_paths_for_ids(base_dir, target_scene, ids):
    if target_scene not in _LAYOUT_RESOLVERS:
        raise ValueError(
            f"No path resolver registered for local folder '{target_scene}'. "
            f"Known layouts: {sorted(_LAYOUT_RESOLVERS)}"
        )
    return _LAYOUT_RESOLVERS[target_scene](base_dir, target_scene, ids)

def write_chunk_shards_to_disk(
    paths,
    local_path_template,
    features,
    mini_batch=8,
    target_part_size_bytes=1 * 1024 ** 3,
):
    """Stream a chunk's paths into ~N-GB parquet shards on local scratch.

    Returns the list of local Path objects that were written, in part order.

    Why shard: downstream consumers (and our own upload step) choke on single
    multi-hundred-GB parquet files — slow to load, slow to retry, slow to scan.
    Aiming for ~1 GB shards keeps everything manageable while staying well
    clear of HF's per-file hints (≤20 GB recommended).

    Behavior:
      - Reads `mini_batch` samples at a time; peak RSS ≈ mini_batch × sample_size.
      - Tracks raw bytes written to the current shard; when it crosses
        `target_part_size_bytes`, closes the shard and opens the next one with
        an incremented part index.
      - `local_path_template` must contain a `{part}` placeholder
        (e.g. scratch_dir / "train-00003-{part:05d}.parquet").

    The byte counter is tracked on the uncompressed payload (PNG + npy bytes).
    Since both are already compressed formats, parquet+snappy adds negligible
    overhead, so the shard on disk ends up close to the target.

    Uploading these shards to the Hub is the caller's responsibility — typically
    in a single atomic `create_commit` so that "shard 0 exists on the Hub"
    reliably implies "the whole chunk finished".
    """
    part_idx = 0
    writer = None
    current_path = None
    bytes_written = 0
    written_paths = []

    def _close_current():
        nonlocal writer, current_path, part_idx, bytes_written
        if writer is None:
            return
        writer.close()
        writer = None
        size_mb = current_path.stat().st_size / 1e6
        print(f"    → part {part_idx:05d} closed at {size_mb:.1f} MB")
        written_paths.append(current_path)
        current_path = None
        part_idx += 1
        bytes_written = 0

    try:
        for i in tqdm(range(0, len(paths), mini_batch), desc="  rows"):
            batch = paths[i : i + mini_batch]
            image_bytes  = [Path(p["image_path"]).read_bytes()   for p in batch]
            depth_bytes  = [Path(p["depth_path"]).read_bytes()   for p in batch]
            normal_bytes = [Path(p["normals_path"]).read_bytes() for p in batch]

            rows = {
                "image":   [{"bytes": b, "path": None} for b in image_bytes],
                "depth":   depth_bytes,
                "normals": normal_bytes,
            }
            sub_ds = Dataset.from_dict(rows, features=features)
            table = sub_ds.data.table

            if writer is None:
                current_path = Path(local_path_template.format(part=part_idx))
                writer = pq.ParquetWriter(str(current_path), table.schema, compression="snappy")

            writer.write_table(table)

            bytes_written += (
                sum(len(b) for b in image_bytes)
                + sum(len(b) for b in depth_bytes)
                + sum(len(b) for b in normal_bytes)
            )

            del rows, sub_ds, table, image_bytes, depth_bytes, normal_bytes

            if bytes_written >= target_part_size_bytes:
                _close_current()

        # Flush any trailing rows that didn't reach the threshold.
        _close_current()
        return written_paths

    except BaseException:
        # On error, don't leave partial files behind — the caller shouldn't
        # have to remember to clean up after a crash.
        if writer is not None:
            try: writer.close()
            except Exception: pass
        if current_path is not None and current_path.exists():
            try: os.remove(current_path)
            except OSError: pass
        for p in written_paths:
            try:
                if p.exists(): os.remove(p)
            except OSError: pass
        raise

if __name__ == "__main__":
    args = get_args()
    api = HfApi()
    if HF_TOKEN: login(token=HF_TOKEN)

    # Safety guard: make absolutely sure we don't accidentally touch the
    # already-uploaded categories on the Hub.
    PROTECTED_CONFIGS = {"indoor", "nature"}
    for hf_config in CONFIG_MAPPING.values():
        if hf_config in PROTECTED_CONFIGS:
            raise RuntimeError(
                f"Refusing to run: HF config '{hf_config}' is in the protected "
                f"list {PROTECTED_CONFIGS}. Remove it from CONFIG_MAPPING."
            )

    # 1. Get a list of all files currently on the Hub to enable skipping
    print("Fetching existing file list from Hugging Face...")
    try:
        existing_files = list_repo_files(REPO_ID, repo_type="dataset")
    except Exception:
        existing_files = []

    scratch_dir = Path(os.environ.get("HF_DATASETS_CACHE", "./hf_cache"))
    features = Features({"image": HfImage(), "depth": Value("binary"), "normals": Value("binary")})

    for local_folder, hf_config in CONFIG_MAPPING.items():
        scene_dir = Path(BASE_DIR) / local_folder
        splits = get_id1_splits(scene_dir, debug=args.debug)
        split_names = ["train", "val", "test"]

        for split_name, all_ids in zip(split_names, splits):
            if not all_ids: continue

            for i in range(0, len(all_ids), CHUNK_SIZE):
                chunk_idx = i // CHUNK_SIZE
                config_label = f"{hf_config}-debug" if args.debug else hf_config

                path_in_repo_template = (
                    f"data/{config_label}/{split_name}-{chunk_idx:05d}-{{part:05d}}.parquet"
                )
                local_path_template = str(
                    scratch_dir / f"{split_name}-{chunk_idx:05d}-{{part:05d}}.parquet"
                )

                # SKIP LOGIC: because each chunk is uploaded via a single
                # atomic `create_commit` below, if shard 0 is on the Hub then
                # ALL of the chunk's shards are on the Hub. So the presence of
                # part 00000 alone is a sufficient completion signal — no
                # marker file needed.
                first_part_in_repo = path_in_repo_template.format(part=0)
                if first_part_in_repo in existing_files:
                    print(f"Skipping chunk {chunk_idx:05d} ({split_name}) — already on Hub")
                    continue

                print(f"\n[{hf_config}][{split_name}] Processing Chunk {chunk_idx + 1}...")
                chunk_ids = all_ids[i : i + CHUNK_SIZE]
                paths = get_paths_for_ids(BASE_DIR, local_folder, chunk_ids)
                if not paths: continue

                # Write all shards to scratch first, then upload them together.
                local_parts = write_chunk_shards_to_disk(
                    paths=paths,
                    local_path_template=local_path_template,
                    features=features,
                    mini_batch=args.mini_batch,
                    target_part_size_bytes=int(args.target_part_size_gb * 1024 ** 3),
                )
                if not local_parts:
                    continue

                # One atomic commit for the whole chunk: everything lands
                # together or nothing does. If the job dies here, the Hub is
                # untouched and the retry will reprocess this chunk cleanly.
                operations = [
                    CommitOperationAdd(
                        path_in_repo=path_in_repo_template.format(part=idx),
                        path_or_fileobj=str(p),
                    )
                    for idx, p in enumerate(local_parts)
                ]
                print(f"[{hf_config}] Committing {len(operations)} shard(s) for chunk {chunk_idx:05d} ({split_name})...")
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=(
                        f"Add {config_label}/{split_name} chunk {chunk_idx:05d} "
                        f"({len(operations)} shard{'s' if len(operations) != 1 else ''})"
                    ),
                )

                # Local cleanup after a successful commit.
                for p in local_parts:
                    try:
                        if p.exists(): os.remove(p)
                    except OSError: pass
                for p in scratch_dir.glob("*.arrow"):
                    try: os.remove(p)
                    except OSError: pass

    print("\nUpload sequence resumed and completed.")