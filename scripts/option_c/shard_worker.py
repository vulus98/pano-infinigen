"""Per-shard pipeline: backup -> re-encode -> upload.

Each function is idempotent. Re-running on an already-completed shard is a no-op
(fast metadata check). Designed to be called from a Slurm array task that
processes a slice of the manifest.
"""
from __future__ import annotations

import io
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from .encoder import (
    decode_npy_or_npz,
    encode_depth_png,
    encode_depth_viz_png,
    encode_normals_jpg,
    transcode_image_to_jpg,
)

logger = logging.getLogger(__name__)


# ---------- step 1: backup -------------------------------------------------

def backup_shard(entry: dict, backup_root: Path, source_repo: str) -> Path:
    """Mirror one HF shard to `backup_root`. Idempotent (skips if size matches)."""
    dst = backup_root / entry["src_path"]
    if dst.exists() and dst.stat().st_size == entry["size"]:
        logger.info("[backup] skip %s (already %.1f MB)", entry["src_path"], entry["size"] / 1024**2)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    downloaded = hf_hub_download(
        repo_id=source_repo,
        repo_type="dataset",
        filename=entry["src_path"],
        local_dir=str(backup_root),
        local_dir_use_symlinks=False,
    )
    elapsed = time.time() - t
    actual = Path(downloaded).stat().st_size
    if entry["size"] and actual != entry["size"]:
        raise RuntimeError(
            f"size mismatch for {entry['src_path']}: expected {entry['size']}, got {actual}"
        )
    logger.info("[backup] %s  %.1f MB  %.1fs", entry["src_path"], actual / 1024**2, elapsed)
    return Path(downloaded)


# ---------- step 2: re-encode ---------------------------------------------

_IMG_STRUCT = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])


def _wrap(b: bytes) -> dict:
    return {"bytes": b, "path": None}


def _encode_row(args):
    """Multiprocessing worker: encode one row, return four byte-blobs.

    Layout (all HF `Image` features):
      image      -> JPEG q=95 (transcoded from source PNG, ~1.5 MB)
      depth      -> 16-bit single-channel PNG (per-config scale, ~1 MB)
      depth_viz  -> 8-bit Spectral RGB PNG preview (~0.3 MB)
      normals    -> 8-bit RGB JPEG q=95 of `(n+1)*127.5` (~2 MB on urban)

    All four are `Image` features so each cell becomes a ~200-byte URL in
    the HF /rows endpoint, making Data Studio responsive at any length up
    to 100. See ``RESEARCH_REPORT.md`` for the experiments that drove this
    schema choice.
    """
    img_bytes, depth_bytes, normals_bytes, depth_max_m = args
    depth_arr = decode_npy_or_npz(depth_bytes).astype("float32", copy=False)
    normals_arr = decode_npy_or_npz(normals_bytes).astype("float32", copy=False)
    return (
        transcode_image_to_jpg(img_bytes),
        encode_depth_png(depth_arr, depth_max_m),
        encode_depth_viz_png(depth_arr),
        encode_normals_jpg(normals_arr, quality=95),
    )


def reencode_shard(backup_file: Path, new_file: Path, depth_max_m: float, cpus: int) -> None:
    """Read backup parquet, encode all rows, write Option C parquet."""
    if new_file.exists() and new_file.stat().st_size > 0:
        # Re-verify schema; if all four expected columns are present, skip.
        try:
            schema = pq.read_schema(new_file)
            cols = set(schema.names)
            if {"image", "depth", "depth_viz", "normals"} <= cols and "normals_viz" not in cols:
                logger.info("[encode] skip %s (already encoded)", new_file.name)
                return
        except Exception:
            pass

    t = time.time()
    tbl = pq.read_table(str(backup_file))
    n = tbl.num_rows
    img_cells = tbl.column("image").to_pylist()
    depth_cells = tbl.column("depth").to_pylist()
    normals_cells = tbl.column("normals").to_pylist()

    inputs = []
    for i in range(n):
        img = img_cells[i]
        ib = img["bytes"] if isinstance(img, dict) else img
        inputs.append((ib, depth_cells[i], normals_cells[i], depth_max_m))

    images = [None] * n
    depths = [None] * n
    depth_vizs = [None] * n
    normals = [None] * n
    if cpus <= 1:
        for i, args in enumerate(inputs):
            ib, dp, dv, nm = _encode_row(args)
            images[i] = _wrap(ib)
            depths[i] = _wrap(dp)
            depth_vizs[i] = _wrap(dv)
            normals[i] = _wrap(nm)
    else:
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            futs = {pool.submit(_encode_row, args): idx for idx, args in enumerate(inputs)}
            for fut in as_completed(futs):
                idx = futs[fut]
                ib, dp, dv, nm = fut.result()
                images[idx] = _wrap(ib)
                depths[idx] = _wrap(dp)
                depth_vizs[idx] = _wrap(dv)
                normals[idx] = _wrap(nm)

    out_tbl = pa.table({
        "image": pa.array(images, type=_IMG_STRUCT),
        "depth": pa.array(depths, type=_IMG_STRUCT),
        "depth_viz": pa.array(depth_vizs, type=_IMG_STRUCT),
        "normals": pa.array(normals, type=_IMG_STRUCT),
    })
    feature_meta = {c: {"_type": "Image"} for c in ("image", "depth", "depth_viz", "normals")}
    out_tbl = out_tbl.replace_schema_metadata({
        b"huggingface": json.dumps({"info": {"features": feature_meta}}).encode()
    })

    new_file.parent.mkdir(parents=True, exist_ok=True)
    # Cap row groups at 4 rows so each stays well below the HF data-studio
    # 300 MB row-group scan limit (~20 MB/row * 4 = ~80 MB, plenty of headroom).
    pq.write_table(out_tbl, str(new_file), compression="snappy", row_group_size=4)
    elapsed = time.time() - t
    logger.info(
        "[encode] %s  %d rows  %.1f MB  %.1fs",
        new_file.name, n, new_file.stat().st_size / 1024**2, elapsed,
    )


# ---------- step 3: upload ------------------------------------------------

def upload_shards_batched(
    pending: list[tuple[Path, str]],
    target_repo: str,
    api: HfApi | None = None,
    skip_if_present: bool = True,
    commit_message: str | None = None,
) -> list[tuple[Path, str]]:
    """Upload all `(local_file, target_path)` pairs as a single HF commit.

    This avoids one-commit-per-shard, which triggers `JobManagerCrashedError`
    and "could not squash the history of the commits" failures on HF when
    many shards land in rapid succession across parallel Slurm tasks.

    Returns the list of pairs that were ACTUALLY uploaded (i.e. those that
    weren't skipped because the Hub already had matching size). The caller
    is responsible for cleaning up the local files of the returned pairs.
    """
    if not pending:
        return []
    api = api or HfApi()

    to_upload: list[tuple[Path, str]] = []
    if skip_if_present:
        try:
            info = api.repo_info(target_repo, repo_type="dataset", files_metadata=True)
            hub_sizes = {sib.rfilename: sib.size for sib in info.siblings}
        except Exception as e:
            logger.warning("[upload] could not fetch hub sizes: %s", e)
            hub_sizes = {}
        for local_file, target_path in pending:
            if hub_sizes.get(target_path) == local_file.stat().st_size:
                logger.info("[upload] skip %s (already on Hub with matching size)", target_path)
            else:
                to_upload.append((local_file, target_path))
    else:
        to_upload = list(pending)

    if not to_upload:
        return []

    ops = [
        CommitOperationAdd(path_in_repo=tp, path_or_fileobj=str(lf))
        for lf, tp in to_upload
    ]
    total_mb = sum(lf.stat().st_size for lf, _ in to_upload) / 1024**2
    t = time.time()
    msg = commit_message or (
        f"Option C re-encode: batch of {len(to_upload)} shard(s), {total_mb:.0f} MB"
    )
    api.create_commit(
        repo_id=target_repo,
        repo_type="dataset",
        operations=ops,
        commit_message=msg,
    )
    elapsed = time.time() - t
    logger.info(
        "[upload] batch of %d shards, %.0f MB total, %.1fs (%s)",
        len(to_upload), total_mb, elapsed, ", ".join(tp.split("/")[-1] for _, tp in to_upload),
    )
    return to_upload
