"""Entry point for Slurm array tasks.

Each task receives `--task-index N` and `--num-tasks M`, takes the slice
``manifest[N::M]`` and runs the selected phase(s) on every shard in that slice.

Examples:

    # Phase 1: backup all shards (Slurm task k of n)
    python -m scripts.option_c.process_batch \\
        --manifest /cluster/scratch/vbozic/panoinfinigen_manifest.json \\
        --task-index $SLURM_ARRAY_TASK_ID --num-tasks $NUM_TASKS \\
        --backup-dir /cluster/scratch/vbozic/panoinfinigen_backup \\
        --do backup

    # Phase 2: re-encode + upload to production
    python -m scripts.option_c.process_batch \\
        --manifest /cluster/scratch/vbozic/panoinfinigen_manifest.json \\
        --task-index $SLURM_ARRAY_TASK_ID --num-tasks $NUM_TASKS \\
        --backup-dir /cluster/scratch/vbozic/panoinfinigen_backup \\
        --new-dir    /cluster/scratch/vbozic/panoinfinigen_new \\
        --target-repo prs-eth/PanoInfinigen --target-prefix "data/" \\
        --cpus 8 --do reencode --do upload

    # Small-scale validation: only urban/val, upload to a test repo
    python -m scripts.option_c.process_batch \\
        --manifest .../manifest.json --task-index 0 --num-tasks 1 \\
        --config urban --split val \\
        --backup-dir .../backup --new-dir .../new \\
        --target-repo vulus98/panoinfinigen-option-c-validation \\
        --target-prefix "data/urban-val/" \\
        --cpus 8 --do backup --do reencode --do upload
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from . import SOURCE_REPO
from .manifest import filter_manifest, slice_for_task
from .shard_worker import backup_shard, reencode_shard, upload_shards_batched


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Per-Slurm-array-task pipeline driver")
    ap.add_argument("--manifest", required=True, help="path to manifest.json")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--config", choices=["indoor", "nature", "urban"], default=None,
                    help="optional: process only this config")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="optional: process only this split")
    ap.add_argument("--backup-dir", required=True, help="local mirror of source shards")
    ap.add_argument("--new-dir", default=None, help="local dir for re-encoded shards (required for reencode/upload)")
    ap.add_argument("--source-repo", default=SOURCE_REPO)
    ap.add_argument("--target-repo", default=SOURCE_REPO)
    ap.add_argument("--target-prefix", default="data/",
                    help="HF path prefix; will be joined with '{config}/{shard_name}'")
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--do", action="append", choices=["backup", "reencode", "upload"], default=None,
                    help="phases to run (repeatable). Default: all three.")
    ap.add_argument("--keep-new-after-upload", action="store_true",
                    help="don't delete the local re-encoded shard after a successful upload")
    ap.add_argument("--upload-batch-size", type=int, default=10,
                    help="how many shards to bundle into a single HF commit (1-100). "
                         "Smaller -> more commits but lower scratch peak; larger -> fewer "
                         "commits but more disk needed before each commit.")
    return ap.parse_args()


def main():
    setup_logging()
    args = parse_args()

    phases = set(args.do or ["backup", "reencode", "upload"])
    if ("reencode" in phases or "upload" in phases) and not args.new_dir:
        sys.exit("--new-dir is required for reencode/upload phases")

    # Don't call huggingface_hub.login() here: it writes to
    # ~/.cache/huggingface/stored_tokens, which races between parallel
    # Slurm array tasks. HfApi() + hf_hub_download() both read HF_TOKEN
    # from the environment, so explicit login is unnecessary.
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    manifest = json.loads(Path(args.manifest).read_text())
    manifest = filter_manifest(manifest, args.config, args.split)
    slice_ = slice_for_task(manifest, args.task_index, args.num_tasks)
    logging.info(
        "task %d/%d  -> %d shards (cpus=%d, phases=%s)",
        args.task_index, args.num_tasks, len(slice_), args.cpus, sorted(phases),
    )

    backup_root = Path(args.backup_dir)
    new_root = Path(args.new_dir) if args.new_dir else None
    n_ok, n_err = 0, 0
    pending: list[tuple[Path, str]] = []  # for batched upload

    def flush_pending():
        if not pending:
            return
        uploaded = upload_shards_batched(pending, args.target_repo, api=api)
        if not args.keep_new_after_upload:
            for local_file, _ in uploaded:
                local_file.unlink(missing_ok=True)
        pending.clear()

    for entry in slice_:
        tag = f"{entry['config']}/{entry['split']}/{entry['shard_name']}"
        try:
            backup_file = backup_root / entry["src_path"]
            if "backup" in phases:
                backup_shard(entry, backup_root, args.source_repo)

            if "reencode" in phases:
                new_file = new_root / entry["config"] / entry["shard_name"]
                reencode_shard(backup_file, new_file, entry["depth_max_m"], args.cpus)

            if "upload" in phases:
                new_file = new_root / entry["config"] / entry["shard_name"]
                if not new_file.exists():
                    raise RuntimeError(f"missing re-encoded shard: {new_file}")
                target_path = f"{args.target_prefix.rstrip('/')}/{entry['config']}/{entry['shard_name']}"
                pending.append((new_file, target_path))
                if len(pending) >= max(1, args.upload_batch_size):
                    flush_pending()

            n_ok += 1
            logging.info("[OK] %s", tag)
        except Exception as e:
            n_err += 1
            logging.exception("[FAIL] %s : %s", tag, e)

    # Final flush for the trailing partial batch.
    if "upload" in phases:
        try:
            flush_pending()
        except Exception as e:
            n_err += 1
            logging.exception("[FAIL] final batch upload: %s", e)

    logging.info("done: %d ok, %d errored", n_ok, n_err)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
