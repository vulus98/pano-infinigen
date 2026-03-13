import os
import random
import argparse
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Image as HfImage, Value
from huggingface_hub import login, HfApi, list_repo_files

# --- Configuration ---
BASE_DIR = "outputs"
REPO_ID = "prs-eth/PanoInfinigen"
HF_TOKEN = os.environ.get("HF_TOKEN")
CHUNK_SIZE = 15 

CONFIG_MAPPING = {
    "indoor": "indoor",
    "outdoor": "nature"
}

def get_args():
    parser = argparse.ArgumentParser(description="Resumable HF Upload")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    return parser.parse_args()

# ... (get_id1_splits, get_paths_for_ids, process_and_load remain the same) ...

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

def get_paths_for_ids(base_dir, target_scene, ids):
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

def process_and_load(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    depths = [Path(p).read_bytes() for p in batch["depth_path"]]
    normals = [Path(p).read_bytes() for p in batch["normals_path"]]
    return {"image": images, "depth": depths, "normals": normals}

if __name__ == "__main__":
    args = get_args()
    api = HfApi()
    if HF_TOKEN: login(token=HF_TOKEN)
    
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
                parquet_filename = f"{split_name}-{chunk_idx:05d}.parquet"
                path_in_repo = f"data/{config_label}/{parquet_filename}"

                # 2. SKIP LOGIC: If file exists, don't even scan the disk
                if path_in_repo in existing_files:
                    print(f"Skipping {path_in_repo} (Already on Hub)")
                    continue

                print(f"\n[{hf_config}][{split_name}] Processing Chunk {chunk_idx + 1}...")
                chunk_ids = all_ids[i : i + CHUNK_SIZE]
                paths = get_paths_for_ids(BASE_DIR, local_folder, chunk_ids)
                if not paths: continue

                path_ds = Dataset.from_list(paths)
                processed_ds = path_ds.map(
                    process_and_load,
                    batched=True,
                    batch_size=12,
                    num_proc=args.cpus,
                    remove_columns=["image_path", "depth_path", "normals_path"],
                    features=features,
                    keep_in_memory=False
                )

                local_parquet_path = scratch_dir / parquet_filename
                processed_ds.to_parquet(str(local_parquet_path))

                print(f"[{hf_config}] Uploading {parquet_filename}...")
                api.upload_file(
                    path_or_fileobj=str(local_parquet_path),
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                )

                # Cleanup
                if local_parquet_path.exists(): os.remove(local_parquet_path)
                processed_ds.cleanup_cache_files()
                for p in scratch_dir.glob("*.arrow"):
                    try: os.remove(p)
                    except: pass

    print("\nUpload sequence resumed and completed.")