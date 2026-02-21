import os
import random
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as HfImage, Value
from huggingface_hub import login

# --- Configuration ---
BASE_DIR = "outputs"
REPO_ID = "prs-eth/Pano-Infinigen"  # Ensure this matches your target repo
HF_TOKEN = os.environ.get("HF_TOKEN") # Best practice: Use an environment variable

def get_id1_splits(scene_dir, train_pct=0.8, val_pct=0.1):
    """Gathers all id_1 folders and rigidly splits them into train/val/test."""
    if not scene_dir.exists():
        return [], [], []
        
    id1_paths = [d for d in scene_dir.iterdir() if d.is_dir()]
    id1_names = [d.name for d in id1_paths]
    
    # Sort and seed for exact reproducibility across runs
    id1_names.sort() 
    random.seed(42)
    random.shuffle(id1_names)
    
    n_total = len(id1_names)
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)
    
    train_ids = id1_names[:n_train]
    val_ids = id1_names[n_train:n_train + n_val]
    test_ids = id1_names[n_train + n_val:]
    
    return train_ids, val_ids, test_ids

def generate_examples(base_dir, target_scene, allowed_id1s):
    """Yields valid triplets ONLY if their id_1 is in the allowed list."""
    allowed_set = set(allowed_id1s) 
    
    scene_dir = Path(base_dir) / target_scene
    if not scene_dir.exists():
        return

    for id1_dir in scene_dir.iterdir():
        if not id1_dir.is_dir() or id1_dir.name not in allowed_set: 
            continue # Skip if this id_1 belongs to a different split
            
        for id2_dir in id1_dir.iterdir():
            if not id2_dir.is_dir(): continue
            
            camera_dir = id2_dir / "frames"
            img_dir = camera_dir / "Image" / "camera_0"
            depth_dir = camera_dir / "Depth" / "camera_0"
            normal_dir = camera_dir / "SurfaceNormal" / "camera_0"
            
            if not img_dir.exists(): continue
            
            for img_path in img_dir.glob("Image_*.png"):
                suffix = img_path.stem.replace("Image_", "")
                
                depth_path = depth_dir / f"Depth_{suffix}.npy"
                normal_path = normal_dir / f"SurfaceNormal_{suffix}.npy"
                
                if depth_path.exists() and normal_path.exists():
                    try:
                        with open(depth_path, "rb") as f_depth: depth_bytes = f_depth.read()
                        with open(normal_path, "rb") as f_normal: normal_bytes = f_normal.read()

                        # CLEANED OUTPUT: No IDs or suffixes included
                        yield {
                            "image": Image.open(img_path).convert("RGB"),
                            "depth": depth_bytes,
                            "normals": normal_bytes,
                        }
                    except Exception:
                        pass # Silently skip corrupted files

if __name__ == "__main__":
    # Log in using the token from your environment or replace with a fresh token string
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: HF_TOKEN not found in environment.")

    # CLEANED SCHEMA: Define only the columns you want to see on the Hub
    features = Features({
        "image": HfImage(),
        "depth": Value("binary"),
        "normals": Value("binary"),
    })

    # Loop through both configurations
    for scene_type in ["indoor", "outdoor"]:
        print(f"\n--- Processing Config: {scene_type.upper()} ---")
        scene_dir = Path(BASE_DIR) / scene_type
        
        # 1. Calculate the id_1 groupings first
        train_ids, val_ids, test_ids = get_id1_splits(scene_dir)
        
        split_mappings = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }
        
        dataset_splits = {}
        
        # 2. Build a specific Dataset for each split
        for split_name, allowed_ids in split_mappings.items():
            if not allowed_ids:
                print(f"Skipping {split_name} split (not enough id_1s).")
                continue
                
            print(f"Building {split_name} split ({len(allowed_ids)} unique id_1 locations)...")
            
            split_dataset = Dataset.from_generator(
                generate_examples,
                gen_kwargs={
                    "base_dir": BASE_DIR, 
                    "target_scene": scene_type,
                    "allowed_id1s": allowed_ids
                },
                features=features
            )
            
            if len(split_dataset) > 0:
                dataset_splits[split_name] = split_dataset
                print(f"  -> {len(split_dataset)} total frames added.")

        # 3. Combine into DatasetDict and push
        if dataset_splits:
            dataset_dict = DatasetDict(dataset_splits)
            print(f"Pushing '{scene_type}' config to HF Hub...")
            dataset_dict.push_to_hub(REPO_ID, config_name=scene_type)
        else:
            print(f"No valid data found to push for {scene_type}.")

    print("\nAll done! You have a clean, visual-only dataset on HF.")