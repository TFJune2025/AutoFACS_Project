import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# CONFIGURATION
SOURCE_DIR = Path("./AutoFACS_Freeze_Copy")
LAKE_DIR = Path("./data_lake")
CODE_DIR = Path("./codebase")
DRY_RUN = True  # Set to False to execute moves

# REFINED MAPPING (Incorporating Inference + Flywheel)
SCHEMA_MAP = {
    "00_raw_data": ["raw_data", "videos", "clips", "input_images_master"],
    "01_curated_data": ["curated", "labeled", "standardized", "flywheel_output"],
    "02_feature_store": ["landmarks", "embeddings", "feature", "tensors"],
    "03_experiments": ["training_runs", "exp_", "wandb", "flywheel_exp"],
    "04_artifacts": ["checkpoint-", "model.pt", "model.onnx", "weights"],
    "05_evaluation": ["inference_output", "benchmarks", "reports", "matrices"],
    "99_archive": ["pexels", "scraper_v1", "original_drive_dump", "graveyard"],
    
    # Codebase mapping
    "codebase/src/agent": ["agent_", "orchestrator", "master_", "boot.py"],
    "codebase/src/flywheel": ["flywheel_logic", "labeling_bridge", "active_tool"],
    "codebase/notebooks": [".ipynb"]
}

def create_manifest(version_name="v41_snapshot"):
    """Generates a cryptographic pointer for the current state."""
    manifest = {
        "version": version_name,
        "timestamp": datetime.now().isoformat(),
        "raw_data": str(LAKE_DIR / "00_raw_data"),
        "curated_data": str(LAKE_DIR / "01_curated_data"),
        "model_artifacts": str(LAKE_DIR / "04_artifacts"),
        "git_commit": "HEAD",  # Placeholder for git_rev_parse
    }
    
    manifest_path = LAKE_DIR / "_manifests" / "datasets" / f"{version_name}.json"
    if not DRY_RUN:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Created pointer at: {manifest_path}")

def organize_and_manifest():
    LAKE_DIR.mkdir(exist_ok=True)
    CODE_DIR.mkdir(exist_ok=True)

    for file_path in SOURCE_DIR.rglob("*"):
        if file_path.is_dir() or ".git" in str(file_path):
            continue

        target_path = "99_archive"
        filename_lower = file_path.name.lower()
        
        for folder, identifiers in SCHEMA_MAP.items():
            if any(id_str in filename_lower for id_str in identifiers):
                target_path = folder
                break
        
        dest_dir = Path("./") / target_path if "codebase" in target_path else LAKE_DIR / target_path
        dest_file = dest_dir / file_path.name

        if DRY_RUN:
            print(f"[PLAN] {file_path.name} -> {dest_dir}/")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest_file))

    create_manifest()

if __name__ == "__main__":
    print(f"Starting migration to Hybrid Numbered Lake...")
    organize_and_manifest()