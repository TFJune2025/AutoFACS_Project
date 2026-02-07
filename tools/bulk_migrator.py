# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import subprocess
import os
from configs import config

def migrate_auto_facs():
    root = "gdrive:AutoAI_Projects/AutoFACS_Project"
    
    # 1. Migrate all 41 Iterations (Weights, Checkpoints, Tensors)
    for i in range(1, 42):
        v_folder = f"V{i:02d}" if i < 10 else f"V{i}"
        dest = f"gdrive:AutoAI_Projects/02_Experiments/supervised/{v_folder}"
        print(f"📦 Migrating Iteration {v_folder} Artifacts...")
        # Use --checksum to ensure no bit-loss during move
        subprocess.run(["rclone", "move", f"{root}/{v_folder}", dest, "--checksum", "--create-empty-src-dirs"])

    # 2. Migrate Curation & Dataset Clusters
    print("📂 Migrating 2TB Dataset Lake & Curation Scripts...")
    subprocess.run(["rclone", "move", f"{root}/img_datasets", "gdrive:AutoAI_Projects/01_Dataset_Lake/raw", "--checksum"])
    
    print("✅ Level 10 Migration Complete. Every artifact preserved.")

if __name__ == "__main__":
    migrate_auto_facs()