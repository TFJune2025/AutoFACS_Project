# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config


def rclone_scout(remote_name, target_path):
    """Bypasses the FUSE-T mount to scan GDrive via API."""
    print(f"--- [RCLONE HEADLESS SCOUT: {target_path}] ---")
    
    # We use lsjson for machine-readable, high-speed metadata
    # --fast-list reduces the number of API calls significantly
    cmd = [
        "rclone", "lsjson", "-R", 
        "--fast-list", 
        "--max-depth", "5",
        f"{remote_name}:{target_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Rclone scan failed. {result.stderr}")
        return []
        
    return json.loads(result.stdout)

class ProposalEngine:
    """The Logic the agent uses to decide the Golden Schema."""
    
    @staticmethod
    def recommend(inventory):
        print("\n" + "!"*50)
        print("AGENTIC DECISION: REPOSITORY RE-ARCHITECTING")
        print("!"*50)
        
        # Mapping the deep-found models to the new schema
        if inventory["MODELS"]["locations"]:
            print(f"\n[MODEL MIGRATION PLAN]")
            print(f"Detected {len(inventory['MODELS']['locations'])} versioned training sets buried deeply.")
            for loc in inventory["MODELS"]["locations"][:3]: # Show top 3 examples
                print(f"  - SOURCE: .../{loc}")
            print(f"  - TARGET: /AutoFACS_Project/models/active/")
            print("  - RATIONALE: Reducing directory depth from 7 to 2 to improve FUSE-T latency.")

        print(f"\n[STORAGE POLICY]")
        print(f"-> PROTECT: Data Lake ({inventory['DATASET_RAW']['size_gb']:.2f} GB) is now isolated.")

def parse_all_files_txt(file_path):
    """Parses the all_files.txt inventory into a structured format for the agent."""
    parsed_data = []
    if not os.path.exists(file_path):
        print(f"ERROR: Inventory file {file_path} not found.")
        return parsed_data

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                try:
                    size = int(parts[0])
                    path_str = parts[1]
                    # Extract the filename/directory name for the 'Name' key
                    name = os.path.basename(path_str)
                    
                    parsed_data.append({
                        "Path": path_str,
                        "Name": name, # Required for model version detection
                        "Size": size,
                        "IsDir": False # all_files.txt contains file entries
                    })
                except ValueError:
                    continue
    return parsed_data

def categorize_rclone_inventory(rclone_data):
    """Parses rclone JSON into the Agent's inventory schema."""
    summary = {
        "MODELS": {"count": 0, "locations": []},
        "DATASET_RAW": {"size_gb": 0.0}
    }
    
    total_size_bytes = 0
    for item in rclone_data:
        path_str = item.get("Path", "")
        # Identify Models
        if "sup_training" in path_str and item.get("IsDir") and item.get("Name", "").startswith("V"):
            summary["MODELS"]["count"] += 1
            summary["MODELS"]["locations"].append(path_str)
        
        # Accumulate Total Size
        total_size_bytes += item.get("Size", 0)

    summary["DATASET_RAW"]["size_gb"] = total_size_bytes / (1024**3)
    return summary

def audit_entire_system():
    # Primary Source of Truth: all_files.txt in project root
    inventory_file = Path(config.PROJECT_ROOT) / "all_files.txt"
    
    print(f"--- [INVENTORY AUDIT: {inventory_file.name}] ---")
    raw_data = parse_all_files_txt(inventory_file)
    
    if raw_data:
        # Pass the parsed inventory to the ProposalEngine
        inventory_summary = categorize_rclone_inventory(raw_data)
        ProposalEngine.recommend(inventory_summary)
    else:
        print("FAIL: No data found in inventory file.")

if __name__ == "__main__":
    # Removed the AI_ROOT argument to match the new zero-parameter signature
    audit_entire_system()