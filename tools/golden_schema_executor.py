import os
import subprocess
import sys
import re

# [AUTO-FACS LEVEL 10 EXECUTOR]
# PROTOCOL: API-Direct Only. NO local file system calls on /data_lake.

def load_map():
    """Level 2: Ingest the Static Map (Robust Parsing)."""
    map_path = "all_files.txt"
    if not os.path.exists(map_path):
        print(f"❌ CRITICAL: Map {map_path} not found.")
        sys.exit(1)
    
    print(f"📖 Ingesting {map_path}...")
    clean_paths = []
    with open(map_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Forensic Fix: Map contains "Size Path". Split on first space.
            # Example: "343297654 FACS_Automation/..." -> ["343297654", "FACS_Automation/..."]
            parts = line.split(maxsplit=1)
            
            # If line has at least 2 parts (Size, Path), take the path.
            # If only 1 part (unexpected), take it as is.
            if len(parts) > 1:
                clean_paths.append(parts[1])
            else:
                clean_paths.append(parts[0])
                
    return clean_paths

def generate_spec_compliant_plan(file_list):
    """
    Logic:
    1. FILTER: Ignore 'Research_Archive' (Strict).
    2. FIX: Prepend 'AutoAI_Projects/' if missing.
    3. SORT: Move Weights to model_weights/V{num}/
    """
    moves = []
    
    # FSD Destinations
    BASE_WEIGHTS = "model_weights"
    
    # Regex
    p_version = re.compile(r"(V\d+)", re.IGNORECASE)
    p_weights = re.compile(r".*\.(pth|safetensors|pt|ckpt)$", re.IGNORECASE)

    print(f"🧠 Sorting items into FSD Structure: {BASE_WEIGHTS}/V{{N}}/")

    for f in file_list:
        # --- SECURITY GATE ---
        if "Research_Archive" in f:
            continue # 🛑 OFF LIMITS
            
        # --- PATH CORRECTION ---
        # If the path starts with FACS_Automation, it is likely inside AutoAI_Projects
        # We need the source path to match Google Drive's reality
        src_path = f
        if f.startswith("FACS_Automation/"):
            src_path = f"AutoAI_Projects/{f}"
        
        # Cleanup for logic check (remove prefixes to check filename)
        # We process the file based on what it IS, not where it IS.
        
        # Skip files already in the destination
        if f.startswith(BASE_WEIGHTS) or f.startswith(f"AutoAI_Projects/{BASE_WEIGHTS}"):
            continue

        # --- SORTING LOGIC ---
        if p_weights.match(f):
            match = p_version.search(f)
            if match:
                v_tag = match.group(1).upper()
                dest = f"{BASE_WEIGHTS}/{v_tag}"
                moves.append((src_path, dest))
            else:
                moves.append((src_path, f"{BASE_WEIGHTS}/Unsorted"))

    return moves

def get_remote_name():
    """Level 5: Identify the Rclone Remote dynamically."""
    result = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True)
    remotes = result.stdout.strip().split('\n')
    if not remotes:
        print("❌ CRITICAL: No rclone remotes found.")
        sys.exit(1)
    # Default to first remote found (usually gdrive:)
    return remotes[0].strip()

def purge_artifacts(file_list, remote_name):
    """Level 10: Server-Side Purge via API."""
    print("⚔️  Phase 1: Analyzing Artifacts (.DS_Store)...")
    
    # Filter strictly from the text file (0ms latency)
    targets = [f for f in file_list if f.endswith('.DS_Store')]
    
    if not targets:
        print("✅ No artifacts found in the Map.")
        return

    print(f"⚠️  Found {len(targets)} artifacts to purge.")
    print("🚀 Initiating API-Direct Purge (Bypassing FUSE mount)...")

    # Write targets to a temporary list for rclone to consume
    # We must strip the local mount prefix if present in the map
    # Assuming map has paths like "/data_lake/folder/file" or relative paths
    
    clean_targets = []
    for t in targets:
        # Remove local mount prefix if it exists in the map
        clean_path = t.replace('/data_lake/', '').replace('data_lake/', '')
        clean_targets.append(clean_path)

    with open("purge_list.txt", "w") as f:
        for t in clean_targets:
            f.write(t + "\n")

    # Execute Rclone Delete (Server-Side)
    # --files-from allows batch deletion without shell arguments limit
    cmd = [
        "rclone", "delete", 
        remote_name, 
        "--files-from", "purge_list.txt", 
        "-v", "--dry-run" # SAFETY FIRST: Dry run enabled
    ]
    
    subprocess.run(cmd)
    print("\nℹ️  This was a DRY RUN. To execute, edit this script and remove '--dry-run'.")
    os.remove("purge_list.txt")

def execute_batch(moves, remote_name):
    """Generates the Level 10 Shell Script."""
    if not moves:
        print("✅ Compliance Achieved: No files need moving.")
        return

    print(f"🚀 Detected {len(moves)} violations of Golden Schema.")
    print("📝 Generating API Remediation Script (server_side_organize.sh)...")
    
    with open("server_side_organize.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# AutoFACS Level 10: FSD Enforcement Script\n")
        f.write("# Generated by golden_schema_executor.py\n\n")
        
        for src, dest in moves:
            # Use --dry-run for safety first
            # Quote paths to handle spaces
            cmd = f'rclone move "{remote_name}{src}" "{remote_name}{dest}" -v --dry-run'
            f.write(cmd + "\n")
            
    print("⚠️  Script generated. Run 'bash server_side_organize.sh' to preview.")

def main():
    print("🚀 AutoFACS Level 10: Golden Schema Executor v3.0")
    print("-------------------------------------------------")
    
    # 1. Acquire Resources
    remote = get_remote_name()
    files = load_map()
    print(f"🔌 Connected to: {remote}")
    print(f"🧠 Memory Loaded: {len(files)} items.")

    # 2. Phase 1: Sanitation (The Janitor)
    print("\n[PHASE 1] Artifact Purge")
    purge_artifacts(files, remote)

    # 3. Phase 2: Organization (The Librarian)
    print("\n[PHASE 2] Golden Schema Enforcement")
    plan = generate_spec_compliant_plan(files)
    execute_batch(plan, remote)
    
    print("\n✅ Execution Cycle Complete.")

if __name__ == "__main__":
    main()