# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import sys
from pathlib import Path
# 1. IMMEDIATE PATH INJECTION
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. THE REST OF IMPORTS
import torch
import re
import os

try:
    from configs.config import MODELS_DIR, DATA_LAKE_ROOT
    from src.architecture import AutoFACSNet
    print("SUCCESS: Project modules resolved.")
except ModuleNotFoundError as e:
    print(f"CRITICAL: Could not find modules. sys.path is currently: {sys.path}")
    raise e

def verify_weights():
    # 1. Hardware Check
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Targeting Device: {device}")

    # 2. Dynamic Model Discovery (Using Config-Driven Path)
    # We use MODELS_DIR from config.py - no need to manually walk and skip gdrive
    search_path = Path(MODELS_DIR)
    
    if not search_path.exists():
        print(f"FAIL: Models directory not found at {search_path}")
        print("Hint: Check if GDrive is mounted and path is correct.")
        return
    
    # Simple list comprehension - no os.walk needed because we've already 
    # bypassed the GDrive root in config.py
    v_dirs = [d for d in search_path.iterdir() if d.is_dir() and d.name.startswith("V")]

    if not v_dirs:
        print(f"FAIL: No versioned directories (V*) found in {search_path}")
        return
    
    # Natural sort to ensure V41 > V9
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    latest_v_dir = max(v_dirs, key=lambda d: natural_key(d.name))
    print(f"Scout Success: Targeting -> {latest_v_dir.name}")

    # 3. Recursive search for the .pth artifact
    weight_files = list(latest_v_dir.rglob("*.pth"))

    if not weight_files:
        print(f"FAIL: No .pth files found in {latest_v_dir.name}")
        return

    # Fix: Use weight_files (the variable actually defined)
    latest_model = max(weight_files, key=lambda f: os.path.getctime(f))
    print(f"Testing Alignment for: {latest_model.relative_to(project_root)}")

    # 4. Initialize Architecture
    try:
        model = AutoFACSNet() 
        model.to(device)
        
        # 4. Load State Dict
        state_dict = torch.load(latest_model, map_location=device)
        model.load_state_dict(state_dict)
        
        print("SUCCESS: Weights and Architecture are perfectly aligned.")
        
    except RuntimeError as e:
        print(f"CRITICAL ALIGNMENT ERROR: {e}")
        print("\nPossible Reason: Layer size mismatch or missing Action Unit definitions.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    verify_weights()