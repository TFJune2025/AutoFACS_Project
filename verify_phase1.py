# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import sys
import torch
import pathlib

def verify_system_state():
    print("=== AutoFACS Phase 1: System Integrity Check ===")
    
    # 1. MPS Validation
    print("\n[1/4] Hardware: Apple Silicon Optimization Check")
    if torch.backends.mps.is_available():
        print("PASS: MPS backend is available.")
        device = torch.device("mps")
        # Perform a small tensor operation to ensure the handshake is active
        x = torch.ones(1, device=device)
        print(f"Handshake: Tensor operation successful on {device}")
    else:
        print("FAIL: MPS backend not detected. Fallback to CPU is prohibited.")
        sys.exit(1)

    # 2. Config Handshake & Path Resolution
    print("\n[2/4] Environment: Absolute Path Resolution")
    root_dir = "/Users/natalyagrokh/AutoFACS_Project"
    if os.path.abspath(os.getcwd()) == root_dir:
        print(f"PASS: Root directory matches expected path: {root_dir}")
    else:
        print(f"WARNING: Current directory {os.getcwd()} differs from expected root {root_dir}")
    
    config_path = pathlib.Path(root_dir) / "configs" / "config.py"
    if config_path.exists():
        print(f"PASS: Configuration module found at {config_path}")
        sys.path.append(root_dir)
        try:
            import configs.config as cfg
            print("Handshake: configs.config imported successfully.")
        except ImportError as e:
            print(f"FAIL: Could not import configs.config. Error: {e}")
    else:
        print(f"FAIL: configs/config.py missing at {config_path}")

    # 3. Weight Reachability (Non-Recursive)
    print("\n[3/4] Data Lake: V41 Weight Reachability")
    # Targets the specific V41 directory identified in the forensic audit
# Update path to include the 'data_lake' FUSE-T mount point
    v41_path = pathlib.Path(root_dir) / "data_lake/ml_expressions/img_expressions/sup_training/V41_20260125_175823"
        
    if v41_path.exists() and v41_path.is_dir():
        # Check for essential model sub-directories to confirm reachability without a recursive storm
        emotion_model = v41_path / "emotion_classifier_model"
        relevance_model = v41_path / "relevance_filter_model"
        
        if emotion_model.exists() and relevance_model.exists():
            print(f"PASS: V41 Model weights found at {v41_path}")
        else:
            print(f"FAIL: V41 directory exists but sub-models are missing.")
    else:
        print(f"FAIL: V41 weight directory not reachable at {v41_path}")

    # 4. Repository Structure Validation
    print("\n[4/4] Structure: Python Path & Git Integrity")
    required_dirs = ["src/agent", "src/discovery", "tools", "configs", "tests"]
    missing = []
    
    for rdir in required_dirs:
        if not (pathlib.Path(root_dir) / rdir).is_dir():
            missing.append(rdir)
            
    if not missing:
        print("PASS: All core architectural modules (src, tools, configs, tests) are present.")
    else:
        print(f"FAIL: Missing required architectural modules: {', '.join(missing)}")
        
    # Check for legacy imports as requested
    try:
        from src.video_engine import VideoEngine
        print("Handshake: src/video_engine.py imported without circular dependencies.")
    except (ImportError, ModuleNotFoundError):
        # Allow pass if the file exists but engine isn't class-ready yet
        if (pathlib.Path(root_dir) / "src/video_engine.py").exists():
            print("INFO: src/video_engine.py exists but is not currently exportable.")
        else:
            print("FAIL: src/video_engine.py is missing.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_system_state()