# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import json
import importlib
from master_orchestrator import safe_path_resolver, get_latest_model_version
from tools.face_scout import scout_faces_in_datasets

def run_dry_test():
    print("--- 🧪 LEVEL 10 INFERENCE DRY RUN ---")
    
    # 1. Verify Discovery
    version = get_latest_model_version()
    print(f"[*] Discovered Version: {version}")
    
    # 2. Scouting for synced data
    print("[*] Scouting Data Lake for a test subject...")
    scout_results = scout_faces_in_datasets("datasets")
    
    if not scout_results['detections']:
        print("❌ FAIL: No synced images with faces found. Transfer more data to test.")
        return

    # Grab the first available image and face
    target = scout_results['detections'][0]
    filename = target['filename']
    face_box = target['coordinates'][0]
    print(f"[*] Target Acquired: {filename} (Face at {face_box})")

    # 3. Dynamic Inference Loading
    try:
        # Import the tool synthesized by the orchestrator
        spec = importlib.util.spec_from_file_location(
            "inference_specialist_current", 
            safe_path_resolver("tools/inference_specialist_current.py")
        )
        inference_tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_tool)
        
        # 4. Execute Inference
        print(f"[*] Running {version} Specialist Analysis...")
        result = inference_tool.run_inference(f"datasets/{filename}", face_box)
        
        print("\n--- ✅ TEST COMPLETE ---")
        print(json.dumps(result, indent=4))

    except Exception as e:
        print(f"❌ INFERENCE FAILED: {e}")
        print("Tip: Ensure your src/architecture.py matches your .pth weights.")

if __name__ == "__main__":
    run_dry_test()