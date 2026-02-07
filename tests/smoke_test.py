# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import configs.config as config
import re # Added for version parsing
from master_orchestrator import safe_path_resolver, get_latest_model_version

def run_smoke_test():
    print("--- 💨 RUNNING LEVEL 10 SMOKE TEST ---")
    
    # Test 1: Valid Access (Unchanged)
    try:
        valid_path = safe_path_resolver("data_lake/AutoFACS_Project")
        print(f"✅ TEST 1 PASSED: Valid access to {valid_path}")
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")

    # Test 2: Jailbreak Attempt (The "Bouncer" Check)
    try:
        print("\nAttempting to escape to research_archive...")
        invalid_path = safe_path_resolver("../research_archive")
        print(f"❌ TEST 2 FAILED: The agent escaped the jail to {invalid_path}!")
    except PermissionError as e:
        print(f"✅ TEST 2 PASSED: Bouncer blocked access. Error: {e}")
    except Exception as e:
        print(f"⚠️ TEST 2 WARNING: Unexpected error type: {type(e).__name__}: {e}")

    # Test 3: Dynamic Version Discovery (The "Vision" Check)
    print("\nVerifying Dynamic Model Discovery...")
    try:
        current_v = get_latest_model_version() # Now imported from orchestrator
        model_filename = f"autofacs_{current_v}.pth"
        model_path = safe_path_resolver(f"models/{model_filename}")
        
        if os.path.exists(model_path):
            print(f"✅ TEST 3 PASSED: Discovered and verified {current_v} at {model_path}")
        else:
            print(f"⚠️ TEST 3 WARNING: Found version {current_v}, but {model_filename} is missing/not synced.")
    except Exception as e:
        print(f"❌ TEST 3 FAILED: Version discovery crashed. Error: {e}")

if __name__ == "__main__":
    run_smoke_test()