# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import subprocess
import json

def build_forensic_index():
    """Uses Rclone API to fetch hashes without reading raw bytes."""
    # Source path from your all_files.txt audit
    target_path = "gdrive:AutoAI_Projects/AutoFACS_Project"
    print(f"🚀 Discovery: Querying Cloud API for {target_path}")
    
    # lsjson fetches SHA1/MD5 hashes from Google's metadata directly
    cmd = ["rclone", "lsjson", target_path, "-R", "--hashes", "--files-only"]
    
    try:
        import subprocess
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        file_registry = json.loads(result.stdout)
        
        # Save manifest for master_orchestrator.py to analyze
        output_file = "lake_audit_manifest.json"
        with open(output_file, "w") as f:
            json.dump(file_registry, f, indent=4)
            
        print(f"✅ Audit complete. {len(file_registry)} artifacts indexed.")
    except Exception as e:
        print(f"❌ API Error: {str(e)}")

if __name__ == "__main__":
    build_forensic_index()