# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Phase1_Handshake")

def check_mps():
    """Verify Apple Silicon GPU Acceleration."""
    if torch.backends.mps.is_available():
        logger.info("Hardware: Apple Silicon (MPS) detected and available.")
        return torch.device("mps")
    else:
        logger.warning("Hardware: MPS not found. Falling back to CPU.")
        return torch.device("cpu")

def verify_components():
    # Ensure project root is in path for relative imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    device = check_mps()
    results = {"VideoEngine": "Fail", "FaceScout": "Fail", "DataLake": "Fail"}

# 1. Verify Data Lake (Updated path to match your environment)
data_lake_path = project_root / "data_lake"
if data_lake_path.exists():
    logger.info(f"Scouting: Data Lake found at {data_lake_path}")
    results["DataLake"] = "Pass"
else:
    logger.error(f"Scouting: Missing 'data_lake' directory.")

    # 1. Verify Data Lake Mockup
    data_lake_path = project_root / "data/raw"
    if data_lake_path.exists():
        logger.info(f"Scouting: Data Lake found at {data_lake_path}")
        results["DataLake"] = "Pass"
    else:
        logger.error("Scouting: Missing 'data/raw' directory.")

    # 2. Verify Video Engine
    try:
        from src.video_engine import VideoEngine
        # Initialization test
        engine = VideoEngine()
        logger.info("VideoEngine: Class initialized successfully.")
        results["VideoEngine"] = "Pass"
    except ImportError:
        logger.error("VideoEngine: Could not import 'VideoEngine' from src.video_engine.")
    except Exception as e:
        logger.error(f"VideoEngine: Initialization failed: {e}")

    # 3. Verify Face Scout
    try:
        # Checking for the scouting function or class
        import tools.face_scout as face_scout
        if hasattr(face_scout, 'FaceScout'):
            scout = face_scout.FaceScout()
            logger.info("FaceScout: Class initialized successfully.")
        elif hasattr(face_scout, 'scout_faces_in_datasets'):
            logger.info("FaceScout: Function 'scout_faces_in_datasets' found.")
        
        results["FaceScout"] = "Pass"
    except ImportError:
        logger.error("FaceScout: Could not import 'tools.face_scout'.")
    except Exception as e:
        logger.error(f"FaceScout: Initialization failed: {e}")

    # Final Report
    print("\n" + "="*30)
    print(" PHASE 1 INTEGRITY REPORT")
    print("="*30)
    for component, status in results.items():
        print(f"{component: <15}: {status}")
    print("="*30)

if __name__ == "__main__":
    verify_components()