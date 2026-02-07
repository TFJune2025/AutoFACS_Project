# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import argparse
import logging
import wandb
import torch
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import Level 10 Modules
from src import modeling, vision_utils, inference_core, video_engine
from configs import scraper_engine, settings

# 1. Environment & Professional Logging Setup
load_dotenv("configs/.env")
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"autofacs_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("AutoFACS_Orchestrator")

# ... (All imports remain the same)

def run_orchestrator():
    parser = argparse.ArgumentParser(description="AutoFACS Level 10 Industry Pipeline")
    # FIX: Added 'meta' to choices so the script accepts the command
    parser.add_argument("--mode", type=str, required=True, choices=["scrape", "mine", "train", "meta"],
                        help="Execution mode: scrape, mine, train, or meta (Agentic)")
    parser.add_argument("--version", type=str, default="V41", help="Project version tag")
    args = parser.parse_args()

    # --- Step 1: Hardware Initialization ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_root = os.path.expanduser(os.getenv("MODEL_OUTPUT_PATH", "~/AutoFACS_Project/outputs"))
    logger.info(f"🚀 Initializing AutoFACS {args.version} on {device}")

    # --- Step 2: WandB Initialization ---
    wandb.init(
        project="AutoFACS_Project",
        name=f"{args.version}_{args.mode}_{datetime.now().strftime('%m%d_%H%M')}",
        config={
            "mode": args.mode, 
            "version": args.version, 
            "device": str(device),
            "conf_threshold": 0.85, 
            "entropy_threshold": 0.45,
            "prob_threshold": 0.95
        }
    )

    try:
        # NEW: Meta-Cognitive Level 10 Mode
        if args.mode == "meta":
            logger.info("🕹️ Entering Level 10 Meta-Cognitive Mode")
            # Defensive Import: Only loads LangGraph if meta mode is active
            from src.agent_architect import build_autofacs_brain
            
            brain = build_autofacs_brain()
            brain.invoke({
                "objective": "Scan Data Lake for AU4 intensity and save high-conviction samples.",
                "messages": [],
                "tools_available": ["video_engine", "inference_core"],
                "iteration_count": 0,
                "scratchpad": ""
            })

        elif args.mode == "scrape":
            logger.info("🌐 Mode: Unified Web Ingestion")
            # scraper_engine.run_unified_ingestion()

        # ... (mine and train modes remain exactly as you have them)

    except Exception as e:
        logger.critical(f"💥 Pipeline Crash: {str(e)}", exc_info=True)
        wandb.alert(title="Pipeline Crash", text=str(e))
    finally:
        wandb.finish()
        logger.info("🏁 Orchestration Session Closed.")

if __name__ == "__main__":
    run_orchestrator()