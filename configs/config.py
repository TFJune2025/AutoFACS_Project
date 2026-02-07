# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os

# --- PROJECT ROOT ---
PROJECT_ROOT = "/Users/natalyagrokh/AutoFACS_Project"

# --- CORE DIRECTORIES ---
GDRIVE_MOUNT = "/Users/natalyagrokh/AutoFACS_Project/data_lake" 
DATA_LAKE_ROOT = os.path.join(GDRIVE_MOUNT, "AutoAI_Projects/AutoFACS_Project/ml_expressions")
MODELS_DIR = DATA_LAKE_ROOT

# --- LOCAL DIRECTORIES ---
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# --- ENVIRONMENT SETTINGS ---
# Using the specific locks from your environment_automl.yml
MODEL_IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]