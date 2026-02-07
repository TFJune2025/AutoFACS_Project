# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os

# --- 📂 Core Project Paths ---
# We use absolute paths to ensure the brain can find the portal
BASE_DIR = os.path.expanduser("~/AutoFACS_Project")
GDRIVE_MOUNT = os.path.join(BASE_DIR, "gdrive")

# This points specifically to your 380GB ML data through the portal
BASE_DATASET_PATH = os.path.join(GDRIVE_MOUNT, "datasets")

# This is where your models and training logs will be saved locally
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "outputs")

# --- ⚙️ Run Configuration ---
RUN_INFERENCE = True
PREPARE_DATASETS = False
USE_EXTERNAL_CURATIONS = True

# --- 🏷️ Label Definitions (From V41/V42 Legacy) ---
RELEVANT_CLASSES = [
    'anger', 'contempt', 'disgust', 'fear', 'happiness',
    'neutral', 'questioning', 'sadness', 'surprise',
    'neutral_speech', 'speech_action'
]
IRRELEVANT_CLASSES = ['hard_case']

# --- 🤖 Model Configuration ---
BASE_MODEL_NAME = "google/vit-base-patch16-224-in21k"