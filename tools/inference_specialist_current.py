# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import torch
import torchvision.transforms as T
from PIL import Image
import os
from master_orchestrator import safe_path_resolver
from src.architecture import AutoFACSNet # <--- The architectural bridge

# Automatically targeting the detected latest version
CURRENT_VERSION = "v1"
MODEL_PATH = safe_path_resolver(f"models/autofacs_{CURRENT_VERSION}.pth")

def run_inference(image_relative_path, face_box):
    """
    Full Inference Pipeline for v1.
    Handles: Path resolution, Cropping, Normalization, and Weight Loading.
    """
    # A. Resolve and Load Image through the Bouncer
    abs_path = safe_path_resolver(os.path.join("data_lake/AutoFACS_Project", image_relative_path))
    img = Image.open(abs_path).convert('RGB')
    
    # B. Precise Crop Logic (Top, Right, Bottom, Left)
    top, right, bottom, left = face_box
    face_crop = img.crop((left, top, right, bottom))
    
    # C. Environment-Locked Normalization (Matches your trained baseline)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(face_crop).unsqueeze(0)
    
    # D. Model Initialization and Weight Loading
    model = AutoFACSNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        output = model(tensor)
    
    return {
        "status": "SUCCESS", 
        "version": CURRENT_VERSION, 
        "target": image_relative_path,
        "raw_output": output.tolist()
    }
