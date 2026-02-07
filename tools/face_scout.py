# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import face_recognition
import os
import json
import re
import configs.config as config
from datetime import datetime

def scout_faces_in_datasets(relative_subfolder="datasets"):
    """
   Tactical Tool: Scans the Data Lake for faces.
    Utilizes the centralized config for path resolution.
    """

    # 1. Access the centralized data lake root
    # Note: config.DATA_LAKE_ROOT already points to the AutoFACS_Project folder
    target_path = os.path.join(config.DATA_LAKE_ROOT, relative_subfolder)
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "folder_scanned": relative_subfolder,
        "detections": []
    }
    
    if os.path.exists(target_path):
        # Shallow scan of the current synced state
        files = [f for f in os.listdir(target_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for file in files:
            img_path = os.path.join(target_path, file)
            image = face_recognition.load_image_file(img_path)
            locations = face_recognition.face_locations(image)
            
            if locations:
                manifest["detections"].append({
                    "filename": file,
                    "face_count": len(locations),
                    "coordinates": locations
                })
                
    return manifest