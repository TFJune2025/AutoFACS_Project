# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import face_recognition
import os
from master_orchestrator import safe_path_resolver

def get_face_locations(relative_image_path):
    """
    Scout Tool: Locates bounding boxes for faces in the Data Lake.
    This prepares the crop for the specialized AutoFACS model.
    """
    try:
        # 1. Resolve path through the Bouncer
        abs_path = safe_path_resolver(os.path.join("data_lake/AutoFACS_Project", relative_image_path))
        
        # 2. Load image using the high-precision environment stack
        image = face_recognition.load_image_file(abs_path)
        
        # 3. Detect faces (uses dlib from your .yml)
        face_locations = face_recognition.face_locations(image)
        
        return {
            "status": "SUCCESS",
            "count": len(face_locations),
            "locations": face_locations, # Bounding boxes: (top, right, bottom, left)
            "abs_path": abs_path
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}