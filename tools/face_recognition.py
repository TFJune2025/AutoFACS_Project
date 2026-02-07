# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import face_recognition
import cv2
import os

def detect_faces_in_v41(relative_image_path):
    """
    Level 10 Tool: Detects human faces in the Data Lake.
    Uses the AutoML environment vision stack.
    """
    from master_orchestrator import safe_path_resolver
    
    # Resolve the path through the security gate
    abs_path = safe_path_resolver(os.path.join("data_lake/AutoFACS_Project", relative_image_path))
    
    # Load and process the image
    image = face_recognition.load_image_file(abs_path)
    face_locations = face_recognition.face_locations(image)
    
    return {
        "status": "SUCCESS",
        "faces_found": len(face_locations),
        "locations": face_locations
    }