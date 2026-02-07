# ==============================================================================
# Copyright (c) 2026 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential.
# ==============================================================================

import cv2
import face_recognition
import os
from PIL import Image
from tqdm import tqdm
from src.inference_core import predict_emotions

def process_video_frames(video_path, save_root, model, processor, device, frame_skip=1):
    """Extracts faces from video and routes them to the Inference Core."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logs = []
    pbar = tqdm(total=total_frames, desc="🎞️ Processing Video")

    while cap.isOpened():
        ret, frame = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret: break

        if frame_id % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Crop and convert to PIL 
                face_arr = frame[top:bottom, left:right]
                face_pil = Image.fromarray(cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB))
                
                # Call the Unified Brain
                emotion_data = predict_emotions(face_pil, model, processor, device)
                
                # Save data for the manifest 
                logs.append({
                    "timestamp": frame_id / fps,
                    "frame": frame_id,
                    **emotion_data
                })
        
        pbar.update(1)
    
    cap.release()
    return logs