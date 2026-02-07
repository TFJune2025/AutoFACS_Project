#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import mediapipe as mp
import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from tqdm import tqdm
import random


# In[2]:


# --- CONFIGURATION ---
# All user-configurable settings are in this class for easy management.
class Config:
    # 1. Your Google Cloud Project Details
    PROJECT_ID = "MLexpressImgSorting"
    LOCATION = "us-central1"

    # 2. Folder Paths
    # This should point to a folder of your clean, pure emotion face crops
    INPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/ferckjalfaga_dataset_adult"
    # This is where the newly generated images will be saved
    OUTPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/generated_neutral_speech_faces"

    # 3. Generation Settings
    # How many different variations to create for EACH input image
    VARIATIONS_PER_IMAGE = 3
    
    # 4. Model Settings
    # The Vertex AI model to use for image editing
    MODEL_NAME = "imagegeneration@006"

# --- DYNAMIC PROMPT COMPONENTS ---
# These lists allow you to systematically create a diverse and balanced dataset.
# Feel free to add or remove items to suit your specific needs.
DEMOGRAPHICS = [
    "a 25-year-old Caucasian woman", "a 30-year-old Black man", "a 45-year-old East Asian woman",
    "a 50-year-old Hispanic man", "a 22-year-old South Asian person", "a 60-year-old Afro-Latina woman",
    "a 35-year-old Middle Eastern man", "a senior white man with a beard", "a young Black woman with braids"
]

MOUTH_SHAPES = {
    "ah_sound": "mouth slightly open as if in the middle of saying 'ah' or 'father'",
    "oh_sound": "mouth rounded as if beginning to say 'oh' or 'boat'",
    "ff_sound": "lips slightly parted, upper teeth may be touching the lower lip, as in the 'f' or 'v' sound",
    "th_sound": "lips parted with the tip of the tongue slightly visible between the teeth",
    "ee_sound": "lips are slightly stretched horizontally, as in the 'ee' sound in 'see', without smiling"
}


# In[3]:


class SpeechInpaintingPipeline:
    """
    An automated pipeline to convert emotional facial expressions to neutral speech expressions.
    It uses MediaPipe for automatic face masking and Vertex AI for AI-powered inpainting.
    """
    def __init__(self):
        # Initialize the Vertex AI model
        vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)
        self.model = ImageGenerationModel.from_pretrained(Config.MODEL_NAME)

        # Initialize MediaPipe Face Mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    def create_lower_face_mask(self, image, landmarks):
        """Creates a binary mask covering the lower half of the face using landmarks."""
        ih, iw, _ = image.shape
        mask = np.zeros((ih, iw), dtype=np.uint8)

        # These specific landmark indices from MediaPipe trace the jawline and up over the nose
        # to create a comprehensive mask for the lower face.
        jaw_points_indices = [
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
            415, 318, 402, 317, 14, 87, 178, 88, 95, 58, 169, 4, 395, 394
        ]
        
        points = []
        for index in jaw_points_indices:
            x = int(landmarks.landmark[index].x * iw)
            y = int(landmarks.landmark[index].y * ih)
            points.append([x, y])

        # Create a convex hull from the points and fill it to create a solid mask
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask

    def run(self):
        """Executes the main processing pipeline."""

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        all_image_paths = []
        for root, _, files in os.walk(Config.INPUT_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(root, file))
        
        if not all_image_paths:
            print(f"No images found in '{Config.INPUT_DIR}' or its subdirectories.")
            return
            
        print(f"Found {len(all_image_paths)} images in all subdirectories to process.")
        
        for input_path in tqdm(all_image_paths, desc="Processing Images"):
            filename = os.path.basename(input_path)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue

            # 1. Detect Landmarks
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                print(f"Warning: No face landmarks detected in {filename}. Skipping.")
                continue

            # 2. Create the custom mask for this image
            mask_array = self.create_lower_face_mask(image, results.multi_face_landmarks[0])
            mask_image = Image.from_array(mask_array)
            source_image = Image.load_from_file(input_path)

            # 3. Loop to create multiple, diverse variations for the same source image
            for i in range(Config.VARIATIONS_PER_IMAGE):
                try:
                    # 3a. Build a dynamic, descriptive prompt
                    demographic = random.choice(DEMOGRAPHICS)
                    mouth_shape_key, mouth_shape_desc = random.choice(list(MOUTH_SHAPES.items()))

                    dynamic_prompt = (
                        f"The lower face of {demographic}, including the mouth, chin, and jaw. "
                        f"The mouth shape should look like this: {mouth_shape_desc}. "
                        f"The expression must be completely neutral. "
                        f"Match the lighting, focus, and skin texture of the original image."
                    )
                    
                    # 3b. Call the AI Inpainting API
                    edited_images = self.model.edit_image(
                        base_image=source_image,
                        mask=mask_image,
                        prompt=dynamic_prompt,
                        # Negative prompt to avoid unwanted results
                        negative_prompt="smiling, frowning, teeth, emotion, cartoon, painting, blurry, deformed, ugly"
                    )
                    
                    # 3c. Save the Output with a descriptive filename
                    base_name = os.path.splitext(filename)[0]
                    demo_slug = demographic.replace(' ', '-').lower()
                    output_filename = f"{base_name}_{demo_slug}_{mouth_shape_key}_v{i+1}.png"
                    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
                    edited_images[0].save(location=output_path)
                    
                except Exception as e:
                    print(f"\nAn error occurred while processing a variation for {filename}: {e}")

        self.face_mesh.close()
        print("\nProcessing complete.")


# In[4]:


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # A simple check to ensure cloud authentication is likely configured.
    # For a more robust solution, use `gcloud auth application-default login`
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
         print("---")
         print("WARNING: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
         print("Please ensure you have authenticated with Google Cloud CLI.")
         print("---")
         
    pipeline = SpeechInpaintingPipeline()
    pipeline.run()


# In[ ]:





# In[ ]:




