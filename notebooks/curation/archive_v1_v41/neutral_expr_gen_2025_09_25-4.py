#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random
import io
import google.generativeai as genai
from PIL import Image


# In[2]:


# --- AUTHENTICATION SETUP ---
# Set the environment variable for Google Cloud authentication directly in the script.
# This line replaces the need to run the 'export' command in your terminal.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/natalyagrokh/AI/gemini-env/curation_pipeline/key.json'


# In[3]:


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
    MODEL_NAME = "gemini-1.5-pro-latest"

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


# In[4]:


class SpeechInpaintingPipeline:
    """
    An automated pipeline to convert emotional facial expressions to neutral speech expressions.
    This version uses the modern GenerativeAI library to support longer timeouts.
    """
    def __init__(self):
        # Initialize the modern Generative AI model with the timeout from your playbook
        self.model = genai.GenerativeModel(
            model_name=Config.MODEL_NAME,
            request_options={"timeout": 1800}  # 30-minute timeout
        )

        # Initialize MediaPipe Face Mesh for landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    def create_lower_face_mask(self, image):
        """Creates a binary mask covering the lower half of the face using landmarks."""
        # This function is slightly modified to only need the image
        ih, iw, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None # Return None if no face is found

        landmarks = results.multi_face_landmarks[0]
        mask = np.zeros((ih, iw), dtype=np.uint8)

        jaw_points_indices = [
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
            415, 318, 402, 317, 14, 87, 178, 88, 95, 58, 169, 4, 395, 394
        ]
        
        points = [
            [int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)]
            for idx in jaw_points_indices
        ]

        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask

    def run(self):
        """Executes the main processing pipeline."""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        all_image_paths = [os.path.join(r, f) for r, _, fs in os.walk(Config.INPUT_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_image_paths:
            print(f"No images found in '{Config.INPUT_DIR}'.")
            return
            
        print(f"Found {len(all_image_paths)} images to process.")
        
        for input_path in tqdm(all_image_paths, desc="Processing Images"):
            try:
                source_image_cv = cv2.imread(input_path)
                if source_image_cv is None:
                    print(f"Warning: Could not read {os.path.basename(input_path)}. Skipping.")
                    continue

                mask_array = self.create_lower_face_mask(source_image_cv)
                if mask_array is None:
                    print(f"Warning: No face landmarks detected in {os.path.basename(input_path)}. Skipping.")
                    continue

                # The new library can accept image data directly
                source_image_pil = Image.open(input_path)
                mask_image_pil = Image.fromarray(mask_array)

                for i in range(Config.VARIATIONS_PER_IMAGE):
                    demographic = random.choice(DEMOGRAPHICS)
                    mouth_shape_key, mouth_shape_desc = random.choice(list(MOUTH_SHAPES.items()))
                    
                    dynamic_prompt = (
                        f"Inpaint the masked area of the image. The original image shows a person's face. "
                        f"The masked area covers their lower face (mouth, chin, jaw). "
                        f"Generate a new lower face for {demographic} where the mouth is shaped for speech: {mouth_shape_desc}. "
                        f"The final expression must be completely neutral. Do not show teeth. "
                        f"Match the lighting, focus, and skin texture of the original, unmasked part of the image."
                    )
                    
                    # Call the API using the modern generate_content method
                    response = self.model.generate_content(
                        [dynamic_prompt, source_image_pil, mask_image_pil]
                    )
                    
                    # The new library returns image bytes directly
                    output_bytes = response.parts[0].data
                    output_image = Image.open(io.BytesIO(output_bytes))
                    
                    # Save the Output
                    filename = os.path.basename(input_path)
                    base_name = os.path.splitext(filename)[0]
                    demo_slug = demographic.replace(' ', '-').lower()
                    output_filename = f"{base_name}_{demo_slug}_{mouth_shape_key}_v{i+1}.png"
                    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
                    output_image.save(output_path)

            except Exception as e:
                print(f"\nAn error occurred while processing a variation for {os.path.basename(input_path)}: {e}")

        self.face_mesh.close()
        print("\nProcessing complete.")

# You will also need to add 'io' to your imports
import io


# In[5]:


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        pipeline = SpeechInpaintingPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\nAn unexpected error occurred during pipeline execution: {e}")
        print("Please check your authentication, project ID, and file paths.")


# In[ ]:




