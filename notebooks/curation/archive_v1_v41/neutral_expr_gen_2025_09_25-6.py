#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random
import io
import json
import base64
import requests
from PIL import Image
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# --- CONFIGURATION ---
class Config:
    PROJECT_ID = "MLexpressImgSorting"
    LOCATION = "us-central1"
    INPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/ferckjalfaga_dataset_adult"
    OUTPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/generated_neutral_speech_faces"
    KEY_FILE_PATH = '/Users/natalyagrokh/AI/gemini-env/curation_pipeline/key.json'
    MODEL_NAME = "gemini-1.5-pro-latest"
    VARIATIONS_PER_IMAGE = 3

# --- DYNAMIC PROMPT COMPONENTS ---
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

# --- AUTHENTICATION HELPER ---
def get_access_token():
    """Generates a short-lived access token from the service account key."""
    creds = service_account.Credentials.from_service_account_file(
        Config.KEY_FILE_PATH, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    if not creds.valid:
        creds.refresh(Request())
    return creds.token

# --- FACE MASK HELPER ---
def create_lower_face_mask(image, face_mesh_processor):
    ih, iw, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_processor.process(rgb_image)
    if not results.multi_face_landmarks: return None
    landmarks = results.multi_face_landmarks[0]
    mask = np.zeros((ih, iw), dtype=np.uint8)
    jaw_points_indices = [
        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
        415, 318, 402, 317, 14, 87, 178, 88, 95, 58, 169, 4, 395, 394
    ]
    points = [[int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)] for idx in jaw_points_indices]
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

# --- MAIN EXECUTION LOGIC ---
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    all_image_paths = [os.path.join(r, f) for r, _, fs in os.walk(Config.INPUT_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_image_paths:
        print(f"No images found in '{Config.INPUT_DIR}'.")
        return
    print(f"Found {len(all_image_paths)} images to process.")

    # Initialize MediaPipe outside the loop
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    for input_path in tqdm(all_image_paths, desc="Processing Images"):
        try:
            source_image_cv = cv2.imread(input_path)
            if source_image_cv is None: continue
            
            mask_array = create_lower_face_mask(source_image_cv, face_mesh)
            if mask_array is None: continue

            # Convert images to base64 for the JSON request
            _, source_buffer = cv2.imencode('.jpg', source_image_cv)
            source_b64 = base64.b64encode(source_buffer).decode('utf-8')

            _, mask_buffer = cv2.imencode('.png', mask_array)
            mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')

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

                # --- Manual API Request Construction ---
                access_token = get_access_token()
                url = f"https://{Config.LOCATION}-aiplatform.googleapis.com/v1/projects/{Config.PROJECT_ID}/locations/{Config.LOCATION}/publishers/google/models/{Config.MODEL_NAME}:generateContent"
                
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                
                body = {
                    "contents": [
                        {"role": "user", "parts": [{"text": dynamic_prompt}]},
                        {"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": source_b64}}]},
                        {"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": mask_b64}}]},
                    ]
                }

                # Make the request using the 'requests' library with a timeout
                response = requests.post(url, headers=headers, json=body, timeout=600) # 10 minute timeout
                response.raise_for_status() # Will raise an error for 4xx or 5xx status codes
                
                response_json = response.json()
                output_b64 = response_json['candidates'][0]['content']['parts'][0]['inlineData']['data']
                output_bytes = base64.b64decode(output_b64)
                output_image = Image.open(io.BytesIO(output_bytes))

                # Save the Output
                filename = os.path.basename(input_path)
                base_name = os.path.splitext(filename)[0]
                demo_slug = demographic.replace(' ', '-').lower()
                output_filename = f"{base_name}_{demo_slug}_{mouth_shape_key}_v{i+1}.png"
                output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
                output_image.save(output_path)

        except Exception as e:
            print(f"\nAn error occurred while processing {os.path.basename(input_path)}: {e}")

    face_mesh.close()
    print("\nProcessing complete.")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    main()


# In[ ]:


import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random
import io
import google.generativeai as genai
from PIL import Image


# In[ ]:


# --- AUTHENTICATION SETUP ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/natalyagrokh/AI/gemini-env/curation_pipeline/key.json'


# In[ ]:


# --- CONFIGURATION ---
class Config:
    # Set your Project ID. The library will use this via the authentication credentials.
    PROJECT_ID = "MLexpressImgSorting"
    
    # Folder Paths
    INPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/ferckjalfaga_dataset_adult"
    OUTPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/generated_neutral_speech_faces"

    # Generation Settings
    VARIATIONS_PER_IMAGE = 3
    
    # Model Settings: Use the modern Gemini model
    MODEL_NAME = "gemini-1.5-pro-latest"

# --- DYNAMIC PROMPT COMPONENTS ---
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


# In[ ]:


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        pipeline = SpeechInpaintingPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\nAn unexpected error occurred during pipeline execution: {e}")


# In[ ]:


class SpeechInpaintingPipeline:
    def __init__(self):
        # Initialize the model without any project/location config.
        # The library will automatically pick it up from the auth credentials.
        self.model = genai.GenerativeModel(model_name=Config.MODEL_NAME)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    def create_lower_face_mask(self, image):
        ih, iw, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks: return None
        landmarks = results.multi_face_landmarks[0]
        mask = np.zeros((ih, iw), dtype=np.uint8)
        jaw_points_indices = [
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
            415, 318, 402, 317, 14, 87, 178, 88, 95, 58, 169, 4, 395, 394
        ]
        points = [[int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)] for idx in jaw_points_indices]
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def run(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        all_image_paths = [os.path.join(r, f) for r, _, fs in os.walk(Config.INPUT_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_image_paths:
            print(f"No images found in '{Config.INPUT_DIR}'.")
            return
        print(f"Found {len(all_image_paths)} images to process.")
        
        for input_path in tqdm(all_image_paths, desc="Processing Images"):
            try:
                source_image_cv = cv2.imread(input_path)
                if source_image_cv is None: continue
                mask_array = self.create_lower_face_mask(source_image_cv)
                if mask_array is None: continue
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
                    
                    # Set the timeout directly in the API call.
                    response = self.model.generate_content(
                        [dynamic_prompt, source_image_pil, mask_image_pil],
                        request_options={"timeout": 1800}  # 30-minute timeout
                    )
                    
                    output_bytes = response.parts[0].data
                    output_image = Image.open(io.BytesIO(output_bytes))
                    
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


# In[ ]:


"""
# --- TROUBLESHOOTING NOTE ---
# This script was developed using syntax from modern Google AI libraries, but the Conda
# environment has an older version installed due to persistent dependency conflicts that
# prevented a successful upgrade. This version mismatch caused a series of errors,
# including a TypeError for the 'request_options' argument in the GenerativeModel
# constructor and an AttributeError because the 'Part.from_pil()' method did not exist.
#
# The current implementation works around this by using syntax compatible with the older
# library: the timeout is set directly in the 'model.generate_content()' call, and
# images are manually converted to in-memory byte streams before being passed to
# 'Part.from_data()'. If this script is moved to a new environment with updated
# libraries, this code can be simplified.
"""


# In[ ]:


# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from tqdm import tqdm
# import random
# import io
# from PIL import Image
# import vertexai
# from vertexai.generative_models import GenerativeModel, Part


# In[ ]:


# # --- AUTHENTICATION SETUP ---
# # Set the environment variable for Google Cloud authentication directly in the script.
# # This line replaces the need to run the 'export' command in your terminal.
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/natalyagrokh/AI/gemini-env/curation_pipeline/key.json'


# In[ ]:


# # --- CONFIGURATION ---
# # All user-configurable settings are in this class for easy management.
# class Config:
#     # 1. Your Google Cloud Project Details
#     PROJECT_ID = "MLexpressImgSorting"
#     LOCATION = "us-central1"

#     # 2. Folder Paths
#     # This should point to a folder of your clean, pure emotion face crops
#     INPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/ferckjalfaga_dataset_adult"
#     # This is where the newly generated images will be saved
#     OUTPUT_DIR = "/Users/natalyagrokh/AI/ml_expressions/img_datasets/generated_neutral_speech_faces"

#     # 3. Generation Settings
#     # How many different variations to create for EACH input image
#     VARIATIONS_PER_IMAGE = 3
    
#     # 4. Model Settings
#     # The Vertex AI model to use for image editing
#     MODEL_NAME = "gemini-1.5-pro-latest"

# # --- DYNAMIC PROMPT COMPONENTS ---
# # These lists allow you to systematically create a diverse and balanced dataset.
# # Feel free to add or remove items to suit your specific needs.
# DEMOGRAPHICS = [
#     "a 25-year-old Caucasian woman", "a 30-year-old Black man", "a 45-year-old East Asian woman",
#     "a 50-year-old Hispanic man", "a 22-year-old South Asian person", "a 60-year-old Afro-Latina woman",
#     "a 35-year-old Middle Eastern man", "a senior white man with a beard", "a young Black woman with braids"
# ]

# MOUTH_SHAPES = {
#     "ah_sound": "mouth slightly open as if in the middle of saying 'ah' or 'father'",
#     "oh_sound": "mouth rounded as if beginning to say 'oh' or 'boat'",
#     "ff_sound": "lips slightly parted, upper teeth may be touching the lower lip, as in the 'f' or 'v' sound",
#     "th_sound": "lips parted with the tip of the tongue slightly visible between the teeth",
#     "ee_sound": "lips are slightly stretched horizontally, as in the 'ee' sound in 'see', without smiling"
# }


# In[ ]:


# # Sets the project and location for your library version
# vertexai.init(
#     project=Config.PROJECT_ID,
#     location=Config.LOCATION,
# )


# In[ ]:


# class SpeechInpaintingPipeline:
#     """
#     An automated pipeline to convert emotional facial expressions to neutral speech expressions.
#     This version uses the native Vertex AI library for robust communication.
#     """
#     def __init__(self):
#         # Initialize the model using the native Vertex AI GenerativeModel class
#         self.model = GenerativeModel(Config.MODEL_NAME)

#         # Initialize MediaPipe Face Mesh for landmark detection
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
#         )

#     def create_lower_face_mask(self, image):
#         """Creates a binary mask covering the lower half of the face using landmarks."""
#         ih, iw, _ = image.shape
        
#         # --- THIS IS THE CORRECTED LINE ---
#         # Changed cv.COLOR_BGR2RGB to cv2.COLOR_BGR2RGB
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         results = self.face_mesh.process(rgb_image)
        
#         if not results.multi_face_landmarks:
#             return None

#         landmarks = results.multi_face_landmarks[0]
#         mask = np.zeros((ih, iw), dtype=np.uint8)

#         jaw_points_indices = [
#             172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
#             415, 318, 402, 317, 14, 87, 178, 88, 95, 58, 169, 4, 395, 394
#         ]
        
#         points = [
#             [int(landmarks.landmark[idx].x * iw), int(landmarks.landmark[idx].y * ih)]
#             for idx in jaw_points_indices
#         ]

#         hull = cv2.convexHull(np.array(points))
#         cv2.fillConvexPoly(mask, hull, 255)
        
#         return mask

#     def run(self):
#         """Executes the main processing pipeline."""
#         os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
#         all_image_paths = [os.path.join(r, f) for r, _, fs in os.walk(Config.INPUT_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if not all_image_paths:
#             print(f"No images found in '{Config.INPUT_DIR}'.")
#             return
            
#         print(f"Found {len(all_image_paths)} images to process.")
        
#         for input_path in tqdm(all_image_paths, desc="Processing Images"):
#             try:
#                 source_image_cv = cv2.imread(input_path)
#                 if source_image_cv is None:
#                     print(f"Warning: Could not read {os.path.basename(input_path)}. Skipping.")
#                     continue

#                 mask_array = self.create_lower_face_mask(source_image_cv)
#                 if mask_array is None:
#                     print(f"Warning: No face landmarks detected in {os.path.basename(input_path)}. Skipping.")
#                     continue

#                 source_image_pil = Image.open(input_path)
#                 mask_image_pil = Image.fromarray(mask_array)

#                 for i in range(Config.VARIATIONS_PER_IMAGE):
#                     demographic = random.choice(DEMOGRAPHICS)
#                     mouth_shape_key, mouth_shape_desc = random.choice(list(MOUTH_SHAPES.items()))
                    
#                     dynamic_prompt = (
#                         f"Inpaint the masked area of the image. The original image shows a person's face. "
#                         f"The masked area covers their lower face (mouth, chin, jaw). "
#                         f"Generate a new lower face for {demographic} where the mouth is shaped for speech: {mouth_shape_desc}. "
#                         f"The final expression must be completely neutral. Do not show teeth. "
#                         f"Match the lighting, focus, and skin texture of the original, unmasked part of the image."
#                     )
                    
#                     # --- THIS IS THE CORRECTED SECTION ---
#                     # Manually convert PIL images to bytes and create Part objects using .from_data
#                     # This is the syntax required by your older library version.
#                     source_bytes = io.BytesIO()
#                     source_image_pil.save(source_bytes, format='JPEG')
#                     source_part = Part.from_data(source_bytes.getvalue(), mime_type="image/jpeg")

#                     mask_bytes = io.BytesIO()
#                     mask_image_pil.save(mask_bytes, format='PNG')
#                     mask_part = Part.from_data(mask_bytes.getvalue(), mime_type="image/png")
#                     # --- END CORRECTION ---

#                     response = self.model.generate_content(
#                         [dynamic_prompt, source_part, mask_part], # Use the new part objects
#                         request_options={"timeout": 1800}
#                     )
                    
#                     output_part = response.candidates[0].content.parts[0]
                    
#                     # The older library may return a different object type. This handles both.
#                     if hasattr(output_part, 'image'):
#                         output_image = output_part.image
#                     else:
#                         output_image_bytes = output_part.inline_data.data
#                         output_image = Image.open(io.BytesIO(output_image_bytes))

#                     # Save the Output
#                     filename = os.path.basename(input_path)
#                     base_name = os.path.splitext(filename)[0]
#                     demo_slug = demographic.replace(' ', '-').lower()
#                     output_filename = f"{base_name}_{demo_slug}_{mouth_shape_key}_v{i+1}.png"
#                     output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
#                     output_image.save(output_path)

#             except Exception as e:
#                 print(f"\nAn error occurred while processing a variation for {os.path.basename(input_path)}: {e}")

#         self.face_mesh.close()
#         print("\nProcessing complete.")


# In[ ]:


# # --- MAIN EXECUTION BLOCK ---
# if __name__ == "__main__":
#     try:
#         pipeline = SpeechInpaintingPipeline()
#         pipeline.run()
#     except Exception as e:
#         print(f"\nAn unexpected error occurred during pipeline execution: {e}")
#         print("Please check your authentication, project ID, and file paths.")

