import os
import google.generativeai as genai
from PIL import Image

# --- Configuration ---
# Set your Google Cloud project details here.
GCP_PROJECT_ID = "mlexpimgsorting-v2"  # Your new, working Project ID
GCP_LOCATION = "us-central1"

# This correctly sets the specific environment variable the Google SDK looks for.
os.environ = "/Users/natalyagrokh/AI/img_curation/mlexpimgsorting-v2-c5a570b110c3.json"

# --- Main Application Logic ---

def perform_inpainting(image_path: str, prompt: str):
    """
    Uses the Gemini 1.5 Pro model on Vertex AI to perform image inpainting.
    """
    print(f"Starting inpainting process for: {image_path}")

    try:
        # 1. CORRECT: Create a client explicitly configured for Vertex AI.
        # This is the modern and correct way to use the SDK for Vertex AI.
        client = genai.Client(
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )

        # 2. Prepare the image and prompt for the API call.
        print("Opening image...")
        image = Image.open(image_path)
        contents = [prompt, image]

        # 3. CORRECT: Make the API call through the client object.
        # The model name is now passed as an argument to this method.
        print("Sending request to Vertex AI API...")
        response = client.generate_content(
            model='gemini-1.5-pro',
            contents=contents
        )

        print("Successfully received response from the model.")
        return response

    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        return None

if __name__ == "__main__":
    # --- Example Usage ---
    example_image_path = "face_image.png"
    example_prompt = "This is an image of a person's face. Perform inpainting to remove the glasses."

    if not os.path.exists(example_image_path):
        print(f"Creating a dummy image file at '{example_image_path}' for demonstration.")
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        dummy_image.save(example_image_path)

    api_response = perform_inpainting(example_image_path, example_prompt)

    if api_response:
        print("\n--- Model Response ---")
        print(api_response.text)
        print("----------------------")