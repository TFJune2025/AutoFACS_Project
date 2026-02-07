import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
# We force 'v1' here to see stable production models
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1'})

print("🔍 AVAILABLE MODELS FOR YOUR KEY:")
for model in client.models.list():
    print(f"-> {model.name} (Supports: {model.supported_actions})")