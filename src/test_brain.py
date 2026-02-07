import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load Secrets
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
project_path = os.getenv("PROJECT_PATH")

# 2. Initialize the 2026 Production Client
# We explicitly set 'v1' to use the stable production endpoints
client = genai.Client(
    api_key=api_key, 
    http_options=types.HttpOptions(api_version='v1')
)

# 3. Data Lake Check
data_lake_path = os.path.join(project_path, "data_lake")
try:
    contents = os.listdir(data_lake_path)
except FileNotFoundError:
    contents = "Error: Mount point not found."

# 4. The Request with Thinking (Standard in 2026)
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Natalya here. My data lake contains: {contents}. Status report for AutoFACS V41?"
    )
    print(f"\n🧠 AGENT RESPONSE:\n{response.text}")
except Exception as e:
    print(f"\n❌ CONNECTION BLOCKED: {e}")
    print("\n💡 TIP: If you see 'limit: 0', ensure you clicked 'Activate' in AI Studio.")