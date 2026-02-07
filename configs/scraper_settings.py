# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential.
# ==============================================================================

import os
from dotenv import load_dotenv

# Load variables from the hidden .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- 📂 Global Storage Paths ---
# Use the environment variable or fall back to the default local path
BASE_SCRAPE_DIR = os.getenv("MODEL_OUTPUT_PATH", os.path.expanduser("~/AutoFACS_Project/gdrive/datasets"))
PEXELS_OUTPUT_DIR = os.path.join(BASE_SCRAPE_DIR, "pexels_scraped")
FLICKR_OUTPUT_DIR = os.path.join(BASE_SCRAPE_DIR, "flickr_scraped")

# --- 🔑 API Credentials ---
# Keys are now pulled directly from your .env file
CREDENTIALS = {
    "pexels": {
        "api_key": os.getenv("PEXELS_API_KEY"),
        "url": "https://api.pexels.com/v1/search"
    },
    "flickr": {
        "api_key": os.getenv("FLICKR_API_KEY"),
        "url": "https://www.flickr.com/services/rest/"
    }
}

# --- 🔍 Common Search Queries ---
SEARCH_QUERIES = [
    "candid+portrait", "street+photography", "spectator+reaction",
    "person+laughing", "pensive+person", "emotions", "contempt"
]