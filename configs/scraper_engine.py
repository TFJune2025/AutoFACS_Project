# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential.
# ==============================================================================

import os
import time
import json as json_mod
import requests
from pathlib import Path
from configs.scraper_settings import CREDENTIALS, BASE_SCRAPE_DIR

def get_next_image_index():
    """Scans the entire 2TB Data Lake to find the global highest image number."""
    highest = 0
    for root, _, _ in os.walk(BASE_SCRAPE_DIR):
        for f in Path(root).glob("image_*.jpg"):
            try:
                num = int(f.stem.split('_')[1])
                if num > highest: highest = num
            except: continue
    return highest + 1

class ScraperFactory:
    """Unified engine for Pexels and Flickr ingestion."""
    
    @staticmethod
    def download_image(url, save_path, metadata):
        """Standardized downloader for all providers"""
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(save_path, "wb") as f: f.write(r.content)
            with open(save_path.replace(".jpg", ".json"), "w") as f:
                json_mod.dump(metadata, f, indent=4)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def scrape_pexels(self, query, save_dir, limit=100):
        """Pexels-specific API logic."""
        headers = {"Authorization": CREDENTIALS["pexels"]["api_key"]}
        # ... logic to fetch from Pexels and call download_image ...
        pass