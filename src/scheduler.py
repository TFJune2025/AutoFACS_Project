# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential.
# ==============================================================================

import schedule
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(filename='logs/scheduler.log', level=logging.INFO)

def daily_ingestion_job():
    """Triggers the unified scraper and logs the results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"🚀 Starting automated scrape at {now}")
    
    try:
        # Calls your main orchestrator in 'Scrape Mode'
        subprocess.run(["python", "main.py", "--mode", "scrape"], check=True)
        logging.info("✅ Daily ingestion successful.")
    except Exception as e:
        logging.error(f"❌ Automation failed: {e}")

# Schedule the job to run every day at 2:00 AM
schedule.every().day.at("02:00").do(daily_ingestion_job)

if __name__ == "__main__":
    print("🤖 AutoFACS Scheduler is active. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)