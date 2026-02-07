import pandas as pd
import os

MANIFEST_PATH = 'DATA_LAKE_MANIFEST_V41.csv'

def verify_heartbeat():
    print(f"🚀 Initializing Resilient Heartbeat Check...")
    
    try:
        # Use sep=None with engine='python' to auto-detect, 
        # but we force comma while allowing for the 'bad' lines.
        df = pd.read_csv(
            MANIFEST_PATH, 
            on_bad_lines='skip', 
            engine='python',
            quoting=1 # csv.QUOTE_ALL
        )
        
        # Robust column detection
        cols = {col.lower(): col for col in df.columns}
        path_key = next((v for k, v in cols.items() if 'path' in k), None)
        
        if not path_key:
            print("❌ Error: Could not find a 'Path' column. Columns found:", df.columns.tolist())
            return

        print(f"✅ Manifest Loaded. Analyzing {len(df):,} entries...")
        
        # Check a sample to ensure paths are valid
        sample_path = df[path_key].iloc[0]
        print(f"🔎 Sample Path: {sample_path}")
        
    except Exception as e:
        print(f"💥 Critical Failure: {e}")

if __name__ == "__main__":
    verify_heartbeat()