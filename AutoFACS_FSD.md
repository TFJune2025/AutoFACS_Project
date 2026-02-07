# AutoFACS Level 10 Functional Specification (FSD)

## 1. Architectural Mandate
- **Core System**: Level 10 Autonomous MLOps for Facial Action Coding System (FACS) analysis.
- **Hardware Target**: Mandatory Apple Silicon (M-Series) optimization using `device='mps'`. CPU fallback is a system failure.
- **Root Directory**: `/Users/natalyagrokh/AutoFACS_Project`

## 2. Data Lake Management (Golden Schema)
- **Mount Point**: `/data_lake` (FUSE-T / GDrive API).
- **Discovery Strategy**: API-Direct scouting via `rclone lsjson`. Direct recursive scanning (os.walk) is prohibited to prevent "Metadata Storms."
- **Inventory Source**: `all_files.txt` is the primary local cache for file metadata.
- **Organization**: 
    - Weights must be flattened to `/data_lake/model_weights/V{iteration}/`.
    - Raw assets reside in `/data_lake/raw_assets/`.

## 3. Agentic Protocol
- **Memory**: Sessions are restored via `agents.md`.
- **Verification**: Every structural change must be preceded by `verify_phase1.py` to confirm hardware/path integrity.
- **Janitorial Logic**: Automatic purging of macOS metadata (`.DS_Store`, `._*`) and `__pycache__`.