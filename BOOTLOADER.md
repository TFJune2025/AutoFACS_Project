# BOOTLOADER.md: AutoFACS Level 10 Initialization

**Status:** Critical System Artifact
**Role:** Session Sequencer & Architecture Enforcer
**Objective:** Initialize the "Iron Gate" environment without triggering FUSE-T metadata deadlocks.

---

## I. System Context (The "Iron Gate")
* **Root Directory:** `/Users/natalyagrokh/AutoFACS_Project`
* **Hardware Constraint:** Apple Silicon (M-Series) | **Required Target:** `device='mps'`
* **Storage Architecture:**
    * **Data Lake:** 2TB Google Drive via FUSE-T (Mount point: `/data_lake`)
    * **Access Protocol:** **API-DIRECT ONLY (Rclone).**
    * **Forbidden:** `os.walk`, `glob.glob`, or recursive `ls` on the mount point.

---

## II. The Initialization Sequence (Boot Loop)
*The Agent must execute these steps sequentially. Do not proceed to Step 3 until Step 2 is verified.*

### ☑️ Step 1: Hardware Guard (The "Brain")
**Goal:** Verify the M-Series MPS (Metal Performance Shaders) backend is accessible.
* **Action:** Execute a Python one-liner to confirm `torch.backends.mps.is_available()`.
* **Failure Condition:** If `False`, abort immediately. Do not attempt CPU fallback (latency is unacceptable).

### ☑️ Step 2: Portal Guard (The "Body")
**Goal:** Verify Cloud Connectivity without touching the File System.
* **Action:** Run `rclone listremotes`.
* **Verification:** Ensure the specific remote (e.g., `gdrive:`) is listed and active.
* **Constraint:** **DO NOT** run `rclone ls` or mount check yet. Only verify the binary and config existence.

### ☑️ Step 3: Memory Injection (The "Eyes")
**Goal:** Load the pre-computed reality to prevent hallucination.
* **Action:** Ingest `all_files.txt` (The Map) and `AutoFACS_FSD.md` (The Law).
* **Logic Check:** Compare the *current* file count in `all_files.txt` against the *last known* count in `agents.md` (if available).

### ☑️ Step 4: Persistence Guard (Level 4)
[cite_start]**Goal:** Enforce History Tracking & Corruption Protection.
* **Action:** Check `git status`.
* **Constraint:** If the tree is dirty (modified files exist), the Agent must propose a `git commit` **BEFORE** starting new logic.
* [cite_start]**Protocol:** "Update documentation and push".

---

## III. Agent Directive: The "Bootloader Protocol"
**Upon ingesting this file, the Agent shall:**

1.  **Generate `boot.py`:** If not present, create a script that automates the `PYTHONPATH` export and runs the Step 1 & Step 2 checks.
2.  **Execute `boot.py`:** Run the script via the terminal.
3.  **Report Status:** Output a structured boot log:
    > * **MPS Status:** [ONLINE/OFFLINE]
    > * **Rclone Binary:** [DETECTED/MISSING]
    > * **Map Loaded:** [YES/NO] - [Count] files indexed.
4.  **Await Command:** Enter **Standby Mode**. Do not scan the lake until explicitly ordered to execute a targeted `rclone` operation defined in the FSD.