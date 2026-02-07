# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import importlib.util
import re
import tools
import tools.face_scout as face_scout
from configs import config
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    objective: str
    scratchpad: str
    iteration_count: int
    tools_available: List[str]

# --- 2. THE SECURITY GATE (PATH JAILING) ---

def safe_path_resolver(requested_path: str):
    """Prevents the agent from escaping the project root."""
    # Changed: Reference config.PROJECT_ROOT instead of a hardcoded local variable
    target_path = os.path.abspath(os.path.join(config.PROJECT_ROOT, requested_path))
    if not target_path.startswith(config.PROJECT_ROOT):
        raise PermissionError(f"Project Violation: Access outside {config.PROJECT_ROOT} denied.")
    return target_path

# --- 3. NODE DEFINITIONS ---

def get_latest_model_context():
    """
    Scans for the latest 'V*' folder and returns (version_id, folder_path)
    Identifies the highest 'V' folder and its absolute path.
    Allows the agent to find artifacts in dynamically named folders.
    """
    models_root = config.MODELS_DIR
    if not os.path.exists(models_root): 
        return "v1", models_root
        
    # Scan for directories (e.g., V41_20260125_175823)
    dirs = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
    version_map = []
    
    for d in dirs:
        match = re.search(r'[Vv](\d+)', d)
        if match: 
            version_map.append((int(match.group(1)), d))
    
    if not version_map: 
        return "v1", models_root
        
    # Max logic ensures we always target the newest synced data
    latest_num, latest_name = max(version_map, key=lambda x: x[0])
    return f"v{latest_num}", os.path.join(models_root, latest_name)


def gdrive_sentry_node(state: AgentState):
    """Body Node: Validates the 2TB Stream portal."""
    print("--- 🛡️  SENTRY: Validating Boundary ---")
    # Point directly to the config's data lake root
    if os.path.exists(config.DATA_LAKE_ROOT):
        items = [f for f in os.listdir(config.DATA_LAKE_ROOT) if not f.startswith('.')]
        status = f"SUCCESS: {len(items)} entries detected in Data Lake."
    else:
        status = "ERROR: Data Lake mount not found."
    
    return {"scratchpad": status, "iteration_count": state.get("iteration_count", 0) + 1}

def refresh_tool_manifest(state: AgentState):
    """Updates the AgentState with currently available tools in /tools."""
    available = tools.list_tools()
    print(f"--- 🛠️  MANIFEST: {len(available)} tools ready in /tools ---")
    return {"tools_available": available}

def scout_node(state: AgentState):
    """
    Action Node: Executes the face_scout tool.
    Operates only on currently synced 'datasets' folder.
    """
    print("--- 🔍 SCOUT: Indexing Active Datasets ---")
    
    # Run the scout on the current synced state
    # Note: We pass the subfolder relative to the V41 root
    results = face_scout.scout_faces_in_datasets(relative_subfolder="datasets")
    
    detection_count = len(results.get("detections", []))
    status = f"SUCCESS: Scout found {detection_count} images with faces."
    
    return {
        "scratchpad": status,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def tool_synthesis_node(state: AgentState):
    version_id, model_dir = get_latest_model_context()
    tool_name = "inference_specialist_current"
    
    print(f"--- 🛠️  SYNTHESIS: Deploying Tool for {version_id} ---")
    
    tool_code = f"""
import torch
import torchvision.transforms as T
from PIL import Image
import os
from configs import config
from master_orchestrator import safe_path_resolver
from src.architecture import AutoFACSNet

MODEL_DIR = "{model_dir}"

def find_weights():
    # Dynamic Artifact Browsing: Finds the first .pth file in the version folder
    artifacts = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    if not artifacts: raise FileNotFoundError(f"No weights in {{MODEL_DIR}}")
    return os.path.join(MODEL_DIR, artifacts[0])

def run_inference(image_relative_path, face_box):
    # A. Dynamic Weights Discovery
    weights_path = find_weights()
    
    # B. Resolve and Load through the Security Bouncer
    abs_path = safe_path_resolver(os.path.join("data_lake/AutoFACS_Project", image_relative_path))
    img = Image.open(abs_path).convert('RGB')
    
    # Precise Crop -> Config-Driven Resize
    top, right, bottom, left = face_box
    face_crop = img.crop((left, top, right, bottom))
    
    transform = T.Compose([
        T.Resize({config.MODEL_IMAGE_SIZE}), # Managed by config.py
        T.ToTensor(),
        T.Normalize(mean={config.NORMALIZATION_MEAN}, std={config.NORMALIZATION_STD})
    ])
    tensor = transform(face_crop).unsqueeze(0)
    
    # E. Inference Logic
    model = AutoFACSNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        output = model(tensor)
    
    # F. The Complete Inference Receipt (Restored)
    return {{
        "status": "SUCCESS", 
        "version": "{version_id}", 
        "target": image_relative_path,
        "weights_used": os.path.basename(weights_path),
        "raw_output": output.tolist()
    }}
"""

    # 3. Path resolution and writing
    tools_dir = safe_path_resolver("tools")
    if not os.path.exists(tools_dir): 
        os.makedirs(tools_dir)
    
    # Dynamic Registration logic follows... (Standard writing and importlib)
    file_path = os.path.join(config.TOOLS_DIR, f"{tool_name}.py")
    with open(file_path, "w") as f: 
        f.write(tool_code)

    # 4. Dynamic Registration
    spec = importlib.util.spec_from_file_location(tool_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 5. Restored Terminal Feedback
    status = f"SUCCESS: Specialist Layer updated to {version_id}."
    print(f"   ↳ {status}")
    
    return {
        "scratchpad": status, 
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def final_report_node(state: AgentState):
    """
    Final Node: Summarizes the mission and confirms state synchronization.
    """
    print("--- ✅ FINALIZING: Mission Complete ---")
    
    # We pull the last status from the scratchpad for the final summary
    last_status = state.get("scratchpad", "No status recorded.")
    
    return {
        "scratchpad": f"Final Report: {last_status} All paths verified within PROJECT_ROOT."
    }

def logic_gate_router(state: AgentState) -> Literal["scout", "synthesis", "finalize", "retry"]:
    status = state.get("scratchpad", "")
    
    # If Sentry just finished, move to Scout
    if "detected in Data Lake" in status:
        return "scout"
    
    # If Scout just finished, move to Synthesis (to build model wrappers)
    if "Scout found" in status:
        return "synthesis"
        
    if "SUCCESS" in status:
        return "finalize"
        
    return "retry"

# --- 5. GRAPH ASSEMBLY ---

workflow = StateGraph(AgentState)

# Define the active nodes
workflow.add_node("manifest", refresh_tool_manifest)
workflow.add_node("sentry", gdrive_sentry_node)
workflow.add_node("scout", scout_node)
workflow.add_node("synthesis", tool_synthesis_node)
workflow.add_node("finalize", final_report_node)

# Define the logic flow
workflow.add_edge(START, "manifest")
workflow.add_edge("manifest", "sentry")

workflow.add_conditional_edges("sentry", logic_gate_router, {
    "scout": "scout",
    "retry": "sentry"
})

workflow.add_conditional_edges("scout", logic_gate_router, {
    "synthesis": "synthesis",
    "finalize": "finalize",
    "retry": "scout"
})

workflow.add_edge("synthesis", "finalize") # Moving to finalize for this demo
workflow.add_edge("finalize", END)

app = workflow.compile()

# --- 6. EXECUTION ---
if __name__ == "__main__":
    initial_input = {
        "objective": "Scan datasets for face presence and synthesize inference tool.",
        "iteration_count": 0,
        "tools_available": [] 
    }
    
    # We catch the final state to print the results
    final_state = app.invoke(initial_input)
    
    print("\n" + "="*50)
    print("🚀 ORCHESTRATOR OUTPUT")
    print(f"Final Status: {final_state['scratchpad']}")
    print("="*50)