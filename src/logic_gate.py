import os
import sys
import importlib.util
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. State Definition ---
class AgentState(TypedDict):
    objective: str
    messages: List[Any]
    tools_available: List[str]
    iteration_count: int
    scratchpad: str

# --- 2. The Evolution Bridge (Dynamic Import Hook) ---
def dynamic_import_tool(tool_name: str, file_path: str):
    """Safely loads synthesized code into the active runtime."""
    try:
        spec = importlib.util.spec_from_file_location(tool_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[tool_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        return f"EVOLUTION FAILED: {e}"
    return "SUCCESS"

# --- 3. The Logic Gate Implementation ---
def logic_gate_node(state: AgentState):
    """
    Reasoning Node: Now includes automatic memory loading for forged tools.
    """
    # MISSION CRITICAL: Load any synthesized tools into memory before reasoning
    for tool_file in state.get("tools_available", []):
        tool_name = tool_file.replace(".py", "")
        if "tool_v" in tool_name and tool_name not in sys.modules:
            path = os.path.join("src/generated", tool_file)
            if os.path.exists(path):
                dynamic_import_tool(tool_name, path)
                print(f"--- 🚀 EVOLUTION: Tool '{tool_name}' loaded into memory ---")

    load_dotenv("configs/.env")
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), 
        http_options=types.HttpOptions(api_version='v1')
    )

    # Self-Healing Check
    repair_context = ""
    if "EVOLUTION FAILED" in state.get("scratchpad", ""):
        repair_context = f"\nSYSTEM ALERT: Previous synthesis failed. Context: {state['scratchpad']}. FIX the code logic."

    prompt = f"""
    Role: Lead GenAI Architect for AutoFACS V41. {repair_context}
    Current Objective: {state['objective']}
    Available Tools: {state['tools_available']}
    Iteration: {state['iteration_count']}
    
    Instruction: Analyze state. Output [CALL_TOOL], [SYNTHESIZE_TOOL], or [FINAL_RESPONSE].
    If synthesizing, provide the code in a ```python block.
    
    Provide your reasoning in the scratchpad.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return {
        "scratchpad": response.text,
        "iteration_count": state['iteration_count'] + 1
    }

def logic_gate_router(state: AgentState) -> Literal["CALL_TOOL", "SYNTHESIZE_TOOL", "FINAL_RESPONSE", "SELF_CORRECT"]:
    decision = state.get("scratchpad", "")
    if "[CALL_TOOL]" in decision: return "CALL_TOOL"
    if "[SYNTHESIZE_TOOL]" in decision: return "SYNTHESIZE_TOOL"
    if "[FINAL_RESPONSE]" in decision: return "FINAL_RESPONSE"
    return "SELF_CORRECT"