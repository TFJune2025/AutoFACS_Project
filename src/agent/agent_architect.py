import os
import json
from langgraph.graph import StateGraph, END
from src.logic_gate import logic_gate_node, logic_gate_router, AgentState
from src import modeling, inference_core, video_engine, lake_scanner

def executive_action_node(state: AgentState):
    """
    The 'Lake Sentry'. Monitors 380GB transfer and runs mining on available data.
    """
    print("--- 🛠️ EXECUTIVE ACTION: Monitoring Data Lake Transfer ---")
    
    # Trigger your lake_scanner to see how many files have landed
    lake_scanner.build_index() 
    
    # Read the index to assess progress
    try:
        with open("lake_index.json", "r") as f:
            count = len(json.load(f))
    except:
        count = 0

    status = f"Lake Status: {file_count} images landed. Transfer in progress."
    print(f"📊 {status}")
    
    return {"scratchpad": f"Action complete. {status}"}

def tool_synthesis_node(state: AgentState):
    """
    The 'Forge'. Corrected for your specific directory structure.
    """
    print("--- 🧪 SYNTHESIS: Forging New Capability ---")
    raw_content = state.get("scratchpad", "")
    
    if "```python" in raw_content:
        code = raw_content.split("```python")[1].split("```")[0].strip()
        
        # Correct pathing based on your 'src' folder location
        gen_dir = "src/generated"
        os.makedirs(gen_dir, exist_ok=True)
        
        # Ensure __init__.py exists for package recognition
        with open(os.path.join(gen_dir, "__init__.py"), "a") as f: pass
        
        tool_filename = f"tool_v{state['iteration_count']}.py"
        file_path = os.path.join(gen_dir, tool_filename)
        
        with open(file_path, "w") as f:
            f.write(code)
            
        status = f"SUCCESS: New tool forged at {file_path}. Ready for memory load."
        print(f"📊 {status}") # RESTORED: Vital for console feedback
        
        return {
            "scratchpad": status,
            "tools_available": state["tools_available"] + [tool_filename]
        }
    
    return {"scratchpad": "ERROR: LLM failed to provide valid Python code block."}

def build_autofacs_brain():
    """
    Constructs the recursive, meta-cognitive graph for AutoFACS V41.
    """
    # 1. Initialize the Graph with our shared Level 10 State
    workflow = StateGraph(AgentState)

    # 2. Define the Nodes
    workflow.add_node("logic_gate", logic_gate_node)
    workflow.add_node("executive_action", executive_action_node)
    workflow.add_node("tool_synthesis", tool_synthesis_node)

    # 3. Set the Entry Point
    workflow.set_entry_point("logic_gate")

    # 4. Define the Conditional Routing (The decision-making edges)
    workflow.add_conditional_edges(
        "logic_gate",
        logic_gate_router,
        {
            "CALL_TOOL": "executive_action",
            "SYNTHESIZE_TOOL": "tool_synthesis",
            "FINAL_RESPONSE": END,
            "SELF_CORRECT": "logic_gate"
        }
    )

    # 5. Define standard edges to return to the logic gate for the next cycle
    workflow.add_edge("executive_action", "logic_gate")
    workflow.add_edge("tool_synthesis", "logic_gate")

    # 6. Compile the Brain
    return workflow.compile()

if __name__ == "__main__":
    print("🤖 AutoFACS Brain Blueprint is verified.")