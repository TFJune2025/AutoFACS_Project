# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import importlib

def list_tools():
    """Lists all valid Python tools in this directory."""
    return [f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
            if f.endswith('.py') and f != '__init__.py']

def load_tool(name):
    """Dynamically loads a tool for the Orchestrator."""
    return importlib.import_module(f'tools.{name}')