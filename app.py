"""
Hugging Face Spaces entrypoint.
Loads 'Medical Assistant.py' (which has a space in its name) via importlib
and launches the Gradio demo bound to all interfaces.
"""

import importlib.util
import os
import sys

# Load the main module from the file with a space in its name
_spec = importlib.util.spec_from_file_location(
    "medical_assistant",
    os.path.join(os.path.dirname(__file__), "Medical Assistant.py"),
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["medical_assistant"] = _module
_spec.loader.exec_module(_module)

demo = _module.demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
