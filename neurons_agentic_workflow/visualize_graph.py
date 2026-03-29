"""
Run with:
    poetry run python neurons_agentic_workflow/visualize_graph.py
Saves graph.png in the project root.
"""

from pathlib import Path
from neurons_agentic_workflow.creative_editor.service import _graph

output = Path(__file__).parent.parent / "graph.png"
output.write_bytes(_graph.get_graph().draw_mermaid_png())
print(f"Graph saved to {output}")
