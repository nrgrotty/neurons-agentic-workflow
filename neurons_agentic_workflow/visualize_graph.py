"""
Plot graph and subgraphs visualization. 
"""

from pathlib import Path
from neurons_agentic_workflow.creative_editor.service.graph import main_graph
from neurons_agentic_workflow.creative_editor.service.nodes import _editor_editor_worker_subgraph

root = Path(__file__).parent.parent

output = root / "graph.png"
output.write_bytes(main_graph.get_graph().draw_mermaid_png())
print(f"Outer graph saved to {output}")

editor_editor_worker_output = root / "editor_editor_worker_subgraph.png"
editor_editor_worker_output.write_bytes(_editor_editor_worker_subgraph.get_graph().draw_mermaid_png())
print(f"editor_editor_worker subgraph saved to {editor_editor_worker_output}")
