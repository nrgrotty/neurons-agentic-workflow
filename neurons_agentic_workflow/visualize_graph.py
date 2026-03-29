"""
Plot graph and subgraphs visualization. 
"""

from pathlib import Path
from neurons_agentic_workflow.creative_editor.service import _graph
from neurons_agentic_workflow.creative_editor.service.nodes import _worker_subgraph

root = Path(__file__).parent.parent

output = root / "graph.png"
output.write_bytes(_graph.get_graph().draw_mermaid_png())
print(f"Outer graph saved to {output}")

worker_output = root / "worker_subgraph.png"
worker_output.write_bytes(_worker_subgraph.get_graph().draw_mermaid_png())
print(f"Worker subgraph saved to {worker_output}")
