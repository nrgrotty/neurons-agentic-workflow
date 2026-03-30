
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.creative_editor.models import (
    GraphState,
    PipelineState,
)
from neurons_agentic_workflow.creative_editor.service.graph import main_graph

async def _recommendation_branch(state: GraphState) -> dict:
    """Run the full per-recommendation graph and return its final_image for fan-in."""
    final = await main_graph.ainvoke(state)
    if isinstance(final, dict):
        final_image = final["final_image"]
        branch_audit = final.get("audit_trail", [])
    else:
        final_image = final.final_image
        branch_audit = final.audit_trail
    return {"final_images": [final_image], "audit_trail": branch_audit}


def _dispatch_recommendations(state: PipelineState) -> list[Send]:
    """Fan out: one Send per recommendation."""
    return [
        Send("recommendation_branch", GraphState(
            image=state.image,
            brand_guidelines=state.brand_guidelines,
            recommendation=rec,
        ).model_dump())
        for rec in state.recommendations
    ]


def _build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("recommendation_branch", _recommendation_branch)
    graph.add_conditional_edges(START, _dispatch_recommendations, ["recommendation_branch"])
    graph.add_edge("recommendation_branch", END)
    return graph.compile()


pipeline = _build_pipeline()
