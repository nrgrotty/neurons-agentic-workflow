
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.models import (
    GraphState,
    PipelineOutput,
    PipelineState,
    RecommendationResult,
)
from neurons_agentic_workflow.service.graph import main_graph

async def _recommendation_branch(state: dict) -> dict:
    """Run the full per-recommendation graph and collect best + other variants."""
    graph_state = GraphState.model_validate(state)
    response = await main_graph.ainvoke(graph_state)
    final_image = response["final_image"]
    all_edited = response.get("edited_images", [])
    other_variants = [p for p in all_edited if p != final_image]
    branch_audit = response.get("audit_trail", [])
    result = RecommendationResult(
        recommendation_id=graph_state.recommendation.id,
        original_image=graph_state.image,
        best_variant=final_image,
        other_variants=other_variants,
    )
    return {"recommendation_results": [result], "audit_trail": branch_audit}


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


async def run_pipeline(state: PipelineState) -> PipelineOutput:
    """Run the pipeline and return a structured PipelineOutput."""
    response = await pipeline.ainvoke(state.model_dump())
    return PipelineOutput(
        recommendation_results=response["recommendation_results"],
        audit_trail=response["audit_trail"],
    )
