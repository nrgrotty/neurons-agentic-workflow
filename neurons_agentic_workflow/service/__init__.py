
"""Service entry points for the creative editor workflow."""
from langsmith import traceable

from neurons_agentic_workflow.models import (
    PipelineInput,
    PipelineOutput,
    PipelineState,
)
from neurons_agentic_workflow.service.pipeline import pipeline



@traceable(run_type="chain", name="run_pipeline")
async def run_pipeline(input: PipelineInput) -> PipelineOutput:
    """Full pipeline: parallel recommendation branches → each runs planner+editor_workers+evaluator."""
    initial_state = PipelineState(
        image=input.image,
        brand_guidelines=input.brand_guidelines,
        recommendations=input.recommendations,
    )
    final_state = await pipeline.ainvoke(initial_state)
    return PipelineOutput(
        recommendation_results=final_state["recommendation_results"],
        audit_trail=final_state.get("audit_trail", []),
    )
