import httpx
from fastapi import APIRouter, HTTPException
from langsmith import traceable

from neurons_agentic_workflow.creative_editor.models import (
    PipelineInput,
    PipelineOutput,
)
from neurons_agentic_workflow.creative_editor.service import (
    run_pipeline,
)

router = APIRouter(prefix="/creative-editor", tags=["creative-editor"])


@router.post("/apply-recommendations", response_model=PipelineOutput)
@traceable(run_type="chain", name="apply_recommendations")
async def apply_recommendations(creative_editor_input: PipelineInput) -> PipelineOutput:
    """Full pipeline: for each recommendation, run planner+editor_editor_workers+synthesizer in parallel."""
    try:
        return await run_pipeline(creative_editor_input)
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the Google Generative AI API. Check network connectivity and DNS resolution.",
        ) from exc
