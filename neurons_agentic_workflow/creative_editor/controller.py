import traceback

import httpx
from fastapi import APIRouter, HTTPException
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from langsmith import traceable

from neurons_agentic_workflow.creative_editor.models import (
    PipelineInput,
    PipelineOutput,
)
from neurons_agentic_workflow.creative_editor.service import (
    run_pipeline,
)
from neurons_agentic_workflow.creative_editor.service.nodes import InvalidImageTypeError, MaxRetriesExceededError

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
            detail=f"Could not reach the Google Generative AI API. Check network connectivity and DNS resolution.\n\n{traceback.format_exc()}",
        ) from exc
    except ResourceExhausted as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Google Generative AI API rate limit exceeded. Please retry after a short delay.\n\n{traceback.format_exc()}",
        ) from exc
    except GoogleAPICallError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google Generative AI API error: {exc.message}\n\n{traceback.format_exc()}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found: {exc.filename}\n\n{traceback.format_exc()}",
        ) from exc
    except InvalidImageTypeError as exc:
        raise HTTPException(
            status_code=415,
            detail=f"{exc}\n\n{traceback.format_exc()}",
        ) from exc
    except MaxRetriesExceededError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{exc}\n\n{traceback.format_exc()}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc}\n\n{traceback.format_exc()}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the pipeline: {exc}\n\n{traceback.format_exc()}",
        ) from exc
