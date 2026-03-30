import tempfile
import traceback
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from langsmith import traceable
from pydantic import BaseModel, ValidationError

from neurons_agentic_workflow.creative_editor.models import (
    BrandGuidelines,
    PipelineInput,
    PipelineOutput,
    Recommendation,
)
from neurons_agentic_workflow.creative_editor.service import (
    run_pipeline,
)
from neurons_agentic_workflow.creative_editor.service.nodes import InvalidImageTypeError, MaxRetriesExceededError

router = APIRouter(prefix="/creative-editor", tags=["creative-editor"])


class PipelineFormInput(BaseModel):
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]


def _parse_pipeline_form_input(
    pipeline_input: str = Form(..., description='JSON with "brand_guidelines" and "recommendations" fields'),
) -> PipelineFormInput:
    try:
        return PipelineFormInput.model_validate_json(pipeline_input)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pipeline_input: {exc}\n\n{traceback.format_exc()}",
        ) from exc


async def _save_uploaded_image(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        return Path(tmp.name)


@router.post("/apply-recommendations", response_model=PipelineOutput)
@traceable(run_type="chain", name="apply_recommendations")
async def apply_recommendations(
    image: UploadFile,
    parsed_input: Annotated[PipelineFormInput, Depends(_parse_pipeline_form_input)],
) -> PipelineOutput:
    """Full pipeline: for each recommendation, run planner+editor_editor_workers+synthesizer in parallel."""
    image_path = await _save_uploaded_image(image)
    try:
        creative_editor_input = PipelineInput(
            image=image_path,
            brand_guidelines=parsed_input.brand_guidelines,
            recommendations=parsed_input.recommendations,
        )
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
    finally:
        image_path.unlink(missing_ok=True)

