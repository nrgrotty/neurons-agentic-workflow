import io
import json
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from langsmith import traceable
from pydantic import ValidationError

from neurons_agentic_workflow.models import (
    BrandGuidelines,
    PipelineInput,
    PipelineOutput,
    Recommendation,
)
from neurons_agentic_workflow.service import (
    run_pipeline,
)
from neurons_agentic_workflow.service.nodes import InvalidImageTypeError, MaxRetriesExceededError

router = APIRouter(prefix="/creative-editor", tags=["creative-editor"])


def _parse_recommendations(
    recommendations: Annotated[
        list[str],
        Form(
            description='Repeat this field for each recommendation. Each value is a JSON object: {"id": "rec_1", "title": "...", "description": "...", "type": "colour_mood|copy_messaging|contrast_salience|composition"}',
        ),
    ],
) -> list[Recommendation]:
    try:
        return [Recommendation.model_validate(json.loads(r)) for r in recommendations]
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid recommendations: {exc}\n\n{traceback.format_exc()}",
        ) from exc


async def _save_uploaded_image(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        return Path(tmp.name)


def _build_zip(output: PipelineOutput) -> io.BytesIO:
    """Pack all edited images and the audit trail into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, editor_output in enumerate(output.final_images):
            img_path = Path(editor_output.edited_image)
            if img_path.exists():
                zf.write(img_path, arcname=img_path.name)
        audit_json = json.dumps(
            [e.model_dump(mode="json") for e in output.audit_trail], indent=2
        )
        zf.writestr("audit_trail.json", audit_json)
    buf.seek(0)
    return buf

@traceable(run_type="chain", name="apply_recommendations")
async def _apply_recommendations(
    image: UploadFile,
    protected_regions: Annotated[list[str], Form(description="Regions that must not be modified (repeat field for multiple values)")],
    typography: Annotated[str, Form(description="Typography rules to maintain")],
    aspect_ratio: Annotated[str, Form(description="Aspect ratio constraint, e.g. '1572x1720'")],
    brand_elements: Annotated[str, Form(description="Brand elements that must remain visible")],
    recommendations: Annotated[list[Recommendation], Depends(_parse_recommendations)],
) -> StreamingResponse:
    """Core logic of the apply-recommendations endpoint, without FastAPI-specific response handling."""
    image_path = await _save_uploaded_image(image)
    try:
        creative_editor_input = PipelineInput(
            image=image_path,
            brand_guidelines=BrandGuidelines(
                protected_regions=protected_regions,
                typography=typography,
                aspect_ratio=aspect_ratio,
                brand_elements=brand_elements,
            ),
            recommendations=recommendations,
        )
        output = await run_pipeline(creative_editor_input)
        zip_buf = _build_zip(output)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=edited_creatives.zip"},
        )
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

@router.post("/apply-recommendations")
async def apply_recommendations(
    image: UploadFile,
    protected_regions: Annotated[list[str], Form(description="Regions that must not be modified (repeat field for multiple values)")],
    typography: Annotated[str, Form(description="Typography rules to maintain")],
    aspect_ratio: Annotated[str, Form(description="Aspect ratio constraint, e.g. '1572x1720'")],
    brand_elements: Annotated[str, Form(description="Brand elements that must remain visible")],
    recommendations: Annotated[list[Recommendation], Depends(_parse_recommendations)],
) -> StreamingResponse:
    """Full pipeline: returns a ZIP containing one edited image per recommendation plus the audit trail."""
    return await _apply_recommendations(
        image=image,
        protected_regions=protected_regions,
        typography=typography,
        aspect_ratio=aspect_ratio,
        brand_elements=brand_elements,
        recommendations=recommendations,
    )
