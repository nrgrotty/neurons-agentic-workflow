"""Tests for the creative editor agents.

All tests call the real Gemini API — requires GOOGLE_API_KEY to be set.

Run with:
    pytest tests/test_editor.py -v -s
"""

from pathlib import Path
import shutil
import pytest

from neurons_agentic_workflow.creative_editor.models import (
    BrandGuidelines,
    EditorWorkerState,
    Recommendation,
    RecommendationType,
    SubTask,
)

from neurons_agentic_workflow.creative_editor.service.nodes import editor_worker_node

IMAGE_PATH = Path(__file__).parent.parent / "input" / "creative_1.png"

BRAND_GUIDELINES = BrandGuidelines(
    protected_regions=["Do not modify or remove the brand logo", "Do not alter the model's face"],
    typography="Maintain existing font style and hierarchy for all text elements",
    aspect_ratio="Maintain original aspect ratio (1572x1720)",
    brand_elements="Ensure logo remains visible and legible at all times",
)

RECOMMENDATION = Recommendation(
    id="rec_1",
    title="Strengthen Headline Impact",
    description=(
        "Add visual punch to the headline through enhanced color contrast, "
        "a soft gradient backdrop, or a geometric shape."
    ),
    type=RecommendationType.CONTRAST_SALIENCE,
)



@pytest.mark.asyncio
async def test_editor_worker_node_integration(tmp_path):
    """Run editor_worker_node against the real Gemini API for one subtask.

    Asserts that the node completes the editor→critic loop and returns
    an edited image together with an audit trail.
    Requires GOOGLE_API_KEY to be set and network access.
    """
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"

    input_image = tmp_path / IMAGE_PATH.name
    shutil.copy(IMAGE_PATH, input_image)

    state = EditorWorkerState(
        image=input_image,
        recommendation=RECOMMENDATION,
        brand_guidelines=BRAND_GUIDELINES,
        subtask=SubTask(description="Add a subtle blue tint to the background."),
    )

    result = await editor_worker_node(state.model_dump())

    assert "edited_images" in result
    assert len(result["edited_images"]) == 1
    
    edited_path = result["edited_images"][0]
    assert Path(edited_path).exists()
    assert len(result.get("audit_trail", [])) > 0
