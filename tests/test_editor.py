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
    EditorInput,
    Recommendation,
    RecommendationType,
    SubTask,
)
from google import genai
from google.genai import types

from neurons_agentic_workflow.creative_editor.service import edit_creative

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
async def test_edit_creative_raw_response(tmp_path):
    """Print raw response parts from gemini-2.5-flash-image to debug missing image output."""
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"

    image_bytes = IMAGE_PATH.read_bytes()
    client = genai.Client()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
                    types.Part(text="Add a subtle blue tint to the background."),
                ]
            )
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    print(f"\nFinish reason: {response.candidates[0].finish_reason}")
    for i, part in enumerate(response.candidates[0].content.parts):
        if part.inline_data is not None:
            print(f"  part[{i}]: IMAGE mime={part.inline_data.mime_type} size={len(part.inline_data.data)}")
        else:
            print(f"  part[{i}]: TEXT {repr(part.text)[:300]}")


@pytest.mark.asyncio
async def test_edit_creative_integration(tmp_path):
    """Hit the real Gemini API and assert an image is returned.

    Requires GOOGLE_API_KEY to be set and network access.
    """
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"

    input_image = tmp_path / IMAGE_PATH.name
    shutil.copy(IMAGE_PATH, input_image)

    editor_input = EditorInput(
        image=input_image,
        subtask=SubTask(description="Add a subtle blue tint to the background."),
    )

    result = await edit_creative(editor_input)

    assert result.edited_image.exists()
    assert result.edited_image.stat().st_size > 0
    print(f"\nEdited image saved to: {result.edited_image}")
