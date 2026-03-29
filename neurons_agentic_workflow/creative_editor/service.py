from pathlib import Path

from pydantic_ai import Agent
from google import genai
from google.genai import types

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    PlannerInput,
    SubTask,
)


planner_agent = Agent(
    "google-gla:gemini-2.5-flash",
    deps_type=PlannerInput,
    output_type=SubTask,
    output_retries=3,
    system_prompt=(
        "You are a creative image editor planner. "
        "Based on the brand guidelines and recommendation, produce a single, precise editing instruction "
        "that applies the recommendation while strictly respecting the brand guidelines. "
        "Do NOT change any protected region, typography, aspect ratio, or brand element."
    ),
)


async def plan_creative_editions(planner_input: PlannerInput) -> SubTask:
    """Run the planner agent and return a SubTask with the editing instruction."""
    result = await planner_agent.run(
        "Create the image editing instruction.",
        deps=planner_input,
    )
    return result.output

async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Apply the SubTask editing instruction to the image using Gemini image generation."""
    client = genai.Client()
    image_bytes = Path(editor_input.image).read_bytes()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=image_bytes,
                        )
                    ),
                    types.Part(
                        text=(
                            "Edit this image following these instructions precisely:\n"
                            f"{editor_input.sub_task.description}"
                        )
                    ),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    input_path = Path(editor_input.image)
    output_path = input_path.with_stem(input_path.stem + "_edited")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            return EditorOutput(edited_image=output_path)

    raise ValueError(f"Editor model did not return an image in its response. Response content: {response.candidates[0].content}")


async def apply_recommendation(input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline: plan edits → apply edits → return edited image."""
    planner_input = PlannerInput(
        brand_guidelines=input.brand_guidelines,
        recommendation=input.recommendation,
    )
    sub_task = await plan_creative_editions(planner_input)
    editor_input = EditorInput(image=input.image, sub_task=sub_task)
    return await edit_creative(editor_input)
