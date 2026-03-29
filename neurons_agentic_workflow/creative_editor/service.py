import logging
from pathlib import Path

from pydantic_ai import Agent, RunContext
from google import genai
from google.genai import types

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    PlannerInput,
    SubTask,
)

logger = logging.getLogger(__name__)


planner_agent = Agent(
    "google-gla:gemini-2.5-flash",
    deps_type=PlannerInput,
    output_type=SubTask,
    output_retries=3,
)


@planner_agent.system_prompt
async def planner_system_prompt(ctx: RunContext[PlannerInput]) -> str:
    return (
        "You are a creative image editor planner. "
        "Produce a single, precise editing instruction that applies the recommendation "
        "while strictly respecting the brand guidelines. "
        "Do NOT change any protected region, typography, aspect ratio, or brand element.\n\n"
        f"Brand Guidelines:\n{ctx.deps.brand_guidelines.model_dump_json(indent=2)}\n\n"
        f"Recommendation:\n{ctx.deps.recommendation.model_dump_json(indent=2)}"
    )


async def plan_creative_editions(planner_input: PlannerInput) -> SubTask:
    """Run the planner agent and return a SubTask with the editing instruction."""
    logger.info(
        "Planner agent starting | recommendation_id=%s | recommendation=%s | brand_guidelines=%s",
        planner_input.recommendation.id,
        planner_input.recommendation.model_dump(),
        planner_input.brand_guidelines.model_dump(),
    )
    try:
        result = await planner_agent.run(
            "Create the image editing instruction.",
            deps=planner_input,
        )
    except Exception:
        logger.exception(
            "Planner agent failed | recommendation_id=%s | recommendation=%s | brand_guidelines=%s",
            planner_input.recommendation.id,
            planner_input.recommendation.model_dump(),
            planner_input.brand_guidelines.model_dump(),
        )
        raise
    logger.info("Planner agent succeeded | sub_task=%r", result.output.description)
    return result.output

async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Apply the SubTask editing instruction to the image using Gemini image generation."""
    prompt_text = (
        "Edit this image following these instructions precisely:\n"
        f"{editor_input.sub_task.description}"
    )
    logger.info(
        "Editor model starting | image=%s | prompt=%r",
        editor_input.image,
        prompt_text,
    )
    client = genai.Client()
    image_bytes = Path(editor_input.image).read_bytes()

    try:
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
                        types.Part(text=prompt_text),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
    except Exception:
        logger.exception(
            "Editor model call failed | image=%s | prompt=%r",
            editor_input.image,
            prompt_text,
        )
        raise

    input_path = Path(editor_input.image)
    output_path = input_path.with_stem(input_path.stem + "_edited")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            logger.info("Editor model succeeded | output_image=%s", output_path)
            return EditorOutput(edited_image=output_path)

    logger.error(
        "Editor model returned no image | image=%s | prompt=%r | response=%s",
        editor_input.image,
        prompt_text,
        response.candidates[0].content,
    )
    raise ValueError(f"Editor model did not return an image in its response. Response content: {response.candidates[0].content}")


async def apply_recommendation(input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline: plan edits → apply edits → return edited image."""
    logger.info(
        "apply_recommendation pipeline starting | image=%s | recommendation_id=%s",
        input.image,
        input.recommendation.id,
    )
    planner_input = PlannerInput(
        brand_guidelines=input.brand_guidelines,
        recommendation=input.recommendation,
    )
    sub_task = await plan_creative_editions(planner_input)
    editor_input = EditorInput(image=input.image, sub_task=sub_task)
    output = await edit_creative(editor_input)
    logger.info(
        "apply_recommendation pipeline complete | image=%s | output_image=%s",
        input.image,
        output.edited_image,
    )
    return output
