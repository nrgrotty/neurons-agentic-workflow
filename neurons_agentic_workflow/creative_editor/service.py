import logging
from pathlib import Path

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    GraphState,
    PlannerInput,
    SubTask,
)

logger = logging.getLogger(__name__)


async def planner_node(state: GraphState) -> dict:
    """Plan a precise editing instruction from the recommendation and brand guidelines."""
    recommendation = state["recommendation"]
    brand_guidelines = state["brand_guidelines"]

    system_prompt = (
        "You are a creative image editor planner. "
        "Produce a single, precise editing instruction that applies the recommendation "
        "while strictly respecting the brand guidelines. "
        "Do NOT change any protected region, typography, aspect ratio, or brand element.\n\n"
        f"Brand Guidelines:\n{brand_guidelines.model_dump_json(indent=2)}\n\n"
        f"Recommendation:\n{recommendation.model_dump_json(indent=2)}"
    )
    user_prompt = "Create the image editing instruction."

    logger.info(
        "Planner node starting | recommendation_id=%s | system_prompt=%r | user_prompt=%r",
        recommendation.id,
        system_prompt,
        user_prompt,
    )

    client = genai.Client()
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_prompt)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=SubTask,
            ),
        )
    except Exception:
        logger.exception(
            "Planner node failed | recommendation_id=%s | system_prompt=%r | user_prompt=%r",
            recommendation.id,
            system_prompt,
            user_prompt,
        )
        raise

    sub_task = SubTask.model_validate_json(response.text)
    logger.info("Planner node succeeded | sub_task=%r", sub_task.description)
    return {"sub_task": sub_task}


async def editor_node(state: GraphState) -> dict:
    """Apply the sub_task editing instruction to the image."""
    image = state["image"]
    sub_task = state["sub_task"]

    prompt_text = (
        "Edit this image following these instructions precisely:\n"
        f"{sub_task.description}"
    )
    logger.info(
        "Editor node starting | image=%s | prompt=%r",
        image,
        prompt_text,
    )

    client = genai.Client()
    image_bytes = Path(image).read_bytes()

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
            "Editor node failed | image=%s | prompt=%r",
            image,
            prompt_text,
        )
        raise

    input_path = Path(image)
    output_path = input_path.with_stem(input_path.stem + "_edited")

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            logger.info("Editor node succeeded | output_image=%s", output_path)
            return {"edited_image": output_path}

    logger.error(
        "Editor node returned no image | image=%s | prompt=%r | response=%s",
        image,
        prompt_text,
        response.candidates[0].content,
    )
    raise ValueError(
        f"Editor model did not return an image in its response. "
        f"Response content: {response.candidates[0].content}"
    )


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("editor", editor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "editor")
    graph.add_edge("editor", END)
    return graph.compile()


_graph = _build_graph()


async def plan_creative_editions(planner_input: PlannerInput) -> SubTask:
    """Run only the planner node and return a SubTask."""
    state: GraphState = {
        "image": None,
        "recommendation": planner_input.recommendation,
        "brand_guidelines": planner_input.brand_guidelines,
        "sub_task": None,
        "edited_image": None,
    }
    result = await planner_node(state)
    return result["sub_task"]


async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Run only the editor node and return an EditorOutput."""
    state: GraphState = {
        "image": editor_input.image,
        "recommendation": None,
        "brand_guidelines": None,
        "sub_task": editor_input.sub_task,
        "edited_image": None,
    }
    result = await editor_node(state)
    return EditorOutput(edited_image=result["edited_image"])


async def apply_recommendation(input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline via LangGraph: planner → editor."""
    logger.info(
        "apply_recommendation pipeline starting | image=%s | recommendation_id=%s",
        input.image,
        input.recommendation.id,
    )
    initial_state: GraphState = {
        "image": input.image,
        "recommendation": input.recommendation,
        "brand_guidelines": input.brand_guidelines,
        "sub_task": None,
        "edited_image": None,
    }
    final_state = await _graph.ainvoke(initial_state)
    output = EditorOutput(edited_image=final_state["edited_image"])
    logger.info(
        "apply_recommendation pipeline complete | image=%s | output_image=%s",
        input.image,
        output.edited_image,
    )
    return output

