import base64
from pathlib import Path

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from neurons_agentic_workflow.creative_editor.models import (
    CriticOutput,
    GraphState,
    SubTask,
)


_planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(SubTask)
_critic_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(CriticOutput)
_refiner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(SubTask)


async def planner_node(state: GraphState) -> dict:
    """Plan a precise editing instruction from the recommendation and brand guidelines."""
    messages = [
        SystemMessage(content=(
            "You are a creative image editor planner. "
            "Produce a single, precise editing instruction that applies the recommendation "
            "while strictly respecting the brand guidelines. "
            "Do NOT change any protected region, typography, aspect ratio, or brand element.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}"
        )),
        HumanMessage(content="Create the image editing instruction."),
    ]
    sub_task: SubTask = await _planner_llm.ainvoke(messages)
    return {"sub_task": sub_task}


@traceable(name="editor-node", run_type="llm")
async def editor_node(state: GraphState) -> dict:
    """Apply the sub_task editing instruction to the image."""
    prompt_text = (
        "Edit this image following these instructions precisely:\n"
        f"{state.sub_task.description}"
    )
    # gemini-2.5-flash-image is an image-generation model not yet available via
    # langchain-google-genai, so we keep the raw genai.Client() call here.
    client = genai.Client()
    image_bytes = Path(state.image).read_bytes()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
                    types.Part(text=prompt_text),
                ]
            )
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    output_path = Path(state.image).with_stem(Path(state.image).stem + "_edited")
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            return {"edited_image": output_path}

    raise ValueError(
        f"Editor model did not return an image in its response. "
        f"Response content: {response.candidates[0].content}"
    )


async def critic_node(state: GraphState) -> dict:
    """Evaluate the edited image against the recommendation and brand guidelines."""
    image_b64 = base64.b64encode(Path(state.edited_image).read_bytes()).decode()
    messages = [
        SystemMessage(content=(
            "You are a creative quality critic. Evaluate whether the edited image "
            "correctly applies the recommendation while strictly respecting the brand guidelines. "
            "Be specific about what works and what does not.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}\n\n"
            f"Editing instruction that was applied: {state.sub_task.description}"
        )),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
            {
                "type": "text",
                "text": (
                    "Does this edited image correctly apply the recommendation "
                    "while respecting the brand guidelines? "
                    "Set approval to true only if both conditions are fully met."
                ),
            },
        ]),
    ]
    result: CriticOutput = await _critic_llm.ainvoke(messages)
    return {"approved": result.approval, "critic_feedback": result.feedback}


async def refiner_node(state: GraphState) -> dict:
    """Produce a refined editing instruction based on critic feedback."""
    messages = [
        SystemMessage(content=(
            "You are a creative image editing optimizer. "
            "Given the original editing instruction and critic feedback, "
            "produce a refined instruction that addresses the issues raised "
            "while strictly respecting the brand guidelines.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}"
        )),
        HumanMessage(content=(
            f"Original instruction: {state.sub_task.description}\n\n"
            f"Critic feedback: {state.critic_feedback}\n\n"
            "Produce a refined editing instruction that addresses the feedback."
        )),
    ]
    refined: SubTask = await _refiner_llm.ainvoke(messages)
    return {"sub_task": refined, "iteration": state.iteration + 1}

