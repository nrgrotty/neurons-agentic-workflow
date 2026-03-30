import base64
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import wrappers
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from neurons_agentic_workflow.creative_editor.models import (
    CriticOutput,
    EvaluationMetrics,
    GraphState,
    PlannerOutput,
    SubTask,
    EditorWorkerState,
)

OUTPUT_FOLDER = Path(__file__).parent.parent.parent.parent / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_ITERATIONS = 3

class _BestImage(BaseModel):
    index: int  # 0-based index into the candidates list

_planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(PlannerOutput)
_metrics_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(EvaluationMetrics)
_critic_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(CriticOutput)
_refiner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(SubTask)
_synthesizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(_BestImage)



async def planner_node(state: GraphState) -> dict:
    """Decompose the recommendation into one or more independent editing subtasks."""
    messages = [
        SystemMessage(content=(
            "You are a creative image editor planner. "
            "Decompose the recommendation into one or more independent, precise editing "
            "instructions, each targeting a distinct visual aspect. "
            "Each subtask must be self-contained and applicable to the original image independently. "
            "Strictly respect the brand guidelines — do NOT change any protected region, "
            "typography, aspect ratio, or brand element.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}"
        )),
        HumanMessage(content="Decompose the recommendation into editing subtasks."),
    ]
    result: PlannerOutput = await _planner_llm.ainvoke(messages)
    return {"subtasks": result.subtasks}


async def metrics_node(state: GraphState) -> dict:
    """Interpret the recommendation to derive named evaluation metrics for the synthesizer."""
    messages = [
        SystemMessage(content=(
            "You are a creative evaluation expert. "
            "Given a recommendation and brand guidelines, define a set of evaluation metrics "
            "that can be used to assess how well an edited image satisfies the recommendation. "
            "Each metric should have a clear name, a description of what to look for, "
            "and a weight (0.0–1.0) reflecting its relative importance. Weights should sum to 1.0.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}"
        )),
        HumanMessage(content="Define the evaluation metrics for assessing edited images."),
    ]
    result: EvaluationMetrics = await _metrics_llm.ainvoke(messages)
    return {"evaluation_metrics": result}


async def synthesizer_node(state: GraphState) -> dict:
    """Select the best edited image using the evaluation metrics derived from the recommendation."""
    if len(state.edited_images) == 1:
        return {"final_image": state.edited_images[0]}

    metrics_description = (
        state.evaluation_metrics.model_dump_json(indent=2)
        if state.evaluation_metrics
        else "No metrics available. Use your best judgement."
    )

    content: list = []
    for i, img_path in enumerate(state.edited_images):
        img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
        content.append({"type": "text", "text": f"Candidate {i} (subtask: {state.subtasks[i].description}):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    content.append({
        "type": "text",
        "text": (
            "Score each candidate against the following evaluation metrics and return the "
            "0-based index of the candidate with the highest weighted score.\n\n"
            f"Evaluation Metrics:\n{metrics_description}"
        ),
    })
    result: _BestImage = await _synthesizer_llm.ainvoke([HumanMessage(content=content)])
    best_index = max(0, min(result.index, len(state.edited_images) - 1))
    return {"final_image": state.edited_images[best_index]}



async def _editor(state: EditorWorkerState) -> dict:
    """Apply the subtask instruction to the image."""
    prompt_text = (
        "Edit this image following these instructions precisely:\n"
        f"{state.subtask.description}"
    )
    # gemini-2.5-flash-image is not yet available via langchain-google-genai
    gemini_client = genai.Client()
    image_bytes = Path(state.image).read_bytes()
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )
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

    output_path = OUTPUT_FOLDER / f"{Path(state.image).stem}_{state.recommendation.id}_subtask{state.subtask.index}.png"
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            output_path.write_bytes(part.inline_data.data)
            return {"edited_image": output_path}

    raise ValueError(
        f"Editor model did not return an image. "
        f"Response content: {response.candidates[0].content}"
    )


async def _critic(state: EditorWorkerState) -> dict:
    """Evaluate the edited image against the recommendation and brand guidelines."""
    image_b64 = base64.b64encode(Path(state.edited_image).read_bytes()).decode()
    messages = [
        SystemMessage(content=(
            "You are a creative quality critic. Evaluate whether the edited image "
            "correctly applies the recommendation while strictly respecting the brand guidelines.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}\n\n"
            f"Editing instruction applied: {state.subtask.description}"
        )),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": (
                "Does this edited image correctly apply the recommendation "
                "while respecting the brand guidelines? "
                "Set approval to true only if both conditions are fully met."
            )},
        ]),
    ]
    result: CriticOutput = await _critic_llm.ainvoke(messages)
    return {"approved": result.approval, "critic_feedback": result.feedback}


async def _refiner(state: EditorWorkerState) -> dict:
    """Produce a refined editing instruction based on critic feedback."""
    messages = [
        SystemMessage(content=(
            "You are a creative image editing optimizer. "
            "Refine the editing instruction to address the critic feedback "
            "while strictly respecting the brand guidelines.\n\n"
            f"Brand Guidelines:\n{state.brand_guidelines.model_dump_json(indent=2)}\n\n"
            f"Recommendation:\n{state.recommendation.model_dump_json(indent=2)}"
        )),
        HumanMessage(content=(
            f"Original instruction: {state.subtask.description}\n\n"
            f"Critic feedback: {state.critic_feedback}\n\n"
            "Produce a refined editing instruction that addresses the feedback."
        )),
    ]
    refined: SubTask = await _refiner_llm.ainvoke(messages)
    return {"sub_task": refined, "iteration": state.iteration + 1}


async def editor_worker_node(state: dict) -> dict:
    """Run the editor_worker subgraph for one subtask. Returns edited_images for fan-in."""
    final = await _editor_worker_subgraph.ainvoke(state)
    edited_image = final["edited_image"] if isinstance(final, dict) else final.edited_image
    return {"edited_images": [edited_image]}


def _should_continue(state: EditorWorkerState) -> Literal["Accepted", "Rejected"]:
    """Evaluator/Optimizer routing: approved or max iterations reached → END, else refine."""
    if state.approved or state.iteration >= MAX_ITERATIONS:
        return "Accepted"
    return "Rejected"


def _build_editor_worker_subgraph():
    graph = StateGraph(EditorWorkerState)
    graph.add_node("editor", _editor)
    graph.add_node("critic", _critic)
    graph.add_node("refiner", _refiner)
    graph.set_entry_point("editor")
    graph.add_edge("editor", "critic")
    graph.add_conditional_edges(
        "critic", _should_continue, {"Accepted": END, "Rejected": "refiner"}
    )
    graph.add_edge("refiner", "editor")
    return graph.compile()


_editor_worker_subgraph = _build_editor_worker_subgraph()

