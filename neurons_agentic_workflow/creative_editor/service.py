from pathlib import Path

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    GraphState,
    PlannerInput,
    SubTask,
)


_planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(SubTask)

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

def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("editor", editor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "editor")
    graph.add_edge("editor", END)
    return graph.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def plan_creative_editions(planner_input: PlannerInput) -> SubTask:
    """Run only the planner node and return a SubTask."""
    state = GraphState(
        recommendation=planner_input.recommendation,
        brand_guidelines=planner_input.brand_guidelines,
    )
    result = await planner_node(state)
    return result["sub_task"]


async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Run only the editor node and return an EditorOutput."""
    state = GraphState(
        image=editor_input.image,
        sub_task=editor_input.subtask,
    )
    result = await editor_node(state)
    return EditorOutput(edited_image=result["edited_image"])


async def apply_recommendation(input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline via LangGraph: planner → editor."""
    initial_state = GraphState(
        image=input.image,
        recommendation=input.recommendation,
        brand_guidelines=input.brand_guidelines,
    )
    final_state = await _graph.ainvoke(initial_state)
    return EditorOutput(edited_image=final_state["edited_image"])


