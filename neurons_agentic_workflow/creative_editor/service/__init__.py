
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    GraphState,
    PipelineState,
    PlannerInput,
    WorkerState,
)
from neurons_agentic_workflow.creative_editor.service.nodes import (
    metrics_node,
    planner_node,
    synthesizer_node,
    worker_node,
)


def _orchestrate(state: GraphState) -> list[Send]:
    """Fan out: send each subtask to a parallel worker."""
    return [
        Send("worker", WorkerState(
            image=state.image,
            recommendation=state.recommendation,
            brand_guidelines=state.brand_guidelines,
            subtask=sub_task,
            subtask_index=i,
        ).model_dump())
        for i, sub_task in enumerate(state.subtasks)
    ]


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge(START, "metrics")
    graph.add_conditional_edges("planner", _orchestrate, ["worker"])
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


_graph = _build_graph()

async def _recommendation_branch(state: GraphState) -> dict:
    """Run the full per-recommendation graph and return its final_image for fan-in."""
    final = await _graph.ainvoke(state)
    final_image = final["final_image"] if isinstance(final, dict) else final.final_image
    return {"final_images": [final_image]}


def _dispatch_recommendations(state: PipelineState) -> list[Send]:
    """Fan out: one Send per recommendation."""
    return [
        Send("recommendation_branch", GraphState(
            image=state.image,
            brand_guidelines=state.brand_guidelines,
            recommendation=rec,
        ).model_dump())
        for rec in state.recommendations
    ]


def _build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("recommendation_branch", _recommendation_branch)
    graph.add_conditional_edges(START, _dispatch_recommendations, ["recommendation_branch"])
    graph.add_edge("recommendation_branch", END)
    return graph.compile()


_pipeline = _build_pipeline()


async def plan_creative_editions(planner_input: PlannerInput) -> list:
    """Run the planner and return the list of subtasks."""
    state = GraphState(
        image=None,
        recommendation=planner_input.recommendation,
        brand_guidelines=planner_input.brand_guidelines,
    )
    result = await planner_node(state)
    return result["sub_tasks"]


async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Run a single editor→critic→refiner loop and return the edited image."""
    state = WorkerState(
        image=editor_input.image,
        recommendation=None,
        brand_guidelines=None,
        subtask=editor_input.subtask,
    )
    result = await worker_node(state.model_dump())
    return EditorOutput(edited_image=result["edited_images"][0])


@traceable(name="apply-recommendations")
async def apply_recommendation(input: CreativeEditorInput) -> list[EditorOutput]:
    """Full pipeline: parallel recommendation branches → each runs planner+workers+synthesizer."""
    initial_state = PipelineState(
        image=input.image,
        brand_guidelines=input.brand_guidelines,
        recommendations=input.recommendations,
    )
    final_state = await _pipeline.ainvoke(initial_state)
    final_images = final_state["final_images"] if isinstance(final_state, dict) else final_state.final_images
    return [EditorOutput(edited_image=img) for img in final_images]
