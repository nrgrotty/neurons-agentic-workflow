
from langsmith import traceable

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    GraphState,
    PipelineState,
    PlannerInput,
    EditorWorkerState,
)
from neurons_agentic_workflow.creative_editor.service.pipeline import pipeline
from neurons_agentic_workflow.creative_editor.service.nodes import (
    planner_node,
    worker_node,
)



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
    state = EditorWorkerState(
        image=editor_input.image,
        recommendation=None,
        brand_guidelines=None,
        subtask=editor_input.subtask,
    )
    result = await worker_node(state.model_dump())
    return EditorOutput(edited_image=result["edited_images"][0])


@traceable(name="apply-recommendations")
async def run_pipeline(input: CreativeEditorInput) -> list[EditorOutput]:
    """Full pipeline: parallel recommendation branches → each runs planner+workers+synthesizer."""
    initial_state = PipelineState(
        image=input.image,
        brand_guidelines=input.brand_guidelines,
        recommendations=input.recommendations,
    )
    final_state = await pipeline.ainvoke(initial_state)
    final_images = final_state["final_images"] if isinstance(final_state, dict) else final_state.final_images
    return [EditorOutput(edited_image=img) for img in final_images]
