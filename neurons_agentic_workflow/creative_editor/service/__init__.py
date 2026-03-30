
"""Service entry points for the creative editor workflow."""
from langsmith import traceable

from neurons_agentic_workflow.creative_editor.models import (
    PipelineInput,
    EditorInput,
    EditorOutput,
    PipelineOutput,
    PipelineState,
    EditorWorkerState,
)
from neurons_agentic_workflow.creative_editor.service.pipeline import pipeline
from neurons_agentic_workflow.creative_editor.service.nodes import (
    editor_worker_node,
)




@traceable(run_type="chain", name="edit_creative")
async def edit_creative(editor_input: EditorInput) -> EditorOutput:
    """Run a single editor→critic→refiner loop and return the edited image."""
    state = EditorWorkerState(
        image=editor_input.image,
        recommendation=None,
        brand_guidelines=None,
        subtask=editor_input.subtask,
    )
    result = await editor_worker_node(state.model_dump())
    return EditorOutput(edited_image=result["edited_images"][0])


@traceable(run_type="chain", name="run_pipeline")
async def run_pipeline(input: PipelineInput) -> PipelineOutput:
    """Full pipeline: parallel recommendation branches → each runs planner+editor_workers+synthesizer."""
    initial_state = PipelineState(
        image=input.image,
        brand_guidelines=input.brand_guidelines,
        recommendations=input.recommendations,
    )
    final_state = await pipeline.ainvoke(initial_state)
    final_images = final_state["final_images"]
    audit_trail = final_state.get("audit_trail", [])
    return PipelineOutput(
        final_images=[EditorOutput(edited_image=img) for img in final_images],
        audit_trail=audit_trail,
    )
