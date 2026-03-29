from fastapi import APIRouter

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    PlannerInput,
    SubTask,
)
from neurons_agentic_workflow.creative_editor.service import (
    apply_recommendation,
    edit_creative,
    plan_creative_editions,
)

router = APIRouter(prefix="/creative-editor", tags=["creative-editor"])


@router.post("/plan", response_model=SubTask)
async def plan(planner_input: PlannerInput) -> SubTask:
    """Run the planner agent: returns an editing SubTask for the given image and recommendation."""
    return await plan_creative_editions(planner_input)


@router.post("/edit", response_model=EditorOutput)
async def edit(editor_input: EditorInput) -> EditorOutput:
    """Run the editor agent: applies a SubTask to the image and returns the edited image."""
    return await edit_creative(editor_input)


@router.post("/apply-recommendations", response_model=EditorOutput)
async def apply_recommendations(creative_editor_input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline: plan edits with the planner agent, then apply them with the editor agent."""
    return await apply_recommendation(creative_editor_input)
