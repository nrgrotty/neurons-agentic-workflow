from fastapi import APIRouter

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorOutput,

)
from neurons_agentic_workflow.creative_editor.service import (
    apply_recommendation,
)

router = APIRouter(prefix="/creative-editor", tags=["creative-editor"])


@router.post("/apply-recommendations", response_model=EditorOutput)
async def apply_recommendations(creative_editor_input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline: plan edits with the planner agent, then apply them with the editor agent."""
    return await apply_recommendation(creative_editor_input)
