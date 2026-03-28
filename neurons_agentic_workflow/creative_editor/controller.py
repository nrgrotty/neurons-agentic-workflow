from fastapi import APIRouter

from neurons_agentic_workflow.creative_editor.models import CreativeFeedback
from neurons_agentic_workflow.creative_editor.service import plan_editions

router = APIRouter()


@router.post("/apply-recommendations")
async def apply_feedback(creative_feedback: CreativeFeedback):
    return await plan_editions(creative_feedback)
