
from typing_extensions import Literal

from langsmith import traceable
from langgraph.graph import StateGraph, END

from neurons_agentic_workflow.creative_editor.models import (
    CreativeEditorInput,
    EditorInput,
    EditorOutput,
    GraphState,
    PlannerInput,
    SubTask,
)
from neurons_agentic_workflow.creative_editor.service.nodes import (
    critic_node,
    editor_node,
    planner_node,
    refiner_node,
)

MAX_ITERATIONS = 3

ApprovalResponse = Literal["Accepted", "Rejected"]
def _should_continue(state: GraphState) -> ApprovalResponse:
    """Evaluator/Optimizer routing: approved or max retries reached → END, else refine."""
    if state.approved or state.iteration >= MAX_ITERATIONS:
        return "Accepted"
    return "Rejected"


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("editor", editor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refiner", refiner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "editor")
    graph.add_edge("editor", "critic")
    graph.add_conditional_edges("critic", _should_continue, {"Accepted": END, "Rejected": "refiner"})
    graph.add_edge("refiner", "editor")
    return graph.compile()


_graph = _build_graph()


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
        sub_task=editor_input.sub_task,
    )
    result = await editor_node(state)
    return EditorOutput(edited_image=result["edited_image"])


@traceable(name="apply-recommendation")
async def apply_recommendation(input: CreativeEditorInput) -> EditorOutput:
    """Full pipeline via LangGraph: planner → editor → critic → [refiner → editor]* → END."""
    initial_state = GraphState(
        image=input.image,
        recommendation=input.recommendation,
        brand_guidelines=input.brand_guidelines,
    )
    final_state = await _graph.ainvoke(initial_state)
    return EditorOutput(edited_image=final_state["edited_image"])
