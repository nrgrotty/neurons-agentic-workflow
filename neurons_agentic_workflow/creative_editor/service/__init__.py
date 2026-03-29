
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
from neurons_agentic_workflow.creative_editor.service.nodes import editor_node, planner_node

def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("editor", editor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "editor")
    graph.add_edge("editor", END)
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
    """Full pipeline via LangGraph: planner → editor."""
    initial_state = GraphState(
        image=input.image,
        recommendation=input.recommendation,
        brand_guidelines=input.brand_guidelines,
    )
    final_state = await _graph.ainvoke(initial_state)
    return EditorOutput(edited_image=final_state["edited_image"])
