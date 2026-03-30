
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.creative_editor.models import (
    EditorWorkerState,
    GraphState,
    SubTask,
)
from neurons_agentic_workflow.creative_editor.service.nodes import (
    evaluation_planner_node,
    editor_planner_node,
    synthesizer_node,
    editor_worker_node,
)

def _orchestrate(state: GraphState) -> list[Send]:
    """Fan out: send each subtask to a parallel editor_worker."""
    return [
        Send("editor_worker", EditorWorkerState(
            image=state.image,
            recommendation=state.recommendation,
            brand_guidelines=state.brand_guidelines,
            subtask=SubTask(description=sub_task.description, index=i),
        ).model_dump())
        for i, sub_task in enumerate(state.subtasks)
    ]


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("editor_planner", editor_planner_node)
    graph.add_node("evaluation_planner", evaluation_planner_node)
    graph.add_node("editor_worker", editor_worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "editor_planner")
    graph.add_edge(START, "evaluation_planner")
    graph.add_conditional_edges("editor_planner", _orchestrate, ["editor_worker"])
    graph.add_edge("editor_worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


main_graph = _build_graph()