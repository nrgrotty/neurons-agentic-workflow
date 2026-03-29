
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.creative_editor.models import (
    EditorWorkerState,
    GraphState,
    SubTask,
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
        Send("worker", EditorWorkerState(
            image=state.image,
            recommendation=state.recommendation,
            brand_guidelines=state.brand_guidelines,
            subtask=SubTask(description=sub_task.description, index=i),
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


main_graph = _build_graph()