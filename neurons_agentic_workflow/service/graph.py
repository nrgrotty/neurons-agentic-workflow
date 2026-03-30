
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from neurons_agentic_workflow.models import (
    EditorWorkerState,
    GraphState,
)
from neurons_agentic_workflow.service.nodes import (
    evaluation_planner_node,
    editor_planner_node,
    synthesizer_node,
    editor_worker_node,
)

NUM_VARIANTS = 3

def _orchestrate(state: GraphState) -> list[Send]:
    """Fan out: send the same editing description to N parallel editor_workers as distinct variants."""
    return [
        Send("editor_worker", EditorWorkerState(
            image=state.image,
            recommendation=state.recommendation,
            brand_guidelines=state.brand_guidelines,
            editing_instructions=state.editing_instructions,
            variant_index=i,
        ).model_dump())
        for i in range(NUM_VARIANTS)
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