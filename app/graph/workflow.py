from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    equipment_node,
    ocr_node,
    overlay_node,
    path_trace_node,
    pipe_node,
    target_pipe_node,
    vlm_node,
)
from app.state import AgentState


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("equipment", equipment_node)
    graph.add_node("pipe", pipe_node)
    graph.add_node("target_pipe", target_pipe_node)
    graph.add_node("path_trace", path_trace_node)
    graph.add_node("overlay", overlay_node)
    graph.add_node("vlm", vlm_node)

    graph.add_edge(START, "ocr")
    graph.add_edge("ocr", "equipment")
    graph.add_edge("equipment", "pipe")
    graph.add_edge("pipe", "target_pipe")
    graph.add_edge("target_pipe", "path_trace")
    graph.add_edge("path_trace", "overlay")
    graph.add_edge("overlay", "vlm")
    graph.add_edge("vlm", END)

    return graph.compile()
