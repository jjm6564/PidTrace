from app.schemas import PipeSegment
from tests.conftest import reload_module


def test_build_pipe_graph_connects_nearby_segments():
    graph_builder = reload_module("app.tools.graph_builder")
    segments = [
        PipeSegment(id="seg_1", points=[(0, 0)], bbox=(0, 0, 10, 10)),
        PipeSegment(id="seg_2", points=[(0, 0)], bbox=(20, 0, 30, 10)),
        PipeSegment(id="seg_3", points=[(0, 0)], bbox=(200, 200, 220, 220)),
    ]

    graph = graph_builder.build_pipe_graph(segments, threshold=25.0)
    traced = graph_builder.trace_connected_component(graph, "seg_1")

    assert set(traced["segment_ids"]) == {"seg_1", "seg_2"}
    assert set(graph_builder.trace_connected_component(graph, "seg_3")["segment_ids"]) == {"seg_3"}


def test_trace_connected_component_returns_empty_for_unknown_segment():
    graph_builder = reload_module("app.tools.graph_builder")
    graph = graph_builder.build_pipe_graph([])

    assert graph_builder.trace_connected_component(graph, "missing") == {"segment_ids": [], "endpoints": []}
