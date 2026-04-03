from typing import Any, Dict, List, Tuple

import networkx as nx

from app.schemas import PipeSegment


def _bbox_gap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return (dx * dx + dy * dy) ** 0.5


def _overlap_1d(a1, a2, b1, b2) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _orientation(seg: PipeSegment) -> str:
    x1, y1, x2, y2 = seg.bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if w > h * 1.5:
        return "horizontal"
    if h > w * 1.5:
        return "vertical"
    return "mixed"


def _segment_endpoints(seg: PipeSegment) -> List[Tuple[int, int]]:
    x1, y1, x2, y2 = seg.bbox
    orientation = _orientation(seg)
    if orientation == "horizontal":
        cy = int((y1 + y2) / 2)
        return [(x1, cy), (x2, cy)]
    if orientation == "vertical":
        cx = int((x1 + x2) / 2)
        return [(cx, y1), (cx, y2)]
    return [(x1, y1), (x2, y2)]


def _should_connect(a: PipeSegment, b: PipeSegment, threshold: float) -> bool:
    gap = _bbox_gap(a.bbox, b.bbox)
    if gap > threshold:
        return False

    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    overlap_x = _overlap_1d(ax1, ax2, bx1, bx2)
    overlap_y = _overlap_1d(ay1, ay2, by1, by2)
    a_orientation = _orientation(a)
    b_orientation = _orientation(b)

    if gap <= 3:
        return True
    if a_orientation == "horizontal" and b_orientation == "horizontal" and overlap_y > 0:
        return True
    if a_orientation == "vertical" and b_orientation == "vertical" and overlap_x > 0:
        return True
    if overlap_x > 0 and overlap_y > 0:
        return True
    return gap <= threshold * 0.5


def build_pipe_graph(pipe_segments: List[PipeSegment], threshold: float = 18.0) -> nx.Graph:
    graph = nx.Graph()

    for seg in pipe_segments:
        graph.add_node(
            seg.id,
            bbox=seg.bbox,
            endpoints=_segment_endpoints(seg),
            orientation=_orientation(seg),
        )

    for i in range(len(pipe_segments)):
        for j in range(i + 1, len(pipe_segments)):
            a = pipe_segments[i]
            b = pipe_segments[j]
            if _should_connect(a, b, threshold):
                graph.add_edge(a.id, b.id, gap=round(_bbox_gap(a.bbox, b.bbox), 2))

    return graph


def trace_connected_component(graph: nx.Graph, segment_id: str) -> Dict[str, Any]:
    if segment_id not in graph:
        return {"segment_ids": [], "endpoints": []}

    component = list(nx.node_connected_component(graph, segment_id))
    subgraph = graph.subgraph(component)

    endpoints = []
    for node_id in component:
        degree = subgraph.degree[node_id]
        if degree <= 1:
            node_data = graph.nodes[node_id]
            endpoints.extend(node_data.get("endpoints", []))

    if not endpoints:
        for node_id in component:
            endpoints.extend(graph.nodes[node_id].get("endpoints", []))

    deduped = []
    seen = set()
    for point in endpoints:
        if point not in seen:
            deduped.append(point)
            seen.add(point)

    return {
        "segment_ids": component,
        "endpoints": deduped[:8],
        "component_size": len(component),
    }
