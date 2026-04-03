import os
import re
from typing import Any, Dict, List, Tuple

from app.config import OUTPUT_DIR
from app.state import AgentState
from app.tools.equipment_detector import find_equipment_candidates
from app.tools.graph_builder import build_pipe_graph, trace_connected_component
from app.tools.ocr_tool import get_image_quality_summary, run_ocr
from app.tools.overlay_tool import save_overlay
from app.tools.pipe_segment_tool import extract_pipe_segments
from app.tools.target_pipe_selector import select_target_pipe_by_text
from app.tools.vlm_tool import VLMTool


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _point_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _looks_like_endpoint_label(text: str) -> bool:
    upper = text.upper().strip()
    normalized = re.sub(r"[^A-Z0-9]", "", upper)
    if "OFF-PAGE" in upper:
        return True
    return bool(
        re.search(r"\bE-\d{2,4}[A-Z]?\b", upper)
        or re.search(r"\bEA-\d{2,4}[A-Z]?\b", upper)
        or re.search(r"\bP-\d{2,4}[A-Z]?(?:/[A-Z])?\b", upper)
        or re.search(r"\bV-\d{2,4}[A-Z]?\b", upper)
        or re.fullmatch(r"E\d{2,4}[A-Z]?", normalized)
        or re.fullmatch(r"EA\d{2,4}[A-Z]?", normalized)
        or re.fullmatch(r"P\d{2,4}[A-Z]?", normalized)
        or re.fullmatch(r"V\d{2,4}[A-Z]?", normalized)
    )


def _nearest_endpoint_hints(state: AgentState, endpoints: List[Tuple[int, int]]) -> List[str]:
    hints = []

    for endpoint in endpoints[:4]:
        ranked = sorted(
            state.get("equipment_candidates", []),
            key=lambda item: _point_distance(endpoint, _bbox_center(item.bbox)),
        )
        for equipment in ranked[:3]:
            distance = round(_point_distance(endpoint, _bbox_center(equipment.bbox)), 1)
            hints.append(f"{equipment.name} ({equipment.equipment_type or 'equipment'}, dist={distance})")

        text_ranked = sorted(
            [item for item in state.get("ocr_texts", []) if _looks_like_endpoint_label(item.text)],
            key=lambda item: _point_distance(endpoint, _bbox_center(item.bbox)),
        )
        for item in text_ranked[:4]:
            distance = round(_point_distance(endpoint, _bbox_center(item.bbox)), 1)
            hints.append(f"{item.text} (ocr-label, dist={distance})")

    deduped = []
    seen = set()
    for hint in hints:
        if hint not in seen:
            deduped.append(hint)
            seen.add(hint)
    return deduped[:10]


def _build_io_metrics(state: AgentState) -> Dict[str, Any]:
    ocr_texts = state.get("ocr_texts", [])
    seed = state.get("selected_pipe_seed", {})
    quality = state.get("image_quality", {})
    return {
        "image_path": state.get("image_path", ""),
        "question": state.get("target_desc", ""),
        "target_text": state.get("target_text", ""),
        "ocr_text_count": len(ocr_texts),
        "ocr_avg_confidence": round(sum(item.score for item in ocr_texts) / len(ocr_texts), 4) if ocr_texts else 0.0,
        "ocr_max_confidence": round(max((item.score for item in ocr_texts), default=0.0), 4),
        "equipment_candidate_count": len(state.get("equipment_candidates", [])),
        "pipe_segment_count": len(state.get("pipe_segments", [])),
        "anchor_text": seed.get("anchor_text", ""),
        "anchor_score": round(float(seed.get("score", 0.0)), 4) if seed.get("score") is not None else 0.0,
        "anchor_match_type": seed.get("anchor_match_type", ""),
        "segment_score": round(float(seed.get("segment_score", 0.0)), 4)
        if seed.get("segment_score") is not None
        else 0.0,
        "input_quality": quality.get("input_quality", {}),
        "output_quality": quality.get("output_quality", {}),
    }


def ocr_node(state: AgentState) -> Dict[str, Any]:
    quality = get_image_quality_summary(state["image_path"])
    ocr_texts = run_ocr(state["image_path"])
    preview = [item.text for item in ocr_texts[:15]]
    avg_score = round(sum(item.score for item in ocr_texts) / len(ocr_texts), 4) if ocr_texts else 0.0
    max_score = round(max((item.score for item in ocr_texts), default=0.0), 4)
    return {
        "ocr_texts": ocr_texts,
        "image_quality": quality,
        "logs": state.get("logs", [])
        + [
            f"OCR complete: {len(ocr_texts)} texts",
            f"OCR confidence: avg={avg_score}, max={max_score}",
            f"Input quality: {quality.get('input_quality', {})}",
            f"Output quality: {quality.get('output_quality', {})}",
            f"OCR preview: {preview}",
        ],
    }


def equipment_node(state: AgentState) -> Dict[str, Any]:
    equipments = find_equipment_candidates(state["ocr_texts"])
    preview = [item.name for item in equipments[:10]]
    return {
        "equipment_candidates": equipments,
        "logs": state.get("logs", [])
        + [
            f"Equipment candidates: {len(equipments)}",
            f"Equipment preview: {preview}",
        ],
    }


def pipe_node(state: AgentState) -> Dict[str, Any]:
    segments = extract_pipe_segments(state["image_path"])
    return {
        "pipe_segments": segments,
        "logs": state.get("logs", []) + [f"Pipe segments: {len(segments)}"],
    }


def target_pipe_node(state: AgentState) -> Dict[str, Any]:
    selected = select_target_pipe_by_text(
        target_text=state["target_text"],
        ocr_texts=state["ocr_texts"],
        pipe_segments=state["pipe_segments"],
    )
    return {
        "selected_pipe_seed": selected,
        "logs": state.get("logs", []) + [f"Target pipe selected: {selected}"],
    }


def path_trace_node(state: AgentState) -> Dict[str, Any]:
    seed = state["selected_pipe_seed"]
    if not seed.get("success"):
        return {
            "best_path": {"segment_ids": [], "endpoints": [], "equipment_hints": []},
            "logs": state.get("logs", []) + ["No target pipe found"],
        }

    graph = build_pipe_graph(state["pipe_segments"])
    traced = trace_connected_component(graph, seed["segment_id"])
    traced["anchor_text"] = seed.get("anchor_text", "")
    traced["equipment_hints"] = _nearest_endpoint_hints(state, traced.get("endpoints", []))
    return {
        "best_path": traced,
        "logs": state.get("logs", []) + [f"Path trace complete: {len(traced['segment_ids'])} segments"],
    }


def overlay_node(state: AgentState) -> Dict[str, Any]:
    out_path = os.path.join(OUTPUT_DIR, "overlays", "overlay.png")
    save_overlay(
        image_path=state["image_path"],
        all_segments=state["pipe_segments"],
        target_segment_ids=state["best_path"]["segment_ids"],
        out_path=out_path,
    )
    return {
        "overlay_path": out_path,
        "logs": state.get("logs", []) + [f"Overlay saved: {out_path}"],
    }


def vlm_node(state: AgentState) -> Dict[str, Any]:
    seed = state.get("selected_pipe_seed", {})
    io_metrics = _build_io_metrics(state)

    if not seed.get("success"):
        target = state.get("target_text", "")
        answer = (
            f"제공해 주신 P&ID(배관 및 계장도)를 확인해 본 결과, 질문하신 배관 {target}은(는) "
            "현재 OCR 결과에서 정확히 식별되지 않아 시작과 끝 설비를 신뢰성 있게 판단할 수 없습니다."
        )
        result = {
            "FROM": "",
            "TO": "",
            "answer": answer,
            "confidence": 0.0,
            "reason": "Target pipe text was not detected in OCR results.",
            "evidence": {
                "direction_evidence": "",
                "notes": "VLM inference skipped because target pipe selection failed.",
                "io_metrics": io_metrics,
            },
        }
        return {
            "final_result": result,
            "logs": state.get("logs", []) + ["VLM skipped: target pipe selection failed"],
        }

    vlm = VLMTool()
    result = vlm.infer_from_to(
        original_image_path=state["image_path"],
        overlay_image_path=state["overlay_path"],
        target_desc=state["target_desc"],
        path_hints=state.get("best_path", {}),
        equipment_hints=state.get("best_path", {}).get("equipment_hints", []),
    )
    evidence = result.setdefault("evidence", {})
    evidence["io_metrics"] = io_metrics
    return {
        "final_result": result,
        "logs": state.get("logs", []) + [f"VLM inference complete: {result}"],
    }
