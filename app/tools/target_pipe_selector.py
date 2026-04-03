import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from app.schemas import OCRText, PipeSegment


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _expand_bbox(bbox, margin: int):
    x1, y1, x2, y2 = bbox
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _bbox_contains_point(bbox, point: Tuple[int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def _bbox_gap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return (dx * dx + dy * dy) ** 0.5


def _bbox_area(bbox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _segment_orientation(seg: PipeSegment) -> str:
    x1, y1, x2, y2 = seg.bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if w > h * 1.5:
        return "horizontal"
    if h > w * 1.5:
        return "vertical"
    return "mixed"


def _normalize_pipe_text(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
    normalized = normalized.replace("O", "0")
    normalized = normalized.replace("I", "1")
    normalized = normalized.replace("L", "1")
    return normalized


def _tokenize_pipe_text(text: str) -> List[str]:
    return [token for token in re.split(r"[^A-Z0-9]+", text.upper()) if token]


def _normalize_pipe_token(token: str) -> str:
    return (
        token.upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
    )


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_similarity(a: str, b: str) -> float:
    return _string_similarity(_normalize_pipe_token(a), _normalize_pipe_token(b))


def _token_hit_score(candidate_tokens: List[str], target_tokens: List[str]) -> Tuple[List[str], float]:
    matched_tokens = []
    score = 0.0

    for target in target_tokens:
        best = 0.0
        best_token = None
        for candidate in candidate_tokens:
            similarity = _token_similarity(candidate, target)
            if similarity > best:
                best = similarity
                best_token = candidate
        if best >= 0.72:
            matched_tokens.append(target)
            score += best

    return matched_tokens, score


def _merge_bbox(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _find_target_anchor(target_text: str, ocr_texts: List[OCRText]) -> Dict[str, Any]:
    normalized_target = _normalize_pipe_text(target_text)
    target_tokens = _tokenize_pipe_text(target_text)

    direct = [item for item in ocr_texts if normalized_target and normalized_target in _normalize_pipe_text(item.text)]
    if direct:
        best = max(direct, key=lambda item: item.score)
        return {
            "success": True,
            "text": best.text,
            "bbox": best.bbox,
            "score": best.score,
            "match_type": "direct",
            "matched_tokens": target_tokens,
        }

    candidates = []

    for item in ocr_texts:
        normalized = _normalize_pipe_text(item.text)
        tokens = _tokenize_pipe_text(item.text)
        token_hits, token_score = _token_hit_score(tokens, target_tokens)
        if not token_hits:
            continue

        coverage = len(token_hits) / max(1, len(target_tokens))
        char_overlap = _string_similarity(normalized, normalized_target)
        score = coverage * 2.0 + char_overlap * 0.7 + token_score * 0.4 + float(item.score) * 0.2
        candidates.append(
            {
                "items": [item],
                "bbox": item.bbox,
                "text": item.text,
                "score": score,
                "matched_tokens": token_hits,
                "match_type": "partial",
            }
        )

    for i, left in enumerate(ocr_texts):
        left_center = _bbox_center(left.bbox)
        left_tokens = _tokenize_pipe_text(left.text)
        if not any(token in target_tokens for token in left_tokens):
            continue

        for j, right in enumerate(ocr_texts):
            if i == j:
                continue

            right_center = _bbox_center(right.bbox)
            if abs(left_center[1] - right_center[1]) > 30:
                continue
            if abs(left_center[0] - right_center[0]) > 260:
                continue

            combo_tokens = _tokenize_pipe_text(left.text) + _tokenize_pipe_text(right.text)
            token_hits, token_score = _token_hit_score(combo_tokens, target_tokens)
            if len(token_hits) < 2:
                continue

            combo_text = f"{left.text} {right.text}"
            normalized_combo = _normalize_pipe_text(combo_text)
            coverage = len(token_hits) / max(1, len(target_tokens))
            char_overlap = _string_similarity(normalized_combo, normalized_target)
            score = coverage * 2.4 + char_overlap * 0.8 + token_score * 0.5 + (float(left.score) + float(right.score)) * 0.15
            candidates.append(
                {
                    "items": [left, right],
                    "bbox": _merge_bbox([left.bbox, right.bbox]),
                    "text": combo_text,
                    "score": score,
                    "matched_tokens": token_hits,
                    "match_type": "merged",
                }
            )

    if not candidates:
        return {"success": False, "reason": f"Target text '{target_text}' not found"}

    best = max(candidates, key=lambda item: (len(item["matched_tokens"]), item["score"]))
    minimum_hits = max(2, min(3, len(target_tokens)))
    if len(best["matched_tokens"]) < minimum_hits and best["score"] < 2.2:
        return {"success": False, "reason": f"Target text '{target_text}' not found"}

    return {
        "success": True,
        "text": best["text"],
        "bbox": best["bbox"],
        "score": best["score"],
        "match_type": best["match_type"],
        "matched_tokens": best["matched_tokens"],
    }


def select_target_pipe_by_text(
    target_text: str,
    ocr_texts: List[OCRText],
    pipe_segments: List[PipeSegment],
) -> Dict[str, Any]:
    anchor_match = _find_target_anchor(target_text, ocr_texts)
    if not anchor_match.get("success"):
        return anchor_match
    if not pipe_segments:
        return {"success": False, "reason": "No pipe segments extracted"}

    anchor_text = anchor_match["text"]
    anchor_bbox = anchor_match["bbox"]
    anchor_center = _bbox_center(anchor_bbox)
    expanded_anchor = _expand_bbox(anchor_bbox, 30)

    scored_segments = []
    for seg in pipe_segments:
        center_inside = _bbox_contains_point(seg.bbox, anchor_center)
        bbox_gap = _bbox_gap(expanded_anchor, seg.bbox)
        area_penalty = min(_bbox_area(seg.bbox), 5000) / 5000.0
        orientation_bonus = 0.2 if _segment_orientation(seg) != "mixed" else 0.0

        score = 0.0
        if center_inside:
            score += 3.0
        if bbox_gap == 0:
            score += 2.0
        score += max(0.0, 1.5 - (bbox_gap / 20.0))
        score += orientation_bonus
        score -= area_penalty * 0.35

        scored_segments.append(
            {
                "segment": seg,
                "score": score,
                "gap": bbox_gap,
                "center_inside": center_inside,
            }
        )

    scored_segments.sort(key=lambda item: item["score"], reverse=True)
    best = scored_segments[0]
    best_seg = best["segment"]

    return {
        "success": True,
        "anchor_text": anchor_text,
        "anchor_bbox": anchor_bbox,
        "anchor_center": anchor_center,
        "segment_id": best_seg.id,
        "segment_bbox": best_seg.bbox,
        "segment_score": round(best["score"], 4),
        "anchor_match_type": anchor_match.get("match_type", "direct"),
        "matched_tokens": anchor_match.get("matched_tokens", []),
        "selection_reason": {
            "gap_to_anchor": round(best["gap"], 2),
            "center_inside": best["center_inside"],
        },
        "top_candidates": [
            {
                "segment_id": item["segment"].id,
                "bbox": item["segment"].bbox,
                "score": round(item["score"], 4),
            }
            for item in scored_segments[:5]
        ],
    }
