from typing import Any, Dict, List, TypedDict

from app.schemas import EquipmentCandidate, Junction, OCRText, PathCandidate, PipeSegment


class AgentState(TypedDict, total=False):
    image_path: str
    image_size: tuple
    target_text: str
    target_desc: str
    ocr_texts: List[OCRText]
    equipment_candidates: List[EquipmentCandidate]
    pipe_segments: List[PipeSegment]
    junctions: List[Junction]
    selected_pipe_seed: Dict[str, Any]
    path_candidates: List[PathCandidate]
    best_path: Dict[str, Any]
    overlay_path: str
    crop_paths: List[str]
    final_result: Dict[str, Any]
    logs: List[str]
