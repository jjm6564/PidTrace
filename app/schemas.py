from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


class OCRText(BaseModel):
    text: str
    bbox: BBox
    score: float


class EquipmentCandidate(BaseModel):
    name: str
    bbox: BBox
    equipment_type: Optional[str] = None
    score: float = 0.0


class PipeSegment(BaseModel):
    id: str
    points: List[Tuple[int, int]]
    bbox: BBox
    thickness: float = 1.0


class Junction(BaseModel):
    id: str
    point: Tuple[int, int]


class PathCandidate(BaseModel):
    segment_ids: List[str]
    endpoints: List[Tuple[int, int]]
    score: float = 0.0


class FromToResult(BaseModel):
    FROM: str = Field(..., description="Source equipment")
    TO: str = Field(..., description="Destination equipment")
    answer: str = Field("", description="Final Korean answer in the requested style")
    confidence: float = 0.0
    reason: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
