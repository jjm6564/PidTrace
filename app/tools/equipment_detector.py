import re
from typing import List, Optional

from app.schemas import EquipmentCandidate, OCRText


EQUIPMENT_HINTS = [
    "PUMP",
    "TANK",
    "VESSEL",
    "HX",
    "HEAT EXCHANGER",
    "COMPRESSOR",
    "COLUMN",
    "REACTOR",
    "DRUM",
    "EXCHANGER",
]

EQUIPMENT_TAG_PATTERNS = [
    re.compile(r"\bP-\d{2,4}[A-Z]?(?:/[A-Z])?\b", re.IGNORECASE),
    re.compile(r"\bV-\d{2,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bE-\d{2,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bEA-\d{2,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bT-\d{2,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bC-\d{2,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bD-\d{2,4}[A-Z]?\b", re.IGNORECASE),
]

NON_EQUIPMENT_HINTS = [
    "LINE",
    "VALVE",
    "NOZZLE",
    "TEE",
    "ELBOW",
    "DRAIN",
    "VENT",
    "INSTR",
    "GAUGE",
    "PSV",
    "MOV",
    "XV-",
    "PV-",
    "FV-",
    "TV-",
    "LV-",
    "PI-",
    "TI-",
    "FI-",
    "LI-",
]


def _normalize_equipment_tag(text: str) -> str:
    upper = re.sub(r"[^A-Z0-9]", "", text.upper())

    match = re.fullmatch(r"(EA)(\d{2,4}[A-Z]?)", upper)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match = re.fullmatch(r"([PVETCD])(\d{2,4}[A-Z]?)", upper)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    return text.strip()


def _infer_equipment_type(text: str) -> Optional[str]:
    upper = _normalize_equipment_tag(text).upper()
    for hint in EQUIPMENT_HINTS:
        if hint in upper:
            return hint
    if upper.startswith("P-"):
        return "PUMP"
    if upper.startswith("V-"):
        return "VESSEL"
    if upper.startswith("E-"):
        return "EXCHANGER"
    if upper.startswith("EA-"):
        return "PACKAGE"
    if upper.startswith("T-"):
        return "TANK"
    if upper.startswith("C-"):
        return "COLUMN"
    return None


def _looks_like_equipment(text: str) -> bool:
    upper = text.upper().strip()
    normalized = _normalize_equipment_tag(text).upper()
    if not upper:
        return False
    if any(bad in upper for bad in NON_EQUIPMENT_HINTS):
        return False
    if any(hint in upper for hint in EQUIPMENT_HINTS):
        return True
    return any(pattern.search(upper) or pattern.search(normalized) for pattern in EQUIPMENT_TAG_PATTERNS)


def find_equipment_candidates(ocr_texts: List[OCRText]) -> List[EquipmentCandidate]:
    candidates = []
    for item in ocr_texts:
        if not _looks_like_equipment(item.text):
            continue

        confidence_boost = 0.15 if _infer_equipment_type(item.text) else 0.0
        candidates.append(
            EquipmentCandidate(
                name=_normalize_equipment_tag(item.text),
                bbox=item.bbox,
                equipment_type=_infer_equipment_type(item.text),
                score=min(1.0, item.score + confidence_boost),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates
