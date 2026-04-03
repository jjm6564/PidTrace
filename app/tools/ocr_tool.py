import os
from typing import Dict, List, Tuple

import numpy as np

from app.schemas import OCRText


_reader = None


def _get_cv2():
    import cv2

    return cv2


def get_reader():
    global _reader
    if _reader is None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        import torch
        import easyocr

        use_gpu = torch.cuda.is_available()
        _reader = easyocr.Reader(["en"], gpu=use_gpu)
    return _reader


def _load_image(image_path: str):
    cv2 = _get_cv2()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return image


def _to_gray(image):
    cv2 = _get_cv2()
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _rotate_image(image, rotation: int):
    cv2 = _get_cv2()
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _enhance_text(gray, scale: float):
    cv2 = _get_cv2()

    # 1) contrast stretching first on the original grayscale
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    if max_val > min_val:
        stretched = (
            (gray.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
        ).clip(0, 255).astype(np.uint8)
    else:
        stretched = gray

    # 2) enlarge with Lanczos interpolation
    enlarged = cv2.resize(stretched, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # 3) local contrast enhancement on the enlarged image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enlarged)

    # 4) stronger unsharp masking to recover text strokes
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharpen = cv2.addWeighted(clahe, 2.5, blur, -1.5, 0)

    # 5) binary thresholding for OCR
    _, binary = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        ("gray", gray),
        ("stretched", stretched),
        ("enlarged", enlarged),
        ("clahe", clahe),
        ("sharpen", sharpen),
        ("binary", binary),
    ]


def _variance_of_laplacian(gray) -> float:
    cv2 = _get_cv2()
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _contrast_range(gray) -> float:
    try:
        p5 = float(np.percentile(gray, 5))
        p95 = float(np.percentile(gray, 95))
        return round(p95 - p5, 4)
    except Exception:
        return 0.0


def _quality_metrics(gray, label: str, scale: float = 1.0) -> Dict[str, object]:
    height, width = gray.shape[:2]
    return {
        "label": label,
        "width": int(width),
        "height": int(height),
        "scale": float(scale),
        "mean_brightness": round(float(np.mean(gray)), 4),
        "std_brightness": round(float(np.std(gray)), 4),
        "contrast_range": _contrast_range(gray),
        "sharpness": round(_variance_of_laplacian(gray), 4),
    }


def get_image_quality_summary(image_path: str) -> Dict[str, object]:
    image = _load_image(image_path)
    gray = _to_gray(image)
    enhanced_variants = _enhance_text(gray, 4.0)

    representative = None
    for name, variant in enhanced_variants:
        if name == "sharpen":
            representative = variant
            break
    if representative is None:
        representative = enhanced_variants[0][1]

    return {
        "input_quality": _quality_metrics(gray, "input", scale=1.0),
        "output_quality": _quality_metrics(representative, "enhanced_output", scale=4.0),
    }


def _prepare_variants(image) -> List[Dict[str, object]]:
    cv2 = _get_cv2()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants: List[Dict[str, object]] = []
    variants.append({"name": "gray", "image": gray, "rotation": 0, "scale": 1.0, "offset": (0, 0)})

    for scale in (2.0, 3.0, 4.0):
        for name, variant in _enhance_text(gray, scale):
            variants.append(
                {
                    "name": f"{name}_{int(scale)}x",
                    "image": variant,
                    "rotation": 0,
                    "scale": scale,
                    "offset": (0, 0),
                }
            )

    for scale in (4.0,):
        for name, variant in _enhance_text(gray, scale):
            for rotation in (90, 270):
                variants.append(
                    {
                        "name": f"{name}_{int(scale)}x_rot{rotation}",
                        "image": _rotate_image(variant, rotation),
                        "rotation": rotation,
                        "scale": scale,
                        "offset": (0, 0),
                    }
                )

    variants.extend(_prepare_pipe_label_crop_variants(image))
    return variants


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _bbox_from_poly(poly) -> Tuple[int, int, int, int]:
    xs = [int(point[0]) for point in poly]
    ys = [int(point[1]) for point in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _prepare_pipe_label_crop_variants(image) -> List[Dict[str, object]]:
    cv2 = _get_cv2()

    try:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        work = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = 255 - binary
        contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return []

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 120 or h < 18:
            continue
        if w < h * 3:
            continue
        if h > 120:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: item[2] * item[3], reverse=True)

    variants: List[Dict[str, object]] = []
    for idx, (x, y, w, h) in enumerate(boxes[:20]):
        x1 = max(0, int((x - 20) / 2))
        y1 = max(0, int((y - 20) / 2))
        x2 = min(width, int((x + w + 20) / 2))
        y2 = min(height, int((y + h + 20) / 2))
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
        if x2 - x1 < 50 or y2 - y1 < 12:
            continue

        try:
            crop = gray[y1:y2, x1:x2]
        except Exception:
            continue

        for name, variant in _enhance_text(crop, 6.0):
            variants.append(
                {
                    "name": f"pipecrop_{idx}_{name}",
                    "image": variant,
                    "rotation": 0,
                    "scale": 6.0,
                    "offset": (x1, y1),
                }
            )

    return variants


def _remap_rotated_point(x: int, y: int, width: int, height: int, rotation: int) -> Tuple[int, int]:
    if rotation == 90:
        return (y, height - 1 - x)
    if rotation == 270:
        return (width - 1 - y, x)
    return (x, y)


def _remap_bbox_from_rotation(bbox, image, rotation: int) -> Tuple[int, int, int, int]:
    if rotation == 0:
        return bbox

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    corners = [
        _remap_rotated_point(x1, y1, width, height, rotation),
        _remap_rotated_point(x2, y1, width, height, rotation),
        _remap_rotated_point(x1, y2, width, height, rotation),
        _remap_rotated_point(x2, y2, width, height, rotation),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _remap_bbox_from_scale_and_offset(
    bbox,
    scale: float,
    offset: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ox, oy = offset
    return (
        int(x1 / scale) + ox,
        int(y1 / scale) + oy,
        int(x2 / scale) + ox,
        int(y2 / scale) + oy,
    )


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _is_similar_detection(existing: OCRText, text: str, bbox) -> bool:
    normalized_existing = _normalize_text(existing.text).lower()
    normalized_new = _normalize_text(text).lower()
    return normalized_existing == normalized_new and _bbox_iou(existing.bbox, bbox) > 0.3


def run_ocr(image_path: str) -> List[OCRText]:
    reader = get_reader()
    image = _load_image(image_path)
    variants = _prepare_variants(image)

    outputs: List[OCRText] = []
    best_by_key: Dict[Tuple[str, int, int], OCRText] = {}

    for variant_info in variants:
        variant = variant_info["image"]
        rotation = int(variant_info["rotation"])
        scale = float(variant_info.get("scale", 1.0))
        offset = variant_info.get("offset", (0, 0))

        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.2,
                rotation_info=[90, 180, 270] if rotation == 0 else None,
                mag_ratio=1.8,
                canvas_size=3200,
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )
        except Exception:
            continue

        for poly, text, score in results:
            normalized = _normalize_text(text)
            if not normalized:
                continue

            bbox = _bbox_from_poly(poly)
            bbox = _remap_bbox_from_rotation(bbox, variant, rotation)
            bbox = _remap_bbox_from_scale_and_offset(bbox, scale, offset)

            duplicate = False
            for existing in outputs:
                if _is_similar_detection(existing, normalized, bbox):
                    duplicate = True
                    if score > existing.score:
                        existing.text = normalized
                        existing.bbox = bbox
                        existing.score = float(score)
                    break

            if duplicate:
                continue

            item = OCRText(text=normalized, bbox=bbox, score=float(score))
            outputs.append(item)
            key = (normalized.lower(), bbox[0] // 10, bbox[1] // 10)
            previous = best_by_key.get(key)
            if previous is None or item.score > previous.score:
                best_by_key[key] = item

    final_outputs = list(best_by_key.values())
    final_outputs.sort(key=lambda item: item.score, reverse=True)
    return final_outputs
