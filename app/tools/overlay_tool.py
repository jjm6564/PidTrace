import os
from typing import List

import cv2

from app.schemas import PipeSegment


def save_overlay(
    image_path: str,
    all_segments: List[PipeSegment],
    target_segment_ids: List[str],
    out_path: str,
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    for seg in all_segments:
        x1, y1, x2, y2 = seg.bbox
        color = (0, 255, 0)
        thickness = 1

        if seg.id in target_segment_ids:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)
    return out_path
