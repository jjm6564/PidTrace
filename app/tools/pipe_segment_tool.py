from typing import List

import cv2
import numpy as np

from app.schemas import PipeSegment


def extract_pipe_segments(image_path: str) -> List[PipeSegment]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    idx = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 and h < 8:
            continue

        points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        segments.append(
            PipeSegment(
                id=f"seg_{idx}",
                points=points,
                bbox=(x, y, x + w, y + h),
                thickness=1.0,
            )
        )
        idx += 1

    return segments
