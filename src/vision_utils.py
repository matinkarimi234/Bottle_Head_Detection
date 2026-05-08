# src/vision_utils.py

from typing import Tuple
import math
import cv2
import numpy as np


def roi_from_fractions(frame_shape, fractions: dict) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1 = int(round(fractions["x1"] * w))
    y1 = int(round(fractions["y1"] * h))
    x2 = int(round(fractions["x2"] * w))
    y2 = int(round(fractions["y2"] * h))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def crop_roi(frame, roi_rect: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = roi_rect
    return frame[y1:y2, x1:x2]


def scale_point(
    point: Tuple[int, int],
    frame_shape,
    reference_width: int,
    reference_height: int,
) -> Tuple[int, int]:
    h, w = frame_shape[:2]
    x = int(round(point[0] * w / reference_width))
    y = int(round(point[1] * h / reference_height))
    return x, y


def scaled_line_points(p1, p2, frame_shape, reference_width, reference_height):
    return (
        scale_point(p1, frame_shape, reference_width, reference_height),
        scale_point(p2, frame_shape, reference_width, reference_height),
    )


def line_distance(point, p1, p2) -> float:
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def x_of_line_at_y(p1, p2, y) -> float:
    x1, y1 = p1
    x2, y2 = p2

    if y2 == y1:
        return (x1 + x2) / 2.0

    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def reached_or_passed_line(point, p1, p2, tolerance_px=8):
    # For left-to-right motion:
    # reached/passed = center near the line OR to the right of the line at the same y.
    px, py = point
    dist = line_distance(point, p1, p2)
    x_line = x_of_line_at_y(p1, p2, py)
    on_or_right = px >= x_line
    return bool((dist <= tolerance_px) or on_or_right), float(dist), float(x_line)


def draw_text_box(
    img,
    text: str,
    org,
    color=(255, 255, 255),
    bg=(0, 0, 0),
    scale=0.55,
    thickness=2,
):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - baseline - 6), (x + tw + 8, y + 6), bg, -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def add_roi_rectangle(img, roi_rect, label, color):
    x1, y1, x2, y2 = roi_rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    draw_text_box(img, label, (x1 + 5, y1 + 25), color=color, bg=(0, 0, 0), scale=0.45)


def draw_line_with_label(img, p1, p2, label, color, thickness=2):
    cv2.line(img, p1, p2, color, thickness)
    cv2.circle(img, p1, 4, color, -1)
    cv2.circle(img, p2, 4, color, -1)
    draw_text_box(img, label, (p1[0] + 5, max(p1[1] + 18, 20)), color=color, bg=(0, 0, 0), scale=0.42)
