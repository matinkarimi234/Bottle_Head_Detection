# src/processing.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np

from src.ncnn_detector import Detection, NcnnYoloDetector
from src.vision_utils import (
    add_roi_rectangle,
    crop_roi,
    draw_line_with_label,
    draw_text_box,
    reached_or_passed_line,
    roi_from_fractions,
    scaled_line_points,
)


@dataclass
class EntryResult:
    present: bool
    foreground_pixels: int
    largest_area: float
    largest_box: Optional[Tuple[int, int, int, int]]
    mask_clean: Optional[np.ndarray] = None


class EntryMotionDetector:
    """
    OpenCV-only entry detector from your notebook.

    It uses:
      bottom-left ROI
      grayscale + Gaussian blur
      absolute difference from median background
      threshold + morph open/close
      foreground pixel count + largest contour area
      consecutive positive frames
    """

    def __init__(self, settings, roi_fractions: dict):
        self.settings = settings
        self.roi_fractions = roi_fractions
        self.background_gray: Optional[np.ndarray] = None
        self.entry_roi_rect: Optional[Tuple[int, int, int, int]] = None
        self.positive_count = 0

    @property
    def is_ready(self) -> bool:
        return self.background_gray is not None and self.entry_roi_rect is not None

    def reset_counter(self) -> None:
        self.positive_count = 0

    def build_background_from_camera(self, camera, frame_count: Optional[int] = None) -> None:
        n = frame_count or self.settings.background_warmup_frames
        frames_gray = []

        print(f"Building entry background from {n} frames...")

        for _ in range(n):
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            if self.entry_roi_rect is None:
                self.entry_roi_rect = roi_from_fractions(frame.shape, self.roi_fractions)

            roi = crop_roi(frame, self.entry_roi_rect)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            frames_gray.append(gray)

        if not frames_gray:
            raise RuntimeError("Could not build entry background: no frames captured.")

        self.background_gray = np.median(np.stack(frames_gray, axis=0), axis=0).astype(np.uint8)
        self.positive_count = 0
        print("Entry background ready.")

    def detect(self, frame: np.ndarray) -> EntryResult:
        if not self.is_ready:
            self.entry_roi_rect = roi_from_fractions(frame.shape, self.roi_fractions)
            raise RuntimeError("Entry background is not ready. Call build_background_from_camera().")

        roi = crop_roi(frame, self.entry_roi_rect)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(self.background_gray, gray)
        _, mask_raw = cv2.threshold(diff, self.settings.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((self.settings.morph_kernel_size, self.settings.morph_kernel_size), np.uint8)
        mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        foreground_pixels = int(cv2.countNonZero(mask_clean))
        largest_area = 0.0
        largest_box = None

        if contours:
            c = max(contours, key=cv2.contourArea)
            largest_area = float(cv2.contourArea(c))
            largest_box = cv2.boundingRect(c)

        raw_present = (
            foreground_pixels >= self.settings.min_foreground_pixels
            and largest_area >= self.settings.min_largest_contour_area
        )

        if raw_present:
            self.positive_count += 1
        else:
            self.positive_count = 0

        present = self.positive_count >= self.settings.consecutive_frames_required

        return EntryResult(
            present=present,
            foreground_pixels=foreground_pixels,
            largest_area=largest_area,
            largest_box=largest_box,
            mask_clean=mask_clean,
        )


@dataclass
class PairResult:
    detections: List[Detection]
    left_bottle: Optional[Detection]
    right_bottle: Optional[Detection]
    detect_roi_rect: Tuple[int, int, int, int]
    inference_ms: float


@dataclass
class LineStatus:
    left_passed_now: bool
    right_passed_now: bool
    left_dist: Optional[float]
    right_dist: Optional[float]
    left_line: Tuple[Tuple[int, int], Tuple[int, int]]
    right_line: Tuple[Tuple[int, int], Tuple[int, int]]


class BottleFrameProcessor:
    def __init__(self, detector: NcnnYoloDetector, settings):
        self.detector = detector
        self.settings = settings

    def detect_pair(self, frame: np.ndarray) -> PairResult:
        detect_roi_rect = roi_from_fractions(frame.shape, self.settings.roi.detect_roi_fractions)
        rx1, ry1, _, _ = detect_roi_rect

        roi = crop_roi(frame, detect_roi_rect)

        t0 = time.time()
        detections_roi = self.detector.detect(roi)
        inference_ms = (time.time() - t0) * 1000.0

        detections_full = []
        for d in detections_roi:
            x1, y1, x2, y2 = d.box_xyxy
            cx, cy = d.center_xy

            mapped = Detection(
                class_id=d.class_id,
                label=d.label,
                confidence=d.confidence,
                box_xyxy=(x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1),
                center_xy=(cx + rx1, cy + ry1),
                area=d.area,
            )
            detections_full.append(mapped)

        detections_full = sorted(detections_full, key=lambda x: x.area, reverse=True)[: self.settings.model.max_targets]
        detections_full = sorted(detections_full, key=lambda x: x.center_xy[0])

        left_bottle = None
        right_bottle = None

        if len(detections_full) >= 2:
            left_bottle = detections_full[0]   # second/trailing bottle
            right_bottle = detections_full[1]  # first/ahead bottle

        return PairResult(
            detections=detections_full,
            left_bottle=left_bottle,
            right_bottle=right_bottle,
            detect_roi_rect=detect_roi_rect,
            inference_ms=inference_ms,
        )

    def check_lines(
        self,
        frame_shape,
        left_bottle: Optional[Detection],
        right_bottle: Optional[Detection],
    ) -> LineStatus:
        line_settings = self.settings.lines

        left_line = scaled_line_points(
            line_settings.left_line_p1,
            line_settings.left_line_p2,
            frame_shape,
            line_settings.reference_frame_width,
            line_settings.reference_frame_height,
        )

        right_line = scaled_line_points(
            line_settings.right_line_p1,
            line_settings.right_line_p2,
            frame_shape,
            line_settings.reference_frame_width,
            line_settings.reference_frame_height,
        )

        left_passed_now = False
        right_passed_now = False
        left_dist = None
        right_dist = None

        if left_bottle is not None:
            left_passed_now, left_dist, _ = reached_or_passed_line(
                left_bottle.center_xy,
                left_line[0],
                left_line[1],
                line_settings.line_tolerance_px,
            )

        if right_bottle is not None:
            right_passed_now, right_dist, _ = reached_or_passed_line(
                right_bottle.center_xy,
                right_line[0],
                right_line[1],
                line_settings.line_tolerance_px,
            )

        return LineStatus(
            left_passed_now=left_passed_now,
            right_passed_now=right_passed_now,
            left_dist=left_dist,
            right_dist=right_dist,
            left_line=left_line,
            right_line=right_line,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        state_name: str,
        cycle_id: int,
        entry_roi_rect: Optional[Tuple[int, int, int, int]],
        pair: Optional[PairResult],
        line_status: Optional[LineStatus],
        left_passed_latched: bool,
        right_passed_latched: bool,
        event_text: str = "",
    ) -> np.ndarray:
        out = frame.copy()
        overlay = self.settings.overlay

        if entry_roi_rect is not None:
            add_roi_rectangle(out, entry_roi_rect, "ENTRY ROI", overlay.entry_roi_color)

        if pair is not None:
            add_roi_rectangle(out, pair.detect_roi_rect, "NCNN ROI", overlay.detect_roi_color)

        if line_status is not None:
            draw_line_with_label(out, line_status.left_line[0], line_status.left_line[1], "LEFT / second", overlay.left_line_color)
            draw_line_with_label(out, line_status.right_line[0], line_status.right_line[1], "RIGHT / first", overlay.right_line_color)

        if pair is not None:
            if pair.left_bottle is not None:
                self._draw_bottle(
                    out,
                    pair.left_bottle,
                    "SECOND/LEFT",
                    left_passed_latched,
                    overlay.passed_color if left_passed_latched else overlay.left_before_color,
                )

            if pair.right_bottle is not None:
                self._draw_bottle(
                    out,
                    pair.right_bottle,
                    "FIRST/RIGHT",
                    right_passed_latched,
                    overlay.passed_color if right_passed_latched else overlay.right_before_color,
                )

        draw_text_box(out, f"STATE: {state_name} | cycle={cycle_id}", (10, 25), scale=0.48)
        draw_text_box(
            out,
            f"right_passed={right_passed_latched} | left_passed={left_passed_latched}",
            (10, 52),
            scale=0.43,
        )

        if pair is not None:
            draw_text_box(
                out,
                f"NCNN {pair.inference_ms:.1f} ms | objects={len(pair.detections)}",
                (10, 79),
                scale=0.43,
            )

        if line_status is not None:
            ld = "N/A" if line_status.left_dist is None else f"{line_status.left_dist:.1f}"
            rd = "N/A" if line_status.right_dist is None else f"{line_status.right_dist:.1f}"
            draw_text_box(out, f"dist left={ld} | dist right={rd}", (10, 106), scale=0.40)

        if event_text:
            draw_text_box(out, event_text, (10, 136), color=(0, 255, 255), scale=0.50)

        return out

    @staticmethod
    def _draw_bottle(img, det: Detection, role: str, passed: bool, color):
        x1, y1, x2, y2 = det.box_xyxy
        cx, cy = det.center_xy

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 9, color, 2)

        status = "PASSED" if passed else "WAIT"
        draw_text_box(
            img,
            f"{role} {det.label} {det.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            color=color,
            scale=0.38,
        )
        draw_text_box(
            img,
            f"{status} C=({cx},{cy})",
            (cx + 6, max(20, cy - 6)),
            color=color,
            scale=0.34,
        )
