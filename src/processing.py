# src/processing.py

from dataclasses import dataclass
from typing import List, Optional
import time

import cv2
import numpy as np

from src.ncnn_detector import Detection, NcnnYoloDetector


@dataclass
class ProcessingSettings:
    draw_boxes: bool = True
    draw_centers: bool = True
    draw_labels: bool = True
    draw_fps: bool = True

    center_radius: int = 5
    box_thickness: int = 2


@dataclass
class ProcessedFrame:
    raw_frame: np.ndarray
    overlay_frame: np.ndarray
    detections: List[Detection]
    inference_ms: float
    total_process_ms: float


class FrameProcessor:
    def __init__(
        self,
        detector: NcnnYoloDetector,
        settings: Optional[ProcessingSettings] = None,
    ):
        self.detector = detector
        self.settings = settings or ProcessingSettings()

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        total_start = time.time()

        raw_frame = frame_bgr.copy()
        overlay_frame = frame_bgr.copy()

        infer_start = time.time()
        detections = self.detector.detect(frame_bgr)
        inference_ms = (time.time() - infer_start) * 1000.0

        self._draw_overlay(
            overlay_frame=overlay_frame,
            detections=detections,
            inference_ms=inference_ms,
        )

        total_process_ms = (time.time() - total_start) * 1000.0

        return ProcessedFrame(
            raw_frame=raw_frame,
            overlay_frame=overlay_frame,
            detections=detections,
            inference_ms=inference_ms,
            total_process_ms=total_process_ms,
        )

    def _draw_overlay(
        self,
        overlay_frame: np.ndarray,
        detections: List[Detection],
        inference_ms: float,
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            cx, cy = det.center_xy

            if self.settings.draw_boxes:
                cv2.rectangle(
                    overlay_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    self.settings.box_thickness,
                )

            if self.settings.draw_centers:
                cv2.circle(
                    overlay_frame,
                    (cx, cy),
                    self.settings.center_radius,
                    (0, 0, 255),
                    -1,
                )

                cv2.line(
                    overlay_frame,
                    (cx - 10, cy),
                    (cx + 10, cy),
                    (0, 0, 255),
                    1,
                )

                cv2.line(
                    overlay_frame,
                    (cx, cy - 10),
                    (cx, cy + 10),
                    (0, 0, 255),
                    1,
                )

            if self.settings.draw_labels:
                text = f"{det.label} {det.confidence:.2f}"

                cv2.putText(
                    overlay_frame,
                    text,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                center_text = f"C=({cx},{cy})"

                cv2.putText(
                    overlay_frame,
                    center_text,
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                )

        if self.settings.draw_fps:
            if inference_ms > 0:
                infer_fps = 1000.0 / inference_ms
            else:
                infer_fps = 0.0

            info = f"NCNN: {inference_ms:.1f} ms | Infer FPS: {infer_fps:.1f} | Objects: {len(detections)}"

            cv2.putText(
                overlay_frame,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )