# src/ncnn_detector.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import time

import cv2
import numpy as np
import ncnn


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]


@dataclass
class NcnnYoloSettings:
    model_dir: str
    img_size: int = 640
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    num_threads: int = 4

    # If automatic name detection fails, set these manually.
    input_name: Optional[str] = None
    output_name: Optional[str] = None


class NcnnYoloDetector:
    def __init__(self, settings: NcnnYoloSettings):
        self.settings = settings
        self.model_dir = Path(settings.model_dir)

        self.net: Optional[ncnn.Net] = None
        self.input_name: Optional[str] = settings.input_name
        self.output_name: Optional[str] = settings.output_name

        self.class_names = self._load_class_names()

    def load(self) -> None:
        param_path, bin_path = self._find_model_files()

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = self.settings.num_threads

        ret_param = self.net.load_param(str(param_path))
        ret_bin = self.net.load_model(str(bin_path))

        if ret_param != 0 or ret_bin != 0:
            raise RuntimeError(
                f"Failed to load NCNN model. "
                f"param result={ret_param}, bin result={ret_bin}"
            )

        self._resolve_input_output_names(param_path)

        print("NCNN model loaded:")
        print(f"  param      : {param_path}")
        print(f"  bin        : {bin_path}")
        print(f"  input name : {self.input_name}")
        print(f"  output name: {self.output_name}")
        print(f"  classes    : {self.class_names}")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.net is None:
            raise RuntimeError("NCNN model is not loaded. Call detector.load() first.")

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("NCNN input/output names are not resolved.")

        original_h, original_w = frame_bgr.shape[:2]

        letterboxed, scale, pad_x, pad_y = self._letterbox(
            frame_bgr,
            self.settings.img_size,
        )

        mat_in = self._make_ncnn_input(letterboxed)

        with self.net.create_extractor() as ex:
            ex.input(self.input_name, mat_in)

            ret, mat_out = ex.extract(self.output_name)

            if ret != 0:
                raise RuntimeError(
                    f"NCNN extract failed for output '{self.output_name}'. "
                    f"Return code: {ret}"
                )

        pred = np.array(mat_out)
        detections = self._decode_yolo_output(
            pred=pred,
            original_w=original_w,
            original_h=original_h,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        return detections

    def _find_model_files(self) -> Tuple[Path, Path]:
        param_files = list(self.model_dir.glob("*.param"))

        if not param_files:
            raise FileNotFoundError(f"No .param file found in {self.model_dir}")

        param_path = param_files[0]
        bin_path = param_path.with_suffix(".bin")

        if not bin_path.exists():
            raise FileNotFoundError(f"Matching .bin file not found: {bin_path}")

        return param_path, bin_path

    def _load_class_names(self) -> List[str]:
        metadata_path = self.model_dir / "metadata.yaml"

        if not metadata_path.exists():
            return ["object"]

        try:
            import yaml

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)

            names = metadata.get("names", None)

            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]

            if isinstance(names, list):
                return names

        except Exception as exc:
            print(f"Warning: could not read metadata.yaml: {exc}")

        return ["object"]

    def _resolve_input_output_names(self, param_path: Path) -> None:
        if self.input_name is not None and self.output_name is not None:
            return

        # Try NCNN Python API first.
        try:
            input_names = list(self.net.input_names())
            output_names = list(self.net.output_names())

            if self.input_name is None and len(input_names) > 0:
                self.input_name = input_names[0]

            if self.output_name is None and len(output_names) > 0:
                self.output_name = output_names[0]

        except Exception:
            pass

        # Fallback: parse .param file.
        parsed_input, parsed_output = self._parse_param_names(param_path)

        if self.input_name is None:
            self.input_name = parsed_input or "in0"

        if self.output_name is None:
            self.output_name = parsed_output or "out0"

    def _parse_param_names(self, param_path: Path) -> Tuple[Optional[str], Optional[str]]:
        input_name = None
        output_name = None

        try:
            with open(param_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 4:
                    continue

                layer_type = parts[0]

                if layer_type == "Input" and len(parts) >= 5:
                    input_name = parts[4]

                if layer_type == "Output" and len(parts) >= 5:
                    output_name = parts[4]

        except Exception as exc:
            print(f"Warning: could not parse param names: {exc}")

        return input_name, output_name

    def _letterbox(
        self,
        image_bgr: np.ndarray,
        target_size: int,
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, int, int]:
        h, w = image_bgr.shape[:2]

        scale = min(target_size / w, target_size / h)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (target_size, target_size, 3),
            color,
            dtype=np.uint8,
        )

        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    def _make_ncnn_input(self, image_bgr_square: np.ndarray):
        h, w = image_bgr_square.shape[:2]

        try:
            mat_in = ncnn.Mat.from_pixels(
                image_bgr_square,
                ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                w,
                h,
            )
            mat_in.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
            return mat_in

        except Exception:
            image_rgb = cv2.cvtColor(image_bgr_square, cv2.COLOR_BGR2RGB)
            blob = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            return ncnn.Mat(blob)

    def _decode_yolo_output(
        self,
        pred: np.ndarray,
        original_w: int,
        original_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        if pred.ndim == 3:
            pred = pred[0]

        # Common YOLO shape:
        #   [84, 8400] or [8400, 84]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.shape[1] < 5:
            raise RuntimeError(f"Unexpected YOLO output shape: {pred.shape}")

        num_classes_from_metadata = len(self.class_names)

        if pred.shape[1] == 5 + num_classes_from_metadata:
            # YOLOv5-like: x, y, w, h, objectness, class scores...
            boxes_xywh = pred[:, 0:4]
            objectness = pred[:, 4]
            class_scores = pred[:, 5:]

            class_ids = np.argmax(class_scores, axis=1)
            scores = objectness * class_scores[np.arange(len(class_scores)), class_ids]

        else:
            # YOLOv8 / YOLO11-like: x, y, w, h, class scores...
            boxes_xywh = pred[:, 0:4]
            class_scores = pred[:, 4:]

            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), class_ids]

        keep = scores >= self.settings.conf_threshold

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # Undo letterbox padding.
        boxes_xyxy[:, [0, 2]] -= pad_x
        boxes_xyxy[:, [1, 3]] -= pad_y

        # Undo resize scale.
        boxes_xyxy /= scale

        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, original_w - 1)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, original_h - 1)

        nms_boxes = []
        for x1, y1, x2, y2 in boxes_xyxy:
            nms_boxes.append(
                [
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                ]
            )

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.tolist(),
            self.settings.conf_threshold,
            self.settings.nms_threshold,
        )

        detections: List[Detection] = []

        if len(indices) == 0:
            return detections

        for i in np.array(indices).flatten():
            x, y, w, h = nms_boxes[i]

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            class_id = int(class_ids[i])

            if 0 <= class_id < len(self.class_names):
                label = self.class_names[class_id]
            else:
                label = str(class_id)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=float(scores[i]),
                    box_xyxy=(x1, y1, x2, y2),
                    center_xy=(center_x, center_y),
                )
            )

        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes_xyxy = np.zeros_like(boxes_xywh)

        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        return boxes_xyxy