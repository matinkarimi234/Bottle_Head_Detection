# tests/test_ncnn_image.py

import argparse
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ncnn_detector import NcnnYoloSettings, NcnnYoloDetector
from src.processing import FrameProcessor, ProcessingSettings


def parse_args():
    parser = argparse.ArgumentParser(description="Test NCNN model on one image.")

    parser.add_argument("--model-dir", required=True, help="Path to NCNN model folder.")
    parser.add_argument("--image", required=True, help="Path to input image.")

    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--output", default="ncnn_image_result.jpg")
    parser.add_argument("--show", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    frame = cv2.imread(args.image)

    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    detector_settings = NcnnYoloSettings(
        model_dir=args.model_dir,
        img_size=args.img_size,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        num_threads=args.threads,
        input_name=args.input_name,
        output_name=args.output_name,
    )

    detector = NcnnYoloDetector(detector_settings)
    detector.load()

    processor = FrameProcessor(
        detector=detector,
        settings=ProcessingSettings(),
    )

    result = processor.process(frame)

    print("Detections:")
    for det in result.detections:
        print(
            f"  label={det.label}, "
            f"conf={det.confidence:.3f}, "
            f"box={det.box_xyxy}, "
            f"center={det.center_xy}"
        )

    print(f"Inference time: {result.inference_ms:.1f} ms")
    print(f"Total process time: {result.total_process_ms:.1f} ms")

    cv2.imwrite(args.output, result.overlay_frame)
    print(f"Saved: {args.output}")

    if args.show:
        cv2.imshow("NCNN Image Test", result.overlay_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()