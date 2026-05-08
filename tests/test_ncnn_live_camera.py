# tests/test_ncnn_live_camera.py

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.camera import UsbCameraSettings, UsbCameraSource
from src.ncnn_detector import NcnnYoloSettings, NcnnYoloDetector
from src.processing import FrameProcessor, ProcessingSettings


def parse_args():
    parser = argparse.ArgumentParser(description="Live USB camera + NCNN test.")

    parser.add_argument("--model-dir", required=True)

    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")

    parser.add_argument("--enable-ae", action="store_true")
    parser.add_argument("--enable-awb", action="store_true")

    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument("--wb-temp", type=int, default=None)

    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--save-frame", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    camera_settings = UsbCameraSettings(
        device_index=args.index,
        device_path=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        disable_auto_exposure=not args.enable_ae,
        disable_auto_white_balance=not args.enable_awb,
        exposure=args.exposure,
        gain=args.gain,
        white_balance_temperature=args.wb_temp,
    )

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

    start_time = time.time()
    frame_counter = 0
    fps_timer = time.time()
    measured_camera_process_fps = 0.0
    saved_once = False

    with UsbCameraSource(camera_settings) as camera:
        print("Live NCNN camera test started.")
        print("Press Q to quit.")
        print("Press S to save overlay frame.")

        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Frame read failed.")
                time.sleep(0.05)
                continue

            result = processor.process(frame)

            frame_counter += 1
            now = time.time()

            if now - fps_timer >= 1.0:
                measured_camera_process_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

                print(
                    f"Loop FPS={measured_camera_process_fps:.1f}, "
                    f"Inference={result.inference_ms:.1f} ms, "
                    f"Objects={len(result.detections)}"
                )

            cv2.putText(
                result.overlay_frame,
                f"Loop FPS: {measured_camera_process_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if args.save_frame and not saved_once:
                cv2.imwrite("ncnn_live_test_frame.jpg", result.overlay_frame)
                print("Saved: ncnn_live_test_frame.jpg")
                saved_once = True

            if args.no_display:
                if args.duration > 0 and now - start_time >= args.duration:
                    break
                continue

            cv2.imshow("USB Camera + NCNN Test", result.overlay_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                cv2.imwrite("ncnn_live_test_frame.jpg", result.overlay_frame)
                print("Saved: ncnn_live_test_frame.jpg")

            if args.duration > 0 and now - start_time >= args.duration:
                break

    cv2.destroyAllWindows()
    print("Live NCNN camera test finished.")


if __name__ == "__main__":
    main()