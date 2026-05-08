# tests/test_usb_camera_connection.py

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.camera import UsbCameraSettings, UsbCameraSource


def parse_args():
    parser = argparse.ArgumentParser(description="USB camera connection test.")

    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="/dev/video0")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")

    parser.add_argument("--enable-ae", action="store_true")
    parser.add_argument("--enable-awb", action="store_true")

    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument("--brightness", type=int, default=None)
    parser.add_argument("--contrast", type=int, default=None)
    parser.add_argument("--saturation", type=int, default=None)
    parser.add_argument("--wb-temp", type=int, default=None)

    parser.add_argument("--flip-h", action="store_true")
    parser.add_argument("--flip-v", action="store_true")

    parser.add_argument("--save-frame", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    settings = UsbCameraSettings(
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
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        white_balance_temperature=args.wb_temp,
        flip_horizontal=args.flip_h,
        flip_vertical=args.flip_v,
    )

    print("Opening USB camera with settings:")
    print(settings)

    frame_counter = 0
    fps_timer = time.time()
    measured_fps = 0.0
    start_time = time.time()
    saved_once = False

    with UsbCameraSource(settings) as camera:
        print("Camera test started.")
        print("Press Q to quit.")
        print("Press S to save frame.")

        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Frame read failed.")
                time.sleep(0.05)
                continue

            frame_counter += 1
            now = time.time()

            if now - fps_timer >= 1.0:
                measured_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now
                print(f"Measured FPS: {measured_fps:.1f}")

            text_1 = (
                f"USB Camera | {args.width}x{args.height} | "
                f"Target FPS: {args.fps} | Measured FPS: {measured_fps:.1f}"
            )
            text_2 = (
                f"AE: {'ON' if args.enable_ae else 'OFF'} | "
                f"AWB: {'ON' if args.enable_awb else 'OFF'} | "
                f"FOURCC: {args.fourcc}"
            )

            cv2.putText(
                frame,
                text_1,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                text_2,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            if args.save_frame and not saved_once:
                cv2.imwrite("usb_camera_test_frame.jpg", frame)
                print("Saved: usb_camera_test_frame.jpg")
                saved_once = True

            if args.no_display:
                if args.duration > 0 and now - start_time >= args.duration:
                    break
                continue

            cv2.imshow("USB Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                cv2.imwrite("usb_camera_test_frame.jpg", frame)
                print("Saved: usb_camera_test_frame.jpg")

            if args.duration > 0 and now - start_time >= args.duration:
                break

    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()