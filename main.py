# main.py

from enum import Enum, auto
from pathlib import Path
import argparse
import csv
import time

import cv2

from settings import SETTINGS
from src.camera import UsbCameraSettings, UsbCameraSource
from src.gpio_backend import GpioBackend, GpioBackendSettings
from src.ncnn_detector import NcnnYoloDetector, NcnnYoloSettings
from src.processing import BottleFrameProcessor, EntryMotionDetector


class AppState(Enum):
    STARTUP = auto()
    IDLE = auto()
    BOTTLE_ENTRANCE = auto()
    INTERSECT_RIGHT_LINE_WITH_FIRST_BOTTLE = auto()
    INTERSECT_LEFT_LINE_WITH_SECOND_BOTTLE = auto()
    BOTTLES_EXIT = auto()


def make_camera_settings():
    c = SETTINGS.camera
    return UsbCameraSettings(
        device_index=c.device_index,
        device_path=c.device_path,
        width=c.width,
        height=c.height,
        fps=c.fps,
        fourcc=c.fourcc,
        disable_auto_exposure=c.disable_auto_exposure,
        disable_auto_white_balance=c.disable_auto_white_balance,
        exposure=c.exposure,
        gain=c.gain,
        brightness=c.brightness,
        contrast=c.contrast,
        saturation=c.saturation,
        white_balance_temperature=c.white_balance_temperature,
        warmup_seconds=c.warmup_seconds,
        flip_horizontal=c.flip_horizontal,
        flip_vertical=c.flip_vertical,
    )


def make_detector():
    m = SETTINGS.model
    return NcnnYoloDetector(
        NcnnYoloSettings(
            model_dir=m.model_dir,
            img_size=m.img_size,
            conf_threshold=m.conf_threshold,
            nms_threshold=m.nms_threshold,
            num_threads=m.num_threads,
            input_name=m.input_name,
            output_name=m.output_name,
            target_class_ids=m.target_class_ids,
            target_class_names=m.target_class_names,
        )
    )


def make_gpio_settings(dummy_gpio: bool = False):
    g = SETTINGS.gpio
    return GpioBackendSettings(
        enabled=g.enabled,
        dummy_mode=g.dummy_mode or dummy_gpio,
        run_enable_input_pin=g.run_enable_input_pin,
        first_bottle_output_pin=g.first_bottle_output_pin,
        second_bottle_output_pin=g.second_bottle_output_pin,
        input_pull_up=g.input_pull_up,
        bounce_time_s=g.bounce_time_s,
        dummy_initial_run_enabled=g.dummy_initial_run_enabled,
    )


def save_event_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "cycle_id",
                "state",
                "event",
                "frame_index",
                "path",
            ],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def log_event(output_dir: Path, cycle_id: int, state: AppState, event: str, frame_index: int, path: str = ""):
    print(f"[cycle {cycle_id:03d}] {state.name}: {event} frame={frame_index} path={path}")
    save_event_row(
        output_dir / "event_log.csv",
        {
            "timestamp": time.time(),
            "cycle_id": cycle_id,
            "state": state.name,
            "event": event,
            "frame_index": frame_index,
            "path": path,
        },
    )


def save_debug_frame(output_dir: Path, subdir: str, cycle_id: int, frame_index: int, frame) -> str:
    folder = output_dir / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"cycle_{cycle_id:03d}_frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Bottle Head Detection live state machine.")

    parser.add_argument("--dummy-gpio", action="store_true", help="Run without real GPIO.")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow.")
    parser.add_argument("--write-video", action="store_true", help="Write overlay video.")
    parser.add_argument("--model-dir", default=None, help="Override NCNN model directory.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 = forever.")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(SETTINGS.app.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display = SETTINGS.app.display and not args.no_display
    write_video = SETTINGS.app.write_overlay_video or args.write_video

    # Optional runtime model path override.
    if args.model_dir:
        object.__setattr__(SETTINGS.model, "model_dir", args.model_dir)

    gpio = GpioBackend(make_gpio_settings(dummy_gpio=args.dummy_gpio))
    camera = UsbCameraSource(make_camera_settings())

    detector = make_detector()
    processor = BottleFrameProcessor(detector=detector, settings=SETTINGS)
    entry_detector = EntryMotionDetector(
        settings=SETTINGS.entry_cv,
        roi_fractions=SETTINGS.roi.entry_roi_fractions,
    )

    state = AppState.STARTUP
    cycle_id = 0
    frame_index = 0

    right_passed_latched = False
    left_passed_latched = False
    no_detection_count = 0

    pair = None
    line_status = None
    event_text = ""
    video_writer = None

    try:
        gpio.open()
        gpio.all_outputs_low()

        camera.open()

        # Startup background should be captured when the entry ROI is empty.
        entry_detector.build_background_from_camera(camera)

        state = AppState.IDLE
        log_event(output_dir, cycle_id, state, "startup_complete", frame_index)

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Frame read failed.")
                time.sleep(0.05)
                continue

            frame_index += 1

            if write_video and video_writer is None:
                h, w = frame.shape[:2]
                video_path = output_dir / "live_overlay.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, SETTINGS.camera.fps, (w, h))
                print(f"Writing overlay video: {video_path}")

            event_text = ""
            pair = None
            line_status = None

            # Global stop condition:
            # If the run-enable input goes LOW, stop the active cycle and return to IDLE.
            if state not in (AppState.STARTUP, AppState.IDLE) and not gpio.is_run_enabled():
                gpio.all_outputs_low()
                detector.unload()
                state = AppState.IDLE
                right_passed_latched = False
                left_passed_latched = False
                no_detection_count = 0
                entry_detector.reset_counter()
                log_event(output_dir, cycle_id, state, "run_enable_low_return_to_idle", frame_index)

            if state == AppState.IDLE:
                gpio.all_outputs_low()

                if gpio.is_run_enabled() or gpio.run_enable_event.is_set():
                    gpio.run_enable_event.clear()
                    state = AppState.BOTTLE_ENTRANCE
                    entry_detector.reset_counter()
                    log_event(output_dir, cycle_id, state, "external_interrupt_run_enable_high", frame_index)

            elif state == AppState.BOTTLE_ENTRANCE:
                entry_result = entry_detector.detect(frame)

                if entry_result.present:
                    cycle_id += 1
                    right_passed_latched = False
                    left_passed_latched = False
                    no_detection_count = 0

                    gpio.set_first_output(True)

                    if SETTINGS.app.load_model_on_demand:
                        detector.load()
                    elif not detector.is_loaded:
                        detector.load()

                    state = AppState.INTERSECT_RIGHT_LINE_WITH_FIRST_BOTTLE
                    event_text = f"BOTTLE ENTERED | cycle {cycle_id}"
                    log_event(output_dir, cycle_id, state, "bottle_entered_pin1_high", frame_index)

                    if SETTINGS.app.save_debug_frames:
                        overlay = processor.draw_overlay(
                            frame=frame,
                            state_name=state.name,
                            cycle_id=cycle_id,
                            entry_roi_rect=entry_detector.entry_roi_rect,
                            pair=None,
                            line_status=None,
                            left_passed_latched=False,
                            right_passed_latched=False,
                            event_text=event_text,
                        )
                        p = save_debug_frame(output_dir, "entrance_frames", cycle_id, frame_index, overlay)
                        log_event(output_dir, cycle_id, state, "entrance_frame_saved", frame_index, p)

            elif state == AppState.INTERSECT_RIGHT_LINE_WITH_FIRST_BOTTLE:
                if not detector.is_loaded:
                    detector.load()

                pair = processor.detect_pair(frame)
                line_status = processor.check_lines(frame.shape, pair.left_bottle, pair.right_bottle)

                if line_status.right_passed_now and not right_passed_latched:
                    right_passed_latched = True

                    # As requested: when first/right bottle intersects right line,
                    # move to left-line state and set Pin2 HIGH.
                    gpio.set_second_output(True)

                    state = AppState.INTERSECT_LEFT_LINE_WITH_SECOND_BOTTLE
                    event_text = "FIRST/RIGHT bottle passed RIGHT line | PIN2 HIGH"
                    log_event(output_dir, cycle_id, state, "first_right_bottle_passed_right_line_pin2_high", frame_index)

            elif state == AppState.INTERSECT_LEFT_LINE_WITH_SECOND_BOTTLE:
                if not detector.is_loaded:
                    detector.load()

                pair = processor.detect_pair(frame)
                line_status = processor.check_lines(frame.shape, pair.left_bottle, pair.right_bottle)

                # Keep right latched even if detection flickers.
                if line_status.right_passed_now:
                    right_passed_latched = True

                if line_status.left_passed_now and not left_passed_latched:
                    left_passed_latched = True
                    state = AppState.BOTTLES_EXIT
                    event_text = "SECOND/LEFT bottle passed LEFT line"
                    log_event(output_dir, cycle_id, state, "second_left_bottle_passed_left_line", frame_index)

                    if SETTINGS.app.save_debug_frames:
                        overlay = processor.draw_overlay(
                            frame=frame,
                            state_name=state.name,
                            cycle_id=cycle_id,
                            entry_roi_rect=entry_detector.entry_roi_rect,
                            pair=pair,
                            line_status=line_status,
                            left_passed_latched=left_passed_latched,
                            right_passed_latched=right_passed_latched,
                            event_text="BOTH CONDITIONS TRUE (&&)",
                        )
                        p = save_debug_frame(output_dir, "exit_frames", cycle_id, frame_index, overlay)
                        log_event(output_dir, cycle_id, state, "exit_frame_saved_both_true", frame_index, p)

            elif state == AppState.BOTTLES_EXIT:
                if not detector.is_loaded:
                    # Already unloaded somehow; go back to entry.
                    state = AppState.BOTTLE_ENTRANCE
                    continue

                pair = processor.detect_pair(frame)
                line_status = processor.check_lines(frame.shape, pair.left_bottle, pair.right_bottle)

                if len(pair.detections) == 0:
                    no_detection_count += 1
                else:
                    no_detection_count = 0

                if no_detection_count >= SETTINGS.app.no_detection_exit_frames:
                    event_text = "BOTTLES EXITED | model shutdown"

                    detector.unload()
                    gpio.all_outputs_low()

                    log_event(output_dir, cycle_id, state, "bottles_exited_model_shutdown_outputs_low", frame_index)

                    # Prepare for the next cycle while RUN_ENABLE input remains HIGH.
                    right_passed_latched = False
                    left_passed_latched = False
                    no_detection_count = 0
                    entry_detector.reset_counter()

                    if gpio.is_run_enabled():
                        state = AppState.BOTTLE_ENTRANCE
                        log_event(output_dir, cycle_id, state, "loop_to_bottle_entrance", frame_index)
                    else:
                        state = AppState.IDLE
                        log_event(output_dir, cycle_id, state, "run_enable_low_after_exit", frame_index)

            overlay_frame = processor.draw_overlay(
                frame=frame,
                state_name=state.name,
                cycle_id=cycle_id,
                entry_roi_rect=entry_detector.entry_roi_rect,
                pair=pair,
                line_status=line_status,
                left_passed_latched=left_passed_latched,
                right_passed_latched=right_passed_latched,
                event_text=event_text,
            )

            if video_writer is not None:
                video_writer.write(overlay_frame)

            if display:
                cv2.imshow("Bottle State Machine", overlay_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("b"):
                    # Rebuild background manually if the scene changed.
                    entry_detector.build_background_from_camera(camera)
                    log_event(output_dir, cycle_id, state, "background_rebuilt_by_keyboard", frame_index)

            if args.max_frames > 0 and frame_index >= args.max_frames:
                print("Reached max frames.")
                break

            time.sleep(SETTINGS.app.loop_sleep_s)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: exiting.")

    finally:
        print("Cleaning up...")
        gpio.all_outputs_low()
        detector.unload()
        camera.close()
        gpio.close()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
