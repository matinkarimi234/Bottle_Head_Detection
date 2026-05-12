# settings.py
#
# Central settings for the live Raspberry Pi bottle detection state machine.
# Edit this file instead of editing main.py.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    device_path: str = "/dev/video0"

    # Your stable Pi settings.
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"

    # Keep these disabled for stable CV thresholds.
    disable_auto_exposure: bool = True
    disable_auto_white_balance: bool = True

    # Camera-dependent. Use:
    #   v4l2-ctl -d /dev/video0 --list-ctrls
    exposure: Optional[int] = None
    gain: Optional[int] = None
    white_balance_temperature: Optional[int] = None

    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None

    warmup_seconds: float = 0.5
    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_dir: str = "/home/pi/Bottle_Head_Detection/models/best_ncnn_model"
    img_size: int = 320
    conf_threshold: float = 0.25
    nms_threshold: float = 0.50
    num_threads: int = 2

    input_name: Optional[str] = None
    output_name: Optional[str] = None

    # From your notebook:
    #   TARGET_CLASS_IDS = [2]  # Without_Cap
    # Set to None if you want all classes.
    target_class_ids: Optional[Tuple[int, ...]] = (2,)
    target_class_names: Optional[Tuple[str, ...]] = ("Without_Cap",)

    max_targets: int = 2


@dataclass(frozen=True)
class RoiConfig:
    # OpenCV-only entrance ROI: bottom-left.
    entry_roi_fractions: dict = None

    # NCNN detect ROI: bottom-half.
    detect_roi_fractions: dict = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "entry_roi_fractions",
            self.entry_roi_fractions
            or {"x1": 0.00, "y1": 0.50, "x2": 0.45, "y2": 1.00},
        )
        object.__setattr__(
            self,
            "detect_roi_fractions",
            self.detect_roi_fractions
            or {"x1": 0.00, "y1": 0.50, "x2": 1.00, "y2": 1.00},
        )


@dataclass(frozen=True)
class EntryCvConfig:
    background_warmup_frames: int = 30
    diff_threshold: int = 35
    morph_kernel_size: int = 7
    min_foreground_pixels: int = 1200
    min_largest_contour_area: int = 800
    consecutive_frames_required: int = 2


@dataclass(frozen=True)
class LineConfig:
    # These are the line points from your notebook.
    # They look like they were tuned on a larger reference video.
    # The processing code scales them to the current camera frame size.
    reference_frame_width: int = 1280
    reference_frame_height: int = 720

    left_line_p1: Tuple[int, int] = (273, 12)
    left_line_p2: Tuple[int, int] = (374, 640)

    right_line_p1: Tuple[int, int] = (1117, 12)
    right_line_p2: Tuple[int, int] = (1010, 640)

    line_tolerance_px: int = 8


@dataclass(frozen=True)
class GpioConfig:
    # Assumption:
    #   RUN_ENABLE_INPUT_PIN is the external interrupt/run-enable input.
    #   The app loops while this input is HIGH.
    #   When it goes LOW, the app returns to IDLE.
    #
    # The output pins are separated to avoid confusion with the input pin.
    enabled: bool = True
    dummy_mode: bool = False

    run_enable_input_pin: int = 17

    # Goes HIGH after Bottle_Entrance is detected.
    first_bottle_output_pin: int = 27

    # Goes HIGH when the first/right bottle passes the right line and
    # the state moves toward the second/left bottle line check.
    second_bottle_output_pin: int = 22

    input_pull_up: bool = False
    bounce_time_s: float = 0.05

    # Useful for PC testing without Raspberry Pi GPIO.
    dummy_initial_run_enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    loop_sleep_s: float = 0.001

    # After both conditions are true, wait until no detections for this many frames.
    no_detection_exit_frames: int = 8

    # For Pi performance, keep display off in production.
    display: bool = True
    save_debug_frames: bool = True
    write_overlay_video: bool = False

    output_dir: str = str(PROJECT_ROOT / "outputs" / "live_state_machine")

    # If True, NCNN is loaded only after entrance is detected and unloaded
    # after Bottles_Exit. This follows your requested "shutdown model" behavior.
    load_model_on_demand: bool = True


@dataclass(frozen=True)
class OverlayConfig:
    entry_roi_color: Tuple[int, int, int] = (255, 255, 0)
    detect_roi_color: Tuple[int, int, int] = (255, 255, 255)

    left_before_color: Tuple[int, int, int] = (0, 255, 255)
    right_before_color: Tuple[int, int, int] = (255, 0, 255)
    passed_color: Tuple[int, int, int] = (0, 255, 0)

    left_line_color: Tuple[int, int, int] = (0, 255, 255)
    right_line_color: Tuple[int, int, int] = (255, 0, 255)
    center_color: Tuple[int, int, int] = (0, 0, 255)


@dataclass(frozen=True)
class Settings:
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    roi: RoiConfig = RoiConfig()
    entry_cv: EntryCvConfig = EntryCvConfig()
    lines: LineConfig = LineConfig()
    gpio: GpioConfig = GpioConfig()
    app: AppConfig = AppConfig()
    overlay: OverlayConfig = OverlayConfig()


SETTINGS = Settings()
