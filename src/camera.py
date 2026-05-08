# src/camera.py

from dataclasses import dataclass
from typing import Optional, Tuple
import subprocess
import time

import cv2
import numpy as np


@dataclass
class UsbCameraSettings:
    device_index: int = 0
    device_path: str = "/dev/video0"

    width: int = 320
    height: int = 240
    fps: int = 15
    fourcc: str = "MJPG"

    disable_auto_exposure: bool = True
    disable_auto_white_balance: bool = True

    exposure: Optional[int] = None
    gain: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    white_balance_temperature: Optional[int] = None

    warmup_seconds: float = 0.5
    flip_horizontal: bool = False
    flip_vertical: bool = False


class UsbCameraSource:
    def __init__(self, settings: UsbCameraSettings):
        self.settings = settings
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> None:
        self._force_v4l2_format_before_open()
        self._apply_v4l2_controls_before_open()

        self.cap = cv2.VideoCapture(self.settings.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open USB camera index {self.settings.device_index}. "
                f"Check with: v4l2-ctl --list-devices"
            )

        self._apply_opencv_settings()

        if self.settings.warmup_seconds > 0:
            time.sleep(self.settings.warmup_seconds)

        self.is_opened = True
        self.print_current_settings()

    def _force_v4l2_format_before_open(self) -> None:
        cmd = [
            "v4l2-ctl",
            "-d",
            self.settings.device_path,
            f"--set-fmt-video=width={self.settings.width},height={self.settings.height},pixelformat={self.settings.fourcc}",
            f"--set-parm={self.settings.fps}",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print("Warning: could not force V4L2 format/FPS.")
                if result.stderr.strip():
                    print(" ", result.stderr.strip())
        except FileNotFoundError:
            print("Warning: v4l2-ctl not found. Install with: sudo apt install v4l-utils")

    def _apply_opencv_settings(self) -> None:
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.settings.fps)

        if self.settings.disable_auto_exposure:
            # On many UVC cameras: 1 = manual, 3 = auto.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        if self.settings.exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings.exposure)

        if self.settings.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, self.settings.gain)

        if self.settings.brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings.brightness)

        if self.settings.contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings.contrast)

        if self.settings.saturation is not None:
            self.cap.set(cv2.CAP_PROP_SATURATION, self.settings.saturation)

        if self.settings.disable_auto_white_balance:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        if self.settings.white_balance_temperature is not None:
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.settings.white_balance_temperature)

    def _apply_v4l2_controls_before_open(self) -> None:
        controls = {}

        if self.settings.disable_auto_exposure:
            controls["exposure_auto"] = 1

        if self.settings.exposure is not None:
            controls["exposure_absolute"] = self.settings.exposure

        if self.settings.gain is not None:
            controls["gain"] = self.settings.gain

        if self.settings.disable_auto_white_balance:
            controls["white_balance_temperature_auto"] = 0

        if self.settings.white_balance_temperature is not None:
            controls["white_balance_temperature"] = self.settings.white_balance_temperature

        if self.settings.brightness is not None:
            controls["brightness"] = self.settings.brightness

        if self.settings.contrast is not None:
            controls["contrast"] = self.settings.contrast

        if self.settings.saturation is not None:
            controls["saturation"] = self.settings.saturation

        for name, value in controls.items():
            self._run_v4l2_set_control(name, value)

    def _run_v4l2_set_control(self, name: str, value: int) -> None:
        cmd = ["v4l2-ctl", "-d", self.settings.device_path, "-c", f"{name}={value}"]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(f"Warning: could not set V4L2 control {name}={value}")
                if result.stderr.strip():
                    print(" ", result.stderr.strip())
        except FileNotFoundError:
            print("Warning: v4l2-ctl not found. Install with: sudo apt install v4l-utils")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None

        if self.settings.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.settings.flip_vertical:
            frame = cv2.flip(frame, 0)

        return True, frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.is_opened = False

    def print_current_settings(self) -> None:
        if self.cap is None:
            return

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print("USB camera opened:")
        print(f"  device index: {self.settings.device_index}")
        print(f"  device path : {self.settings.device_path}")
        print(f"  resolution  : {int(width)}x{int(height)}")
        print(f"  fps         : {fps}")
        print(f"  fourcc      : {fourcc_text}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
