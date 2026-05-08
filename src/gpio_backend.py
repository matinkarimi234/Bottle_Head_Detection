# src/gpio_backend.py

from dataclasses import dataclass
from threading import Event
from typing import Optional


@dataclass
class GpioBackendSettings:
    enabled: bool = True
    dummy_mode: bool = False

    run_enable_input_pin: int = 17
    first_bottle_output_pin: int = 27
    second_bottle_output_pin: int = 22

    input_pull_up: bool = False
    bounce_time_s: float = 0.05
    dummy_initial_run_enabled: bool = True


class GpioBackend:
    """
    GPIO abstraction.

    Real mode uses gpiozero:
        sudo apt install python3-gpiozero python3-lgpio

    Dummy mode is for Windows/PC/no-GPIO testing.
    """

    def __init__(self, settings: GpioBackendSettings):
        self.settings = settings
        self.run_enable_event = Event()

        self._input = None
        self._first_output = None
        self._second_output = None
        self._dummy_run_enabled = settings.dummy_initial_run_enabled

    def open(self) -> None:
        if not self.settings.enabled or self.settings.dummy_mode:
            print("GPIO running in dummy mode.")
            if self._dummy_run_enabled:
                self.run_enable_event.set()
            return

        try:
            from gpiozero import DigitalInputDevice, DigitalOutputDevice
        except ImportError as exc:
            raise ImportError(
                "gpiozero is not installed. Install with:\n"
                "  sudo apt install python3-gpiozero python3-lgpio\n"
                "or set gpio.dummy_mode=True in settings.py."
            ) from exc

        self._input = DigitalInputDevice(
            self.settings.run_enable_input_pin,
            pull_up=self.settings.input_pull_up,
            bounce_time=self.settings.bounce_time_s,
        )

        self._first_output = DigitalOutputDevice(
            self.settings.first_bottle_output_pin,
            active_high=True,
            initial_value=False,
        )

        self._second_output = DigitalOutputDevice(
            self.settings.second_bottle_output_pin,
            active_high=True,
            initial_value=False,
        )

        self._input.when_activated = self._on_run_enable_activated
        self._input.when_deactivated = self._on_run_enable_deactivated

        if self._input.is_active:
            self.run_enable_event.set()

        print("GPIO opened:")
        print(f"  run enable input pin    : {self.settings.run_enable_input_pin}")
        print(f"  first bottle output pin : {self.settings.first_bottle_output_pin}")
        print(f"  second bottle output pin: {self.settings.second_bottle_output_pin}")

    def _on_run_enable_activated(self):
        print("GPIO interrupt: RUN_ENABLE HIGH")
        self.run_enable_event.set()

    def _on_run_enable_deactivated(self):
        print("GPIO interrupt: RUN_ENABLE LOW")

    def is_run_enabled(self) -> bool:
        if not self.settings.enabled or self.settings.dummy_mode:
            return self._dummy_run_enabled

        if self._input is None:
            return False

        return bool(self._input.is_active)

    def set_first_output(self, value: bool) -> None:
        if not self.settings.enabled or self.settings.dummy_mode:
            print(f"[GPIO dummy] first output = {value}")
            return

        if self._first_output is not None:
            self._first_output.value = bool(value)

    def set_second_output(self, value: bool) -> None:
        if not self.settings.enabled or self.settings.dummy_mode:
            print(f"[GPIO dummy] second output = {value}")
            return

        if self._second_output is not None:
            self._second_output.value = bool(value)

    def all_outputs_low(self) -> None:
        self.set_first_output(False)
        self.set_second_output(False)

    def close(self) -> None:
        self.all_outputs_low()

        for dev in [self._input, self._first_output, self._second_output]:
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass

        self._input = None
        self._first_output = None
        self._second_output = None
