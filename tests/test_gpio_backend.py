# tests/test_gpio_backend.py

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from settings import SETTINGS
from src.gpio_backend import GpioBackend, GpioBackendSettings


def main():
    s = SETTINGS.gpio

    gpio = GpioBackend(
        GpioBackendSettings(
            enabled=s.enabled,
            dummy_mode=s.dummy_mode,
            run_enable_input_pin=s.run_enable_input_pin,
            first_bottle_output_pin=s.first_bottle_output_pin,
            second_bottle_output_pin=s.second_bottle_output_pin,
            input_pull_up=s.input_pull_up,
            bounce_time_s=s.bounce_time_s,
            dummy_initial_run_enabled=s.dummy_initial_run_enabled,
        )
    )

    gpio.open()

    try:
        print("Testing outputs...")
        gpio.set_first_output(True)
        time.sleep(1)
        gpio.set_first_output(False)

        gpio.set_second_output(True)
        time.sleep(1)
        gpio.set_second_output(False)

        print("Now watching run-enable input for 20 seconds...")
        t0 = time.time()
        while time.time() - t0 < 20:
            print("run enabled:", gpio.is_run_enabled())
            time.sleep(0.5)

    finally:
        gpio.close()


if __name__ == "__main__":
    main()
