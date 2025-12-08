# services/button_capture_service.py

import threading
from typing import Optional

from utils.logger import setup_logger
from config.settings import settings
from services.camera_service import camera_service
from services.upload_service import upload_service

logger = setup_logger(__name__, level=settings.log_level)


class ButtonCaptureService:
    """
    Listens to a GPIO button and triggers camera capture + upload
    whenever the button is pressed.

    Uses RPi.GPIO event detection with debouncing.
    """

    def __init__(self, pin: int = 23):
        # Default: BCM 23 = physischer Pin 16
        self.pin = pin
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """
        Set up GPIO and start listening for button press events.
        Can be called safely multiple times (only first call takes effect).
        """
        if self._started:
            logger.info("ButtonCaptureService already started, skipping init")
            return

        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # FALLING edge: HIGH -> LOW (bei Pull-Up + Taster nach GND)
            GPIO.add_event_detect(
                self.pin,
                GPIO.FALLING,
                callback=self._handle_press,
                bouncetime=300,  # ms debounce
            )

            self._started = True
            logger.info(f"ButtonCaptureService started on GPIO {self.pin} (BCM)")

        except ImportError:
            logger.error(
                "RPi.GPIO not available. ButtonCaptureService will not run. "
                "Are you on a Raspberry Pi?"
            )
        except Exception as e:
            logger.error(f"Failed to start ButtonCaptureService on GPIO {self.pin}: {e}", exc_info=True)

    def _handle_press(self, channel: int) -> None:
        """
        Callback executed in a separate thread when the button is pressed.
        """
        logger.info(f"Button press detected on GPIO {channel}")

        # Ensure only one capture/upload at a time
        if not self._lock.acquire(blocking=False):
            logger.warning("Capture already running, ignoring button press")
            return

        try:
            # 1) Capture image
            logger.info("Capturing image (button trigger)...")
            image_path = camera_service.capture()

            if not image_path:
                logger.error("Button trigger: capture returned no image_path")
                return

            # 2) Upload to Server 2
            logger.info(f"Uploading {image_path.name} to Server 2 (button trigger)...")
            upload_response = upload_service.upload_and_cleanup(image_path)

            logger.info(
                f"Button trigger upload finished: "
                f"status={upload_response.get('status', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"Button-triggered capture/upload failed: {e}", exc_info=True)
        finally:
            # VERY IMPORTANT: release lock so the next press works again
            self._lock.release()


# Global instance
button_capture_service = ButtonCaptureService(pin=23)  # BCM 23 = physischer Pin 16