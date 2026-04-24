"""
Camera acquisition for camerapi.

Returns raw frames (numpy arrays) plus source info. All disk I/O is
delegated to storage_service — this module only talks to the camera.

Priority order:
  1. Basler GigE camera via pypylon (if available + use_camera=true)
  2. OpenCV VideoCapture fallback (if pypylon missing / Basler fails)
  3. sample.jpg fallback (if camera disabled or everything fails)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
    logger.info("pypylon available - Basler camera support enabled")
except ImportError:
    PYPYLON_AVAILABLE = False
    logger.warning("pypylon not available - falling back to OpenCV VideoCapture")


class CameraService:
    """Acquires frames from Basler (preferred) or OpenCV, or uses sample.jpg."""

    def __init__(self) -> None:
        self.use_camera: bool = settings.use_camera
        self.camera_index: int = settings.camera_index
        self.fallback_path: Path = Path(settings.fallback_image_path)

        # Pylon handles — lazy-initialized on first capture
        self._tl_factory: Any = None
        self._camera: Any = None
        self._converter: Any = None

        logger.info(
            f"CameraService initialized: use_camera={self.use_camera}, "
            f"pypylon={PYPYLON_AVAILABLE}"
        )

    # ---- Public API ---------------------------------------------------

    def capture_frame(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Acquire a single frame.

        Returns:
            (frame, source_info) where source_info has at minimum:
              source         : "basler" | "opencv" | "fallback"
              camera_serial  : Optional[str]
              camera_model   : Optional[str]

        Raises:
            RuntimeError if no frame can be obtained (including fallback).
        """
        if self.use_camera:
            try:
                return self._capture_from_live_camera()
            except Exception as e:
                logger.warning(
                    f"Live camera capture failed ({e}); falling back to sample image"
                )

        return self._load_fallback()

    def get_status(self) -> dict[str, Any]:
        """Summary for /api/v1/status — does not actually capture."""
        status: dict[str, Any] = {
            "use_camera": self.use_camera,
            "camera_backend": "pypylon" if PYPYLON_AVAILABLE else "opencv",
            "camera_index": self.camera_index,
            "fallback_available": self.fallback_path.exists(),
        }

        if self.use_camera and PYPYLON_AVAILABLE:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                status["camera_available"] = len(devices) > 0
                if devices:
                    status["camera_name"] = devices[0].GetFriendlyName()
                    status["camera_serial"] = devices[0].GetSerialNumber()
                    status["camera_model"] = devices[0].GetModelName()
            except Exception as e:
                status["camera_available"] = False
                status["camera_error"] = str(e)
        elif self.use_camera:
            cap = cv2.VideoCapture(self.camera_index)
            status["camera_available"] = cap.isOpened()
            cap.release()
        else:
            status["camera_available"] = False

        return status

    # ---- Live camera path --------------------------------------------

    def _capture_from_live_camera(self) -> tuple[np.ndarray, dict[str, Any]]:
        if PYPYLON_AVAILABLE:
            return self._capture_basler()
        return self._capture_opencv()

    # ---- Basler / pypylon --------------------------------------------

    def _init_pylon(self) -> None:
        """
        Lazy init of pylon factory, camera, format converter, and WiFi-tuned
        acquisition settings. Called once; camera stays open for subsequent grabs.
        """
        if self._camera is not None:
            return

        self._tl_factory = pylon.TlFactory.GetInstance()
        devices = self._tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found")

        device = devices[0]
        logger.info(
            f"Using Basler camera: model={device.GetModelName()}, "
            f"serial={device.GetSerialNumber()}"
        )

        self._camera = pylon.InstantCamera(self._tl_factory.CreateDevice(device))

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._camera.Open()

        # BayerRG8: 1 byte/pixel over the wire; debayer to BGR8 on CPU.
        self._try_set(lambda: self._camera.PixelFormat.SetValue("BayerRG8"), "PixelFormat=BayerRG8")

        # Max resolution for PCB inspection.
        try:
            max_w = self._camera.Width.GetMax()
            max_h = self._camera.Height.GetMax()
            self._camera.Width.SetValue(max_w)
            self._camera.Height.SetValue(max_h)
            logger.info(f"Set camera to max resolution: {max_w}x{max_h}")
        except Exception as e:
            logger.warning(f"Could not set max resolution: {e}")

        # WiFi-friendly throughput tuning.
        self._try_set(lambda: self._camera.GevSCPSPacketSize.SetValue(1400), "GevSCPSPacketSize=1400")
        self._try_set(lambda: self._camera.GevSCPD.SetValue(250000), "GevSCPD=250000")

        try:
            max_limit = self._camera.DeviceLinkThroughputLimit.GetMax()
            limit = min(max_limit, 2_000_000)
            self._camera.DeviceLinkThroughputLimit.SetValue(limit)
            logger.info(f"Set DeviceLinkThroughputLimit to {limit / 1_000_000:.1f} Mbit/s")
        except Exception as e:
            logger.warning(f"Could not set DeviceLinkThroughputLimit: {e}")

        try:
            self._camera.MaxNumBuffer = 64
        except Exception as e:
            logger.warning(f"Could not set MaxNumBuffer: {e}")

        # Stream-grabber tuning (packet resend / long timeouts for WiFi).
        try:
            sn = self._camera.GetStreamGrabberNodeMap()
            for name, val in [
                ("EnableResend", True),
                ("PacketTimeout", 100000),
                ("FrameRetention", 5000000),
                ("MaxNumResendsPerBuffer", 500),
            ]:
                try:
                    node = sn.GetNode(name)
                    if node is not None:
                        node.SetValue(val)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not access stream grabber nodemap: {e}")

        logger.info("Camera configured for single-shot capture over WiFi")

    @staticmethod
    def _try_set(fn, label: str) -> None:
        try:
            fn()
            logger.info(f"Set {label}")
        except Exception as e:
            logger.warning(f"Could not set {label}: {e}")

    def _capture_basler(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Grab one frame with retry. Returns (frame, source_info)."""
        self._init_pylon()

        info: dict[str, Any] = {
            "source": "basler",
            "camera_serial": None,
            "camera_model": None,
        }
        try:
            device_info = self._camera.GetDeviceInfo()
            info["camera_serial"] = device_info.GetSerialNumber()
            info["camera_model"] = device_info.GetModelName()
        except Exception:
            pass

        max_attempts = 3
        last_error: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            logger.info(f"GrabOne attempt {attempt}/{max_attempts} (30s timeout)...")
            grab_result = self._camera.GrabOne(
                30000, pylon.TimeoutHandling_ThrowException
            )
            try:
                if grab_result.GrabSucceeded():
                    image = self._converter.Convert(grab_result)
                    frame = image.GetArray()
                    logger.info(f"Grab succeeded on attempt {attempt}")
                    return frame, info
                last_error = grab_result.ErrorDescription
                logger.warning(
                    f"Grab attempt {attempt}/{max_attempts} failed: {last_error}"
                )
            finally:
                grab_result.Release()

        raise RuntimeError(f"Basler grab failed after {max_attempts} attempts: {last_error}")

    # ---- OpenCV fallback ---------------------------------------------

    def _capture_opencv(self) -> tuple[np.ndarray, dict[str, Any]]:
        cap = cv2.VideoCapture(self.camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV could not open camera index {self.camera_index}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
            cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("OpenCV capture returned empty frame")

            return frame, {"source": "opencv", "camera_serial": None, "camera_model": None}
        finally:
            cap.release()

    # ---- sample.jpg fallback -----------------------------------------

    def _load_fallback(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.fallback_path.exists():
            self._create_test_image()

        if not self.fallback_path.exists():
            raise RuntimeError(f"Fallback image missing: {self.fallback_path}")

        frame = cv2.imread(str(self.fallback_path))
        if frame is None:
            raise RuntimeError(f"Could not read fallback image: {self.fallback_path}")

        logger.info(f"Using fallback image: {self.fallback_path}")
        return frame, {"source": "fallback", "camera_serial": None, "camera_model": None}

    def _create_test_image(self) -> None:
        """Generate a minimal test image so development works without a camera."""
        width, height = 1920, 1080
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (20, 40, 20)
        cv2.putText(
            image,
            "TEST IMAGE",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(self.fallback_path), image)
        logger.info(f"Created test image at {self.fallback_path}")


camera_service = CameraService()
