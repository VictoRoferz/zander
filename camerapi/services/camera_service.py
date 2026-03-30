"""
Camera Service for Server 1 (Raspberry Pi 3)
Handles image capture from Basler camera via pylon or fallback image.
Falls back to OpenCV VideoCapture if pypylon is not available.
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from config.settings import settings
from utils.logger import setup_logger


logger = setup_logger(__name__, level=settings.log_level)

# Try to import pypylon for Basler camera support
try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
    logger.info("pypylon available - Basler camera support enabled")
except ImportError:
    PYPYLON_AVAILABLE = False
    logger.warning("pypylon not available - falling back to OpenCV VideoCapture")


class CameraService:
    """
    Service for capturing images from Basler camera or fallback source.
    Uses pypylon with lazy initialization — camera stays open between captures.
    Falls back to cv2.VideoCapture if pypylon is not available.
    """

    def __init__(self):
        self.use_camera = settings.use_camera
        self.camera_index = settings.camera_index
        self.fallback_path = Path(settings.fallback_image_path)
        self.temp_dir = settings.temp_dir

        # Pylon objects — lazy initialized
        self._tl_factory = None
        self._camera = None
        self._converter = None

        logger.info(
            f"CameraService initialized: use_camera={self.use_camera}, "
            f"pypylon={PYPYLON_AVAILABLE}"
        )

    def capture(self) -> Optional[Path]:
        """Capture image from camera or use fallback."""
        if self.use_camera:
            image_path = self._capture_from_camera()
            if image_path:
                logger.info(f"Image captured from camera: {image_path}")
                return image_path
            else:
                logger.warning("Camera capture failed, falling back to sample image")
                return self._use_fallback()
        else:
            logger.info("Camera disabled, using fallback image")
            return self._use_fallback()

    def _capture_from_camera(self) -> Optional[Path]:
        """Capture image. Uses pypylon for Basler, falls back to OpenCV."""
        if PYPYLON_AVAILABLE:
            return self._capture_basler()
        return self._capture_opencv()

    # ----- Basler / pypylon -----

    @staticmethod
    def _align_node(node, target: int) -> int:
        try:
            inc = node.GetInc()
            mn = node.GetMin()
            mx = node.GetMax()
            val = max(mn, min(target, mx))
            aligned = val - (val % inc)
            return max(mn, min(aligned, mx))
        except Exception:
            # Fallback: clamp only
            try:
                return max(node.GetMin(), min(target, node.GetMax()))
            except Exception:
                return target

    def _init_pylon(self) -> None:
        """
        Lazy initialization of pylon factory, camera, and format converter.
        Configures resolution, pixel format, throughput, and packet settings
        for single-shot capture optimized for Pi3's 100 Mbit/s Ethernet.
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

        # Format converter: always outputs BGR8 for OpenCV compatibility
        # When camera sends BayerRG8, the converter handles debayering on the Pi CPU
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._camera.Open()

        # 0) Pixel format — BayerRG8 saves 66% bandwidth vs BGR8
        if settings.camera_pixel_format == "bayer":
            try:
                self._camera.PixelFormat.SetValue("BayerRG8")
                logger.info("Set PixelFormat to BayerRG8 (1 byte/px, debayer on CPU)")
            except Exception as e:
                logger.warning(f"Could not set BayerRG8, keeping default: {e}")
        else:
            logger.info("Using color pixel format (BGR8, 3 bytes/px)")

        # 1) Resolution — use camera max or custom values
        if settings.camera_resolution_mode == "max":
            target_width = self._camera.Width.GetMax()
            target_height = self._camera.Height.GetMax()
            logger.info(f"Resolution mode 'max': using camera maximum {target_width}x{target_height}")
        else:
            target_width = settings.camera_width
            target_height = settings.camera_height
            logger.info(f"Resolution mode 'custom': targeting {target_width}x{target_height}")

        try:
            w = CameraService._align_node(self._camera.Width, target_width)
            h = CameraService._align_node(self._camera.Height, target_height)
            self._camera.Width.SetValue(w)
            self._camera.Height.SetValue(h)
            logger.info(
                f"Set camera ROI to {w}x{h} "
                f"(range {self._camera.Width.GetMin()}-{self._camera.Width.GetMax()} x "
                f"{self._camera.Height.GetMin()}-{self._camera.Height.GetMax()})"
            )
        except Exception as e:
            logger.warning(f"Could not set resolution: {e}")

        # 2) Host-side buffers (low count for single-shot)
        try:
            buf_count = settings.camera_max_num_buffer
            self._camera.MaxNumBuffer = buf_count
            logger.info(f"Set MaxNumBuffer to {buf_count}")
        except Exception as e:
            logger.warning(f"Could not set MaxNumBuffer: {e}")

        # 3) Packet size (standard MTU, Pi3 can't do jumbo frames)
        try:
            self._camera.GevSCPSPacketSize.SetValue(settings.camera_packet_size)
            logger.info(f"Set GevSCPSPacketSize to {self._camera.GevSCPSPacketSize.GetValue()}")
        except Exception as e:
            logger.warning(f"Could not set GevSCPSPacketSize: {e}")

        # 4) Inter-packet delay — prevents burst overload on Pi3's slow NIC
        try:
            self._camera.GevSCPD.SetValue(settings.camera_inter_packet_delay_ticks)
            logger.info(f"Set GevSCPD (inter-packet delay) to {settings.camera_inter_packet_delay_ticks}")
        except Exception as e:
            logger.warning(f"Could not set GevSCPD: {e}")

        # 5) Frame retention (time to wait for missing packets within a frame)
        try:
            self._camera.FrameRetention.SetValue(settings.camera_frame_retention_us)
            logger.info(f"Set FrameRetention to {self._camera.FrameRetention.GetValue()} µs")
        except Exception as e:
            logger.warning(f"Could not set FrameRetention (not supported on all models): {e}")

        # 6) Throughput limit
        try:
            max_limit = self._camera.DeviceLinkThroughputLimit.GetMax()
            limit = min(max_limit, settings.camera_throughput_bps)
            self._camera.DeviceLinkThroughputLimit.SetValue(limit)
            logger.info(f"Set DeviceLinkThroughputLimit to {limit / 1_000_000:.1f} Mbit/s")
        except Exception as e:
            logger.warning(f"Could not set DeviceLinkThroughputLimit: {e}")

        logger.info("Camera configured for single-shot capture (no continuous grabbing)")

    def _capture_basler(self) -> Optional[Path]:
        """Capture a single image from Basler camera using pypylon single-shot.
        Retries once if the first grab fails (common after cold initialization)."""
        try:
            self._init_pylon()

            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                grab_result = self._camera.GrabOne(
                    settings.camera_grab_timeout_ms,
                    pylon.TimeoutHandling_ThrowException
                )

                try:
                    if grab_result.GrabSucceeded():
                        image = self._converter.Convert(grab_result)
                        frame = image.GetArray()
                        return self._save_frame(frame)
                    else:
                        logger.warning(
                            f"Basler grab attempt {attempt}/{max_attempts} failed: "
                            f"{grab_result.ErrorDescription}"
                        )
                finally:
                    grab_result.Release()

            logger.error("Basler grab failed after all attempts")
            return None

        except Exception as e:
            logger.error(f"Basler capture exception: {e}", exc_info=True)
            return None

    # ----- OpenCV fallback -----

    def _capture_opencv(self) -> Optional[Path]:
        """Capture image using OpenCV (fallback for non-Basler cameras)."""
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
            cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)

            ret, frame = cap.read()

            if not ret or frame is None:
                logger.error("Failed to capture frame from camera")
                return None

            return self._save_frame(frame)

        except Exception as e:
            logger.error(f"OpenCV capture exception: {e}", exc_info=True)
            return None

        finally:
            if cap is not None:
                cap.release()

    # ----- Common helpers -----

    def _save_frame(self, frame: np.ndarray) -> Optional[Path]:
        """Save a numpy frame as JPEG and return the path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"capture_{timestamp}.jpg"
        image_path = self.temp_dir / filename

        success = cv2.imwrite(
            str(image_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not success:
            logger.error(f"Failed to save image to {image_path}")
            return None

        height, width = frame.shape[:2]
        logger.debug(
            f"Captured {width}x{height} image, size: "
            f"{image_path.stat().st_size / 1024:.1f} KB"
        )

        return image_path

    def _use_fallback(self) -> Path:
        """Use fallback image for testing without camera."""
        if not self.fallback_path.exists():
            logger.warning(
                f"Fallback image not found at {self.fallback_path}, "
                "creating test image"
            )
            self._create_test_image()

        if not self.fallback_path.exists():
            raise RuntimeError(
                f"Fallback image not available at {self.fallback_path}"
            )

        logger.info(f"Using fallback image: {self.fallback_path}")
        return self.fallback_path

    def _create_test_image(self) -> None:
        """Create a minimal test image for development/testing."""
        try:
            width, height = 1920, 1080
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:] = (20, 40, 20)

            text = f"TEST IMAGE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(image, text, (50, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.imwrite(str(self.fallback_path), image)
            logger.info(f"Created test image at {self.fallback_path}")

        except Exception as e:
            logger.error(f"Failed to create test image: {e}", exc_info=True)

    def cleanup(self, image_path: Path) -> None:
        """Clean up temporary captured image."""
        try:
            if image_path.exists() and image_path.parent == self.temp_dir:
                image_path.unlink()
                logger.debug(f"Cleaned up temporary image: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {image_path}: {e}")

    def get_status(self) -> dict:
        """Get camera service status."""
        status = {
            "use_camera": self.use_camera,
            "camera_backend": "pypylon" if PYPYLON_AVAILABLE else "opencv",
            "camera_index": self.camera_index,
            "resolution_mode": settings.camera_resolution_mode,
            "pixel_format": settings.camera_pixel_format,
            "throughput_limit_mbps": settings.camera_throughput_bps / 1_000_000,
            "fallback_available": self.fallback_path.exists(),
            "temp_dir": str(self.temp_dir),
            "temp_dir_exists": self.temp_dir.exists(),
        }

        if self.use_camera and PYPYLON_AVAILABLE:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                status["camera_available"] = len(devices) > 0
                if devices:
                    status["camera_name"] = devices[0].GetFriendlyName()
                    status["camera_serial"] = devices[0].GetSerialNumber()
            except Exception:
                status["camera_available"] = False
        elif self.use_camera:
            cap = cv2.VideoCapture(self.camera_index)
            status["camera_available"] = cap.isOpened()
            cap.release()

        return status


# Global camera service instance
camera_service = CameraService()