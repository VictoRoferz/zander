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

import socket
import struct
import subprocess
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


# Transport profiles: pylon node values tuned per network medium.
# Ethernet: full throughput, no spacing, small retransmit window.
# WiFi: throttled, large inter-packet delay, generous resends.
ETHERNET_PROFILE: dict[str, Any] = {
    "label": "ethernet",
    "packet_size": 1500,           # standard MTU; raise to 9000 with jumbo frames
    "inter_packet_delay": 0,       # GigE link is fast enough
    "throughput_limit_bps": None,  # None = use camera max
    "max_num_buffer": 16,
    "stream_grabber": {
        "EnableResend": True,
        "PacketTimeout": 20000,    # 20 ms
        "FrameRetention": 200000,  # 200 ms
        "MaxNumResendsPerBuffer": 5,
    },
}
WIFI_PROFILE: dict[str, Any] = {
    "label": "wifi",
    "packet_size": 1400,
    "inter_packet_delay": 250000,
    "throughput_limit_bps": 2_000_000,  # 2 Mbit/s
    "max_num_buffer": 64,
    "stream_grabber": {
        "EnableResend": True,
        "PacketTimeout": 100000,    # 100 ms
        "FrameRetention": 5000000,  # 5 s
        "MaxNumResendsPerBuffer": 500,
    },
}


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
        Lazy init of pylon factory, camera, format converter, and a tuned
        acquisition profile (Ethernet vs WiFi). Called once; camera stays
        open for subsequent grabs.
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

        # Pick transport profile BEFORE opening the camera so we can log it
        # alongside the rest of the settings.
        profile = self._select_profile(device)
        logger.info(f"Transport profile: {profile['label']}")

        self._camera = pylon.InstantCamera(self._tl_factory.CreateDevice(device))

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._camera.Open()

        # BayerRG8: 1 byte/pixel over the wire; debayer to BGR8 on CPU.
        self._try_set(
            lambda: self._camera.PixelFormat.SetValue("BayerRG8"),
            "PixelFormat=BayerRG8",
        )

        # Max resolution for PCB inspection.
        try:
            max_w = self._camera.Width.GetMax()
            max_h = self._camera.Height.GetMax()
            self._camera.Width.SetValue(max_w)
            self._camera.Height.SetValue(max_h)
            logger.info(f"Set camera to max resolution: {max_w}x{max_h}")
        except Exception as e:
            logger.warning(f"Could not set max resolution: {e}")

        # Apply the chosen profile.
        self._apply_profile(profile)

        logger.info(
            f"Camera configured for single-shot capture ({profile['label']} profile)"
        )

    def _select_profile(self, device: Any) -> dict[str, Any]:
        """
        Resolve which acquisition profile to use.

        Order of precedence:
          1. settings.camera_transport == "ethernet" / "wifi" → forced.
          2. "auto" → look up the OS route to the camera's IP and pick
             ethernet for eth*/en*, wifi for wl*.
          3. Anything else / detection fails → fall back to wifi (safer).
        """
        forced = (settings.camera_transport or "auto").strip().lower()
        if forced == "ethernet":
            logger.info("camera_transport=ethernet (forced via .env)")
            return ETHERNET_PROFILE
        if forced == "wifi":
            logger.info("camera_transport=wifi (forced via .env)")
            return WIFI_PROFILE

        camera_ip = self._camera_ip(device)
        if not camera_ip:
            logger.warning("Could not read camera IP — falling back to WiFi profile")
            return WIFI_PROFILE

        iface = self._interface_for_ip(camera_ip)
        logger.info(f"Camera IP {camera_ip} reachable via interface '{iface or '?'}'")
        if iface and (iface.startswith("eth") or iface.startswith("en")):
            return ETHERNET_PROFILE
        if iface and iface.startswith("wl"):
            return WIFI_PROFILE
        logger.warning(
            f"Unrecognized interface '{iface}' for camera IP {camera_ip} — "
            "falling back to WiFi profile"
        )
        return WIFI_PROFILE

    @staticmethod
    def _camera_ip(device: Any) -> Optional[str]:
        """Best-effort extraction of the camera's IPv4 from pylon DeviceInfo."""
        getter = getattr(device, "GetIpAddress", None)
        if getter is None:
            return None
        try:
            value = getter()
        except Exception as e:
            logger.debug(f"GetIpAddress failed: {e}")
            return None
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int):
            try:
                return socket.inet_ntoa(struct.pack("!I", value))
            except Exception:
                return None
        return None

    @staticmethod
    def _interface_for_ip(camera_ip: str) -> Optional[str]:
        """Run `ip -o route get <ip>` and parse the `dev <iface>` token."""
        try:
            result = subprocess.run(
                ["ip", "-o", "route", "get", camera_ip],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"ip route get failed: {e}")
            return None
        parts = result.stdout.split()
        if "dev" in parts:
            idx = parts.index("dev") + 1
            if idx < len(parts):
                return parts[idx]
        return None

    def _apply_profile(self, profile: dict[str, Any]) -> None:
        """Push profile values into the camera's GenICam nodes."""
        # Packet size + inter-packet delay (camera-side).
        self._try_set(
            lambda: self._camera.GevSCPSPacketSize.SetValue(profile["packet_size"]),
            f"GevSCPSPacketSize={profile['packet_size']}",
        )
        self._try_set(
            lambda: self._camera.GevSCPD.SetValue(profile["inter_packet_delay"]),
            f"GevSCPD={profile['inter_packet_delay']}",
        )

        # Throughput cap.
        try:
            max_limit = self._camera.DeviceLinkThroughputLimit.GetMax()
            target = profile["throughput_limit_bps"]
            limit = max_limit if target is None else min(max_limit, target)
            self._camera.DeviceLinkThroughputLimit.SetValue(limit)
            logger.info(
                f"DeviceLinkThroughputLimit = {limit / 1_000_000:.1f} Mbit/s "
                f"(max {max_limit / 1_000_000:.1f})"
            )
        except Exception as e:
            logger.warning(f"Could not set DeviceLinkThroughputLimit: {e}")

        # Host-side buffers.
        try:
            self._camera.MaxNumBuffer = profile["max_num_buffer"]
            logger.info(f"MaxNumBuffer = {profile['max_num_buffer']}")
        except Exception as e:
            logger.warning(f"Could not set MaxNumBuffer: {e}")

        # Stream-grabber knobs (host-side resend behavior).
        try:
            sn = self._camera.GetStreamGrabberNodeMap()
            for name, val in profile["stream_grabber"].items():
                try:
                    node = sn.GetNode(name)
                    if node is not None:
                        node.SetValue(val)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not access stream grabber nodemap: {e}")

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
