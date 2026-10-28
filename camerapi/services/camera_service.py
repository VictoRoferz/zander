"""
Camera acquisition for camerapi.

Returns raw frames (numpy arrays) plus source info. All disk I/O is
delegated to storage_service — this module only talks to the camera.

Priority order:
  1. Basler GigE camera via pypylon (if available + use_camera=true)
  2. OpenCV VideoCapture fallback (if pypylon missing / Basler fails)
  3. sample.jpg fallback (if camera disabled or everything fails)

This is THE capture path while the live preview is off. With the preview
on, services/live_camera.py owns the camera instead and reuses this class
through a private instance (open_pylon / reset_pylon); it also hands this
singleton a capture its live grabs got back incomplete (then release()).

Routes call capture_frame() from the threadpool, so the keypad and the
dashboard can race: every camera access here is serialized by self._lock.
"""
from __future__ import annotations

import socket
import struct
import subprocess
import sys
import threading
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


# The one-shot path's GrabOne timeout. Also the live path's budget on the WiFi
# profile, where one full frame takes ~8 s over the throttled link.
ONE_SHOT_GRAB_TIMEOUT_MS = 30000

# Transport profiles: pylon node values tuned per network medium.
# Ethernet: throttled a little below line rate, no spacing, quick resends.
# WiFi: throttled, large inter-packet delay, generous resends.
# Units (pylon): PacketTimeout and FrameRetention in MILLISECONDS (Basler
# defaults 40 / 200); DeviceLinkThroughputLimit in BYTES per second. The old
# values were written as if they were microseconds / bits.
# CAMERA_* settings in .env override these per site (_with_overrides).
ETHERNET_PROFILE: dict[str, Any] = {
    "label": "ethernet",
    "packet_size": 1500,           # standard MTU; raise to 9000 with jumbo frames
    "inter_packet_delay": 0,       # ticks between packets (camera-side spacing)
    # 100 MB/s = 800 Mbit/s: a little below the GigE line rate, so a switch or
    # router between camera and PC has room (the camera bursts a whole frame).
    "throughput_limit_bytes_s": 100_000_000,
    "max_num_buffer": 16,
    "stream_grabber": {
        "EnableResend": True,
        # The driver asks the camera to re-send a missing packet after this
        # long. It was 20000 (20 s), so a lost packet was never re-sent
        # within FrameRetention: every loss became an incomplete frame
        # (on site, 2026-10-28). 40 ms = Basler's default.
        "PacketTimeout": 40,
        # 1 s for all packets of a frame, resends included (the station's
        # value since Oct 2026; the old 200000 = 200 s outlived the 30 s
        # GrabOne budget).
        "FrameRetention": 1000,
        "MaxNumResendsPerBuffer": 25,
    },
}
WIFI_PROFILE: dict[str, Any] = {
    "label": "wifi",
    "packet_size": 1400,
    "inter_packet_delay": 250000,
    "throughput_limit_bytes_s": 2_000_000,  # 2 MB/s (16 Mbit/s)
    "max_num_buffer": 64,
    "stream_grabber": {
        "EnableResend": True,
        "PacketTimeout": 100000,    # 100 s (meant as 100 ms); Pi-era profile
        "FrameRetention": 5000000,  # 83 min (meant as 5 s); Pi-era profile
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
        # Transport profile of the open camera ("ethernet" | "wifi"), else None.
        self.profile_label: Optional[str] = None

        # Serializes all camera access (capture + status probes). Routes run
        # in the threadpool, so two triggers can arrive at the same time.
        self._lock = threading.Lock()

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
        with self._lock:
            if self.use_camera:
                try:
                    return self._capture_from_live_camera()
                except Exception as e:
                    logger.warning(
                        f"Live camera capture failed ({e}); falling back to sample image"
                    )

            return self.load_fallback()

    def get_status(self) -> dict[str, Any]:
        """Summary for /api/v1/status — does not actually capture."""
        status: dict[str, Any] = {
            "use_camera": self.use_camera,
            "camera_backend": "pypylon" if PYPYLON_AVAILABLE else "opencv",
            "camera_index": self.camera_index,
            "fallback_available": self.fallback_path.exists(),
        }

        # A capture holds the lock for up to 3 x 30 s; never make /status wait
        # for it (and never probe the camera while it is grabbing).
        if not self._lock.acquire(timeout=0.5):
            status["camera_busy"] = True
            return status
        try:
            self._probe_camera(status)
        finally:
            self._lock.release()
        return status

    def _probe_camera(self, status: dict[str, Any]) -> None:
        """Fill the availability fields of get_status (caller holds the lock)."""
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

    # ---- Live camera path --------------------------------------------

    def _capture_from_live_camera(self) -> tuple[np.ndarray, dict[str, Any]]:
        if PYPYLON_AVAILABLE:
            return self._capture_basler()
        return self._capture_opencv()

    # ---- Basler / pypylon --------------------------------------------

    def open_pylon(self) -> tuple[Any, Any]:
        """
        Open (or reuse) the Basler camera and return (camera, converter).

        Used by the live camera owner (services/live_camera.py) on its own
        private CameraService instance. Raises on failure with no half-open
        handle left behind, so the next call retries from scratch.
        """
        self._init_pylon()
        return self._camera, self._converter

    def release(self) -> None:
        """
        Close the one-shot handle so another owner may open the camera (the
        live owner after a one-shot capture or when the preview is switched
        on). Waits for a capture in progress; the next capture re-opens.
        """
        with self._lock:
            self.reset_pylon()

    def reset_pylon(self) -> None:
        """
        Release the camera handle (best effort) so the next open starts clean.
        Used after a device loss: a removed GigE device stays unusable until
        it is destroyed and re-created.
        """
        camera = self._camera
        self._camera = None
        self._converter = None
        self.profile_label = None
        if camera is None:
            return
        for name in ("StopGrabbing", "Close", "DestroyDevice"):
            try:
                getattr(camera, name)()
            except Exception as e:
                logger.debug(f"reset_pylon: {name} failed: {e}")

    def _init_pylon(self) -> None:
        """
        Lazy init of pylon factory, camera, format converter, and a tuned
        acquisition profile (Ethernet vs WiFi). Called once; camera stays
        open for subsequent grabs.

        Any failure after the InstantCamera exists closes and drops it before
        re-raising. (A half-open handle used to stick: the next call returned
        early with a camera that had never been opened.)
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
        profile = self._with_overrides(self._select_profile(device))
        logger.info(f"Transport profile: {profile['label']}")

        self._camera = pylon.InstantCamera(self._tl_factory.CreateDevice(device))
        try:
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
        except Exception as e:
            logger.warning(f"Basler init failed ({e}); releasing the camera handle")
            self.reset_pylon()
            raise

        self.profile_label = profile["label"]
        logger.info(
            f"Camera configured for single-shot capture ({profile['label']} profile)"
        )

    def _select_profile(self, device: Any) -> dict[str, Any]:
        """
        Resolve which acquisition profile to use.

        Order of precedence:
          1. settings.camera_transport == "ethernet" / "wifi" -> forced.
          2. "auto" on Windows/macOS -> ethernet (direct-attach GigE is the
             norm on the hub PC; there is no `ip` binary to probe anyway).
          3. "auto" on Linux -> look up the OS route to the camera's IP and
             pick ethernet for eth*/en*, wifi for wl*; detection failure
             falls back to wifi (safer on a Pi-style deployment).
        """
        forced = (settings.camera_transport or "auto").strip().lower()
        if forced == "ethernet":
            logger.info("camera_transport=ethernet (forced via .env)")
            return ETHERNET_PROFILE
        if forced == "wifi":
            logger.info("camera_transport=wifi (forced via .env)")
            return WIFI_PROFILE

        if not sys.platform.startswith("linux"):
            logger.info(
                "Non-Linux host — defaulting to ethernet profile (direct-attach "
                "GigE assumed). Set CAMERA_TRANSPORT=wifi to override."
            )
            return ETHERNET_PROFILE

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

    @staticmethod
    def _with_overrides(profile: dict[str, Any]) -> dict[str, Any]:
        """The profile with the CAMERA_* network settings from .env applied."""
        p = {**profile, "stream_grabber": dict(profile["stream_grabber"])}
        changed = []
        if settings.camera_packet_size is not None:
            p["packet_size"] = settings.camera_packet_size
            changed.append(f"packet_size={p['packet_size']}")
        if settings.camera_inter_packet_delay is not None:
            p["inter_packet_delay"] = settings.camera_inter_packet_delay
            changed.append(f"inter_packet_delay={p['inter_packet_delay']}")
        if settings.camera_throughput_limit_mbps is not None:
            p["throughput_limit_bytes_s"] = int(settings.camera_throughput_limit_mbps * 125_000)
            changed.append(f"throughput={settings.camera_throughput_limit_mbps:g} Mbit/s")
        if settings.camera_packet_timeout_ms is not None:
            p["stream_grabber"]["PacketTimeout"] = settings.camera_packet_timeout_ms
            changed.append(f"PacketTimeout={settings.camera_packet_timeout_ms} ms")
        if settings.camera_frame_retention_ms is not None:
            p["stream_grabber"]["FrameRetention"] = settings.camera_frame_retention_ms
            changed.append(f"FrameRetention={settings.camera_frame_retention_ms} ms")
        if changed:
            logger.info(f"Network settings from .env: {', '.join(changed)}")
        return p

    @staticmethod
    def stream_statistics(camera: Any) -> dict[str, Any]:
        """
        The GigE stream grabber's counters (packets lost, resends, underruns),
        best effort: {} when the driver has none. Logged after a failed grab,
        so the log says WHY a frame came back incomplete.
        """
        names = (
            "Statistic_Total_Buffer_Count", "Statistic_Failed_Buffer_Count",
            "Statistic_Buffer_Underrun_Count", "Statistic_Total_Packet_Count",
            "Statistic_Failed_Packet_Count", "Statistic_Resend_Request_Count",
            "Statistic_Resend_Packet_Count",
        )
        stats: dict[str, Any] = {}
        try:
            nodemap = camera.GetStreamGrabberNodeMap()
        except Exception:
            return stats
        for name in names:
            try:
                node = nodemap.GetNode(name)
                if node is not None:
                    stats[name.replace("Statistic_", "")] = node.GetValue()
            except Exception:
                pass
        return stats

    @classmethod
    def log_stream_statistics(cls, camera: Any, where: str) -> None:
        stats = cls.stream_statistics(camera)
        if stats:
            logger.warning(
                f"{where}: stream statistics "
                + ", ".join(f"{k}={v}" for k, v in stats.items())
            )

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

        # Throughput cap (bytes per second). On ace 2 cameras the limit only
        # applies with DeviceLinkThroughputLimitMode = On.
        try:
            max_limit = self._camera.DeviceLinkThroughputLimit.GetMax()
            target = profile["throughput_limit_bytes_s"]
            limit = max_limit if target is None else min(max_limit, target)
            if limit < max_limit:
                self._try_set(
                    lambda: self._camera.DeviceLinkThroughputLimitMode.SetValue("On"),
                    "DeviceLinkThroughputLimitMode=On",
                )
            self._camera.DeviceLinkThroughputLimit.SetValue(limit)
            logger.info(
                f"DeviceLinkThroughputLimit = {limit / 1e6:.1f} MB/s = {limit * 8 / 1e6:.0f} Mbit/s "
                f"(max {max_limit / 1e6:.1f} MB/s)"
            )
        except Exception as e:
            logger.warning(f"Could not set DeviceLinkThroughputLimit: {e}")

        # Host-side buffers.
        try:
            self._camera.MaxNumBuffer = profile["max_num_buffer"]
            logger.info(f"MaxNumBuffer = {profile['max_num_buffer']}")
        except Exception as e:
            logger.warning(f"Could not set MaxNumBuffer: {e}")

        # Stream-grabber knobs (host-side resend behavior). Logged one by one:
        # a value the driver refuses used to vanish without a trace.
        try:
            sn = self._camera.GetStreamGrabberNodeMap()
            applied = []
            for name, val in profile["stream_grabber"].items():
                try:
                    node = sn.GetNode(name)
                    if node is None:
                        logger.warning(f"Stream grabber has no {name} node; left as is")
                        continue
                    node.SetValue(val)
                    applied.append(f"{name}={val}")
                except Exception as e:
                    logger.warning(f"Could not set stream grabber {name}={val}: {e}")
            if applied:
                logger.info(f"Stream grabber: {', '.join(applied)} (times in ms)")
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
            logger.info(
                f"GrabOne attempt {attempt}/{max_attempts} "
                f"({ONE_SHOT_GRAB_TIMEOUT_MS // 1000}s timeout)..."
            )
            grab_result = None
            try:
                # Inside the try: a timeout used to escape the retry loop.
                grab_result = self._camera.GrabOne(
                    ONE_SHOT_GRAB_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException
                )
                if grab_result.GrabSucceeded():
                    image = self._converter.Convert(grab_result)
                    frame = image.GetArray()
                    logger.info(f"Grab succeeded on attempt {attempt}")
                    return frame, info
                last_error = grab_result.ErrorDescription
                logger.warning(
                    f"Grab attempt {attempt}/{max_attempts} failed: {last_error}"
                )
            except pylon.TimeoutException as e:
                # A timeout means the image stream is not arriving (e.g. the
                # Windows-firewall case). Retrying would multiply the 30 s wait
                # past the keypad (40 s) and dashboard (60 s) request timeouts,
                # so fail over after one attempt, exactly as before.
                last_error = f"timeout: {e}"
                logger.warning(f"Grab attempt {attempt}/{max_attempts} timed out: {e}")
                break
            except Exception:
                self._reset_if_removed()
                raise
            finally:
                if grab_result is not None:
                    grab_result.Release()

        if self._camera is not None:
            self.log_stream_statistics(self._camera, "One-shot grab failed")
        self._reset_if_removed()
        raise RuntimeError(f"Basler grab failed after {max_attempts} attempts: {last_error}")

    def _reset_if_removed(self) -> None:
        """
        Drop the handle if pylon flagged the device as removed (cable pulled,
        heartbeat timeout), so the next capture re-opens the camera instead of
        failing on a dead handle until camerapi restarts.
        """
        camera = self._camera
        if camera is None:
            return
        try:
            removed = bool(camera.IsCameraDeviceRemoved())
        except Exception:
            removed = False
        if removed:
            logger.warning("Basler device was removed; the next capture re-opens it")
            self.reset_pylon()

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

    def load_fallback(self) -> tuple[np.ndarray, dict[str, Any]]:
        """sample.jpg as (frame, source_info). File I/O only, never the camera."""
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
