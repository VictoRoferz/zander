"""
A small stand-in for pypylon's `pylon` module, so camerapi's Basler code can
be tested without a camera or the SDK.

Install it BEFORE anything imports services.camera_service:

    import _fake_pylon
    _fake_pylon.install()            # sys.modules["pypylon"] -> this fake
    _fake_pylon.reset(devices=1)     # one camera on the "network"

Behaviour mirrors the parts of pypylon camerapi uses:
  - exceptions: GenericException > RuntimeException / TimeoutException / ...
  - TlFactory.GetInstance().EnumerateDevices() -> the mutable DEVICES list
  - InstantCamera: generic GenICam nodes (SetValue/GetValue/GetMax/GetMin/
    GetInc) reached by attribute; a missing node raises
    LogicalErrorException like pypylon does
  - StartGrabbing/StopGrabbing/IsGrabbing/RetrieveResult/GrabOne,
    IsCameraDeviceRemoved, Open/Close/DestroyDevice
  - GrabOne while grabbing raises (as pylon does), so tests can prove the
    stream is stopped first; a GrabOne whose frame takes longer than its
    timeout (grab_delay) waits the timeout and raises TimeoutException
  - every frame is stamped with a global, increasing counter at pixel [0, 0]
    (read it back with read_stamp)
  - CALLS records (camera_id, method, args) for ordering checks
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
import types
from collections import deque
from typing import Any, Optional

import numpy as np


class GenericException(Exception):
    pass


class RuntimeException(GenericException):
    pass


class TimeoutException(GenericException):
    pass


class LogicalErrorException(GenericException):
    pass


class AccessException(GenericException):
    pass


class OutOfRangeException(GenericException):
    pass


TimeoutHandling_Return = 0
TimeoutHandling_ThrowException = 1
GrabStrategy_OneByOne = 0
GrabStrategy_LatestImageOnly = 1
GrabStrategy_LatestImages = 2
GrabStrategy_UpcomingImage = 3
PixelType_BGR8packed = 35127316
OutputBitAlignment_MsbAligned = 1

# ---- Global test state ---------------------------------------------------------

CALLS: list = []       # (camera_id, method, args)
CAMERAS: list = []     # every InstantCamera created, in order
DEVICES: list = []     # what EnumerateDevices returns
# Knobs applied to the NEXT InstantCamera (see reset()).
CONFIG: dict = {}
_counter = itertools.count(1)
_lock = threading.Lock()


def stamp(frame: np.ndarray, n: int) -> None:
    frame[0, 0, 0] = n % 256
    frame[0, 0, 1] = (n // 256) % 256
    frame[0, 0, 2] = (n // 65536) % 256


def read_stamp(frame: np.ndarray) -> int:
    px = frame[0, 0]
    return int(px[0]) + 256 * int(px[1]) + 65536 * int(px[2])


def next_frame(width: int = 64, height: int = 48) -> tuple[np.ndarray, int]:
    with _lock:
        n = next(_counter)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    stamp(frame, n)
    return frame, n


def calls(camera: Optional["InstantCamera"] = None, method: Optional[str] = None) -> list:
    out = CALLS
    if camera is not None:
        out = [c for c in out if c[0] == camera.id]
    if method is not None:
        out = [c for c in out if c[1] == method]
    return out


def methods(camera: "InstantCamera") -> list:
    return [c[1] for c in CALLS if c[0] == camera.id]


def reset(devices: int = 1, **config: Any) -> None:
    """Fresh state: `devices` cameras on the network, knobs for new cameras.

    Knobs: frame_rate_node ("AcquisitionFrameRate" | "AcquisitionFrameRateAbs"
    | None), fail_open (an exception for Open()), open_gate (a threading.Event
    Open() waits for, at most 5 s), width, height.
    """
    CALLS.clear()
    CAMERAS.clear()
    DEVICES[:] = [DeviceInfo(serial=f"2000000{i}") for i in range(devices)]
    CONFIG.clear()
    CONFIG.update(config)


def last_camera() -> "InstantCamera":
    return CAMERAS[-1]


# ---- Node map ------------------------------------------------------------------


class Node:
    def __init__(self, cam: "InstantCamera", name: str, value: Any = None,
                 vmin: float = 0, vmax: float = 1_000_000_000, inc: float = 1) -> None:
        self._cam = cam
        self._name = name
        self.value = value
        self.vmin, self.vmax, self.inc = vmin, vmax, inc

    def SetValue(self, value: Any) -> None:
        self._cam._log("SetValue", self._name, value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < self.vmin or value > self.vmax:
                raise OutOfRangeException(f"{self._name}={value} out of range")
        self.value = value

    def GetValue(self) -> Any:
        return self.value

    def GetMax(self) -> Any:
        return self.vmax

    def GetMin(self) -> Any:
        return self.vmin

    def GetInc(self) -> Any:
        return self.inc


class NodeMap:
    def __init__(self, nodes: dict) -> None:
        self._nodes = nodes

    def GetNode(self, name: str) -> Optional[Node]:
        return self._nodes.get(name)


# ---- Transport layer -------------------------------------------------------------


class DeviceInfo:
    def __init__(self, model: str = "acA2500-14gc", serial: str = "20000000",
                 ip: str = "192.168.177.95") -> None:
        self._model, self._serial, self._ip = model, serial, ip

    def GetModelName(self) -> str:
        return self._model

    def GetSerialNumber(self) -> str:
        return self._serial

    def GetFriendlyName(self) -> str:
        return f"Basler {self._model} ({self._serial})"

    def GetIpAddress(self) -> str:
        return self._ip


class Device:
    def __init__(self, info: DeviceInfo) -> None:
        self.info = info


class TlFactory:
    _instance: Optional["TlFactory"] = None

    @classmethod
    def GetInstance(cls) -> "TlFactory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def EnumerateDevices(self) -> list:
        return list(DEVICES)

    def CreateDevice(self, info: DeviceInfo) -> Device:
        return Device(info)


# ---- Grab results and conversion -------------------------------------------------


class GrabResult:
    def __init__(self, frame: Optional[np.ndarray] = None, valid: bool = True,
                 ok: bool = True, error: str = "") -> None:
        self.frame = frame
        self._valid = valid
        self._ok = ok
        self.ErrorDescription = error
        self.ErrorCode = 0 if ok else 0xE1000014
        self.released = False

    def IsValid(self) -> bool:
        return self._valid

    def GrabSucceeded(self) -> bool:
        if not self._valid:
            raise RuntimeException("GrabSucceeded on an empty grab result")
        return self._ok

    def Release(self) -> None:
        self.released = True


class PylonImage:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def GetArray(self) -> np.ndarray:
        return self._frame.copy()


class ImageFormatConverter:
    def __init__(self) -> None:
        self.OutputPixelFormat = None
        self.OutputBitAlignment = None

    def Convert(self, result: GrabResult) -> PylonImage:
        if result.frame is None:
            raise RuntimeException("nothing to convert")
        return PylonImage(result.frame)


# ---- Camera ------------------------------------------------------------------------


class InstantCamera:
    _ids = itertools.count(1)

    def __init__(self, device: Device) -> None:
        d = self.__dict__
        d["id"] = next(InstantCamera._ids)
        d["device"] = device
        d["opened"] = False
        d["grabbing"] = False
        d["strategy"] = None
        d["removed"] = False
        d["destroyed"] = False
        d["fail_open"] = CONFIG.get("fail_open")
        d["grabone_timeouts"] = 0       # next N GrabOne calls raise TimeoutException
        d["grabone_incomplete"] = 0     # next N GrabOne calls return an incomplete image
        d["grab_delay"] = 0.0           # seconds GrabOne takes
        d["retrieve_script"] = deque()  # "timeout" | "incomplete" entries for RetrieveResult
        d["MaxNumBuffer"] = 10
        width, height = CONFIG.get("width", 64), CONFIG.get("height", 48)
        nodes = {
            "PixelFormat": Node(self, "PixelFormat", "Mono8"),
            "Width": Node(self, "Width", 32, vmin=16, vmax=width),
            "Height": Node(self, "Height", 24, vmin=16, vmax=height),
            "GevSCPSPacketSize": Node(self, "GevSCPSPacketSize", 1500, vmax=9000),
            "GevSCPD": Node(self, "GevSCPD", 0),
            "DeviceLinkThroughputLimit": Node(self, "DeviceLinkThroughputLimit", 125_000_000, vmax=125_000_000),
            "DeviceLinkThroughputLimitMode": Node(self, "DeviceLinkThroughputLimitMode", "Off"),
            "AcquisitionFrameRateEnable": Node(self, "AcquisitionFrameRateEnable", False),
        }
        rate_node = CONFIG.get("frame_rate_node", "AcquisitionFrameRate")
        if rate_node:
            nodes[rate_node] = Node(self, rate_node, 10.0, vmin=0.1, vmax=100.0)
        d["_nodes"] = nodes
        stream_nodes = {
            name: Node(self, name, None)
            for name in ("EnableResend", "PacketTimeout", "FrameRetention", "MaxNumResendsPerBuffer")
        }
        for name, value in (("Statistic_Failed_Packet_Count", 7), ("Statistic_Resend_Request_Count", 3)):
            stream_nodes[name] = Node(self, name, value)
        d["_stream_nodes"] = NodeMap(stream_nodes)
        CAMERAS.append(self)

    # pypylon resolves unknown attributes as GenICam nodes and raises for a
    # node the camera does not have.
    def __getattr__(self, name: str) -> Node:
        nodes = self.__dict__.get("_nodes", {})
        if name in nodes:
            return nodes[name]
        raise LogicalErrorException(f"Node not existing: {name}")

    def _log(self, method: str, *args: Any) -> None:
        CALLS.append((self.id, method, args))

    def node(self, name: str) -> Node:
        return self._nodes[name]

    # -- lifecycle
    def Open(self) -> None:
        self._log("Open")
        gate = CONFIG.get("open_gate")
        if gate is not None:
            gate.wait(5.0)
        if self.fail_open is not None:
            raise self.fail_open
        self.__dict__["opened"] = True

    def IsOpen(self) -> bool:
        return self.opened

    def Close(self) -> None:
        self._log("Close")
        self.__dict__["opened"] = False
        self.__dict__["grabbing"] = False

    def DestroyDevice(self) -> None:
        self._log("DestroyDevice")
        self.__dict__["opened"] = False
        self.__dict__["grabbing"] = False
        self.__dict__["destroyed"] = True

    def IsCameraDeviceRemoved(self) -> bool:
        return self.removed

    def GetDeviceInfo(self) -> DeviceInfo:
        return self.device.info

    def GetStreamGrabberNodeMap(self) -> NodeMap:
        return self._stream_nodes

    # -- grabbing
    def StartGrabbing(self, strategy: int = GrabStrategy_OneByOne) -> None:
        self._log("StartGrabbing", strategy)
        if self.removed:
            raise RuntimeException("device has been removed")
        if self.grabbing:
            raise RuntimeException("already grabbing")
        self.__dict__["grabbing"] = True
        self.__dict__["strategy"] = strategy

    def StopGrabbing(self) -> None:
        self._log("StopGrabbing")
        self.__dict__["grabbing"] = False

    def IsGrabbing(self) -> bool:
        return self.grabbing

    def _frame(self) -> np.ndarray:
        frame, _ = next_frame(int(self._nodes["Width"].value), int(self._nodes["Height"].value))
        return frame

    def RetrieveResult(self, timeout_ms: int, handling: int = TimeoutHandling_ThrowException) -> GrabResult:
        self._log("RetrieveResult", timeout_ms, handling)
        if self.removed:
            raise RuntimeException("device has been removed")
        if not self.grabbing:
            return GrabResult(valid=False)
        if self.retrieve_script:
            step = self.retrieve_script.popleft()
            if step == "timeout":
                time.sleep(min(timeout_ms, 20) / 1000.0)
                if handling == TimeoutHandling_ThrowException:
                    raise TimeoutException("RetrieveResult timed out")
                return GrabResult(valid=False)
            if step == "incomplete":
                return GrabResult(frame=None, ok=False, error="The buffer was incompletely grabbed")
        time.sleep(0.002)
        return GrabResult(frame=self._frame())

    def GrabOne(self, timeout_ms: int, handling: int = TimeoutHandling_ThrowException) -> GrabResult:
        self._log("GrabOne", timeout_ms, handling)
        if self.removed:
            raise RuntimeException("device has been removed")
        if self.grabbing:
            raise RuntimeException("GrabOne while grabbing: the camera is already grabbing")
        if self.grabone_timeouts > 0:
            self.__dict__["grabone_timeouts"] -= 1
            raise TimeoutException(f"GrabOne timed out after {timeout_ms} ms")
        if self.grabone_incomplete > 0:
            self.__dict__["grabone_incomplete"] -= 1
            return GrabResult(frame=None, ok=False, error="The buffer was incompletely grabbed")
        if self.grab_delay:
            if self.grab_delay * 1000.0 > timeout_ms:
                # The frame needs longer than the timeout (e.g. ~8 s over the
                # 2 Mbit/s WiFi profile): pylon gives up after the timeout.
                time.sleep(timeout_ms / 1000.0)
                raise TimeoutException(f"GrabOne timed out after {timeout_ms} ms")
            time.sleep(self.grab_delay)
        return GrabResult(frame=self._frame())

    def __setattr__(self, name: str, value: Any) -> None:
        # Plain attributes (MaxNumBuffer, test knobs) are stored directly.
        self.__dict__[name] = value


def install() -> types.ModuleType:
    """Make `from pypylon import pylon` return this module."""
    this = sys.modules[__name__]
    package = types.ModuleType("pypylon")
    package.__path__ = []  # a package, so "pypylon.pylon" imports resolve
    package.pylon = this
    package.genicam = this
    sys.modules["pypylon"] = package
    sys.modules["pypylon.pylon"] = this
    sys.modules["pypylon.genicam"] = this
    return this
