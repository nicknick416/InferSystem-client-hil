"""30 Hz tactile ring buffer for the QGF client (additive module; touches no existing code).

Each sensor gets its own capture thread clocked by the camera itself (blocking
``VideoCapture.read()`` at CAP_PROP_FPS=30) — appends happen on the SENSOR frame clock,
decoupled from the inference request rate (QGF doc §11.10: the training Δt grid is a strict
33 ms lattice; request-time sampling would misalign it).

Ring entries are JPEG-encoded at capture time (~1 ms/frame, amortized across the thread) at
the NATIVE resolution — letterboxing to 224 happens model-side with the same openpi
resize_with_pad the training pipeline used. Channel contract: frames are encoded from the
native BGR capture; the server decodes and converts to RGB.

Baseline semantics: the first frame captured after ``clear()`` becomes the episode baseline
(matches the server/policy convention "baseline = first frame after reset").
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("tactile_ring")


class TactileRingBuffer:
    def __init__(
        self,
        name: str,
        wire_key: str,
        device_path: str,
        *,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        maxlen: int = 30,
        jpeg_quality: int = 92,
    ):
        self.name = name
        self.wire_key = wire_key
        self.device_path = device_path
        self.fps = fps
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._ring: collections.deque[bytes] = collections.deque(maxlen=maxlen)
        self._baseline_jpg: bytes | None = None
        self._latest_bgr: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self._capture_time = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device_path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"tactile sensor {self.name!r} failed to open {self.device_path!r} — "
                "check /dev/v4l/by-path/ and Config/flexiv_vtla_qgf.yaml"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"tactile-ring-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            # A read() on a dead sensor blocks up to OpenCV's 10 s V4L2 select timeout. The
            # thread MUST exit before the interpreter finalizes: a daemon thread waking after
            # finalization gets pthread_exit'ed inside OpenCV's C++ frames and aborts the
            # process ("FATAL: exception not rethrown"). Healthy sensors join in <50 ms.
            t.join(timeout=12.0)
            self._thread = None
        if self._cap is not None and (t is None or not t.is_alive()):
            self._cap.release()
        self._cap = None

    def _run(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        fail_streak = 0
        while self._running:
            ok, frame = self._cap.read()  # blocking; the camera IS the 30 Hz clock
            if not ok:
                fail_streak += 1
                logger.warning("tactile %s: frame grab failed (streak %d)", self.name, fail_streak)
                if fail_streak >= 30:  # ~1 s dead: reopen — survives transient USB re-enumeration
                    logger.warning("tactile %s: reopening %s", self.name, self.device_path)
                    try:
                        self._cap.release()
                        time.sleep(0.5)
                        self._cap = cv2.VideoCapture(self.device_path)
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except cv2.error:
                        logger.exception("tactile %s: reopen failed", self.name)
                    fail_streak = 0
                time.sleep(0.05)
                continue
            fail_streak = 0
            captured_at = time.monotonic()
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpg = buf.tobytes()
            with self._lock:
                if self._baseline_jpg is None:
                    self._baseline_jpg = jpg  # first frame after clear() = episode baseline
                self._ring.append(jpg)
                self._latest_bgr = frame
                self._capture_time = captured_at
                self.frame_count += 1

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Episode boundary: atomically drop history AND re-arm the baseline (doc §11.10)."""
        with self._lock:
            self._ring.clear()
            self._baseline_jpg = None
            self._latest_bgr = None
            self._capture_time = 0.0

    def snapshot(self) -> tuple[list[bytes], bytes | None]:
        with self._lock:
            return list(self._ring), self._baseline_jpg

    def latest_bgr_timed(self):
        # Published frames are immutable: the capture thread replaces the array.
        # Return the reference and timestamp together without waiting or copying.
        if not self._lock.acquire(False):
            return None, None
        try:
            return self._latest_bgr, self._capture_time
        finally:
            self._lock.release()

    def latest_bgr(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_bgr is None else self._latest_bgr.copy()

    def wait_for_frames(self, n: int = 1, timeout_s: float = 3.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self._lock:
                if len(self._ring) >= n:
                    return True
            time.sleep(0.05)
        return False

    def snapshot_timed(self):
        """Same-lock history/baseline/current capture timestamp for HIL pairing."""
        with self._lock:
            return list(self._ring), self._baseline_jpg, self._capture_time
