"""RGB capture/encoding off the control thread; frozen payload at every tick."""
from __future__ import annotations

import threading
import time
from Inference.client import encode_native_png
from Inference.obs_mapping import build_camera_key_map, map_image_keys


class CameraWorker:
    def __init__(self, sensors, config):
        self.sensors = sensors
        # RGB ownership must match runtime; tactile is supplied by rings in snapshot().
        enabled = config.inference.enabled_cameras
        self.names = [n for n in (enabled if enabled is not None else list(config.cameras))
                      if n not in config.tactile]
        self.mapping = build_camera_key_map(config.cameras, self.names, config.tactile)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while self.running:
                images = self.sensors.read_images(self.names)
                stamp = time.monotonic()
                if set(images) != set(self.names):
                    raise RuntimeError(f'Missing RGB views: {set(self.names)-set(images)}')
                payload = {}
                for key, bgr in map_image_keys(images, self.mapping).items():
                    # Preserve native geometry; the model server owns training preprocessing.
                    payload[key] = encode_native_png(bgr, key)
                with self.lock:
                    self.latest = (payload, stamp)
        except Exception as exc:
            self.error = exc

    def snapshot(self, state, rings, prompt, max_age=.25, history_frames=30):
        if self.error:
            raise RuntimeError('RGB capture failed') from self.error
        with self.lock:
            item = self.latest
        if item is None:
            raise RuntimeError('RGB not ready')
        images, rgb_time = item
        now = time.monotonic()
        if now-rgb_time > max_age:
            raise RuntimeError('RGB observation stale')
        payload = dict(images, cmd='predict', state=list(state), prompt=prompt)
        histories, baselines, stamps = {}, {}, {'rgb_read_complete': rgb_time}
        for ring in rings:
            frames, base, stamp = ring.snapshot_timed()
            if not frames or base is None or now-stamp > max_age:
                raise RuntimeError(f'Tactile observation stale/unready: {ring.name}')
            histories[ring.wire_key] = frames[-history_frames:]
            baselines[ring.wire_key] = base
            payload[ring.wire_key] = frames[-1]
            stamps[ring.wire_key] = stamp
        k = min(map(len, histories.values()))
        payload['extra'] = {'tactile_history': {key:v[-k:] for key,v in histories.items()},
                            'tactile_baseline': baselines}
        return payload, stamps

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(12)
            if self.thread.is_alive():
                raise RuntimeError('RGB worker did not stop; do not close sensors concurrently')
