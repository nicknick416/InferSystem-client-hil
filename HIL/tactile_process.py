"""Bounded latest-frame shared-memory transport; control never waits for worker."""
import ctypes
import json
import multiprocessing as mp
import time
import numpy as np


def _run(config, frame, header, input_lock, result, output_lock, stop, zero, zero_size):
    import cv2
    from HIL.tactile_stuck_detection import TactileStuckDetector
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    energy_mode = config.get('method') in ('energy', 'local_normal')
    if energy_mode:
        from HIL.tactile_energy import TactileEnergy
        if config.get('method') == 'local_normal':
            from HIL.tactile_normal import LocalNormal
            detector = LocalNormal(config)
        else:
            detector = TactileEnergy(config)
    else:
        detector = TactileStuckDetector(config)
    zero_loaded = False
    seen = generation = None
    while not stop.is_set():
        item = None
        if input_lock.acquire(False):
            try:
                seq, gen, stamp, h, w, hold_epoch, active, opening = header[:]
                if seq and seq != seen:
                    seen = seq
                    h, w = int(h), int(w)
                    pixels = np.frombuffer(frame, np.uint8, count=h*w*3).reshape(h,w,3).copy()
                    if energy_mode and not zero_loaded and zero_size.value:
                        detector.set_zero(bytes(zero[:zero_size.value]))
                        zero_loaded = True
                    item = gen, stamp, pixels, hold_epoch, active, opening
            finally:
                input_lock.release()
        if item is None:
            stop.wait(.002)
            continue
        gen, stamp, pixels, hold_epoch, active, opening = item
        if gen != generation:
            detector.reset()
            generation = gen
        start = time.monotonic()
        try:
            report = (detector.update(pixels, stamp, hold_epoch, bool(active), bool(opening))
                      if energy_mode else detector.update(pixels, stamp))
        except Exception as exc:
            detector.reset()
            report = dict(stuck=False, reason='worker_error', error=str(exc))
        report.update(generation=gen, hold_epoch=hold_epoch, capture_time=stamp,
                      processing_ms=(time.monotonic()-start)*1000)
        data = json.dumps(report).encode()
        if len(data) < len(result)-1 and output_lock.acquire(False):
            try:
                result[:len(data)] = data
                result[len(data)] = 0
            finally:
                output_lock.release()


class TactileProcess:
    CAPACITY = 1920*1080*3

    def __init__(self, config):
        self.config = config
        self.generation = 0
        self.seq = 0
        self.last_stamp = None
        self.mode = None
        self.last = None
        self.process = None
        self.hold_epoch = 0
        self.hold_width = None
        self.opening = False
        self.zero_encoded = None
        self.zero_sent = False

    def start(self):
        if self.process is not None:
            return
        ctx = mp.get_context('spawn')
        self.frame = ctx.RawArray(ctypes.c_ubyte, self.CAPACITY)
        self.header = ctx.RawArray(ctypes.c_double, 8)
        self.input_lock, self.output_lock = ctx.Lock(), ctx.Lock()
        self.result = ctx.RawArray(ctypes.c_ubyte, 8192)
        self.zero = ctx.RawArray(ctypes.c_ubyte, self.CAPACITY)
        self.zero_size = ctx.RawValue(ctypes.c_int, 0)
        self.stop = ctx.Event()
        self.process = ctx.Process(target=_run, args=(self.config, self.frame, self.header,
            self.input_lock, self.result, self.output_lock, self.stop, self.zero, self.zero_size), daemon=True)
        self.process.start()

    def reset(self):
        self.generation += 1
        self.last_stamp = self.last = None

    def calibrate_hold(self, width):
        self.hold_epoch += 1
        self.hold_width = float(width)
        self.opening = False
        self.last = None

    def set_context(self, encoded, command_width):
        if not self.zero_sent and isinstance(encoded, bytes):
            self.zero_encoded = encoded
        if self.hold_width is not None and command_width is not None:
            if command_width > self.hold_width + self.config.get('opening_delta_m', .002):
                self.opening = True
                self.last = None

    def update(self, pixels, stamp, mode, now):
        if mode != self.mode:
            self.reset()
            self.mode = mode
        fallback = dict(stuck=False, reason='waiting_worker', score=None, increase=None, threshold=None)
        if mode != 'policy' and self.config.get('method') not in ('energy', 'local_normal'):
            return dict(fallback, reason='non_policy')
        if self.process is None or not self.process.is_alive():
            return dict(fallback, reason='worker_unavailable')
        valid = (isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and
                 pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.size <= self.CAPACITY and
                 isinstance(stamp, (float,int)) and np.isfinite(stamp) and
                 0 <= now-stamp <= self.config['max_sample_gap_s'])
        if not valid:
            self.reset()
            return dict(fallback, reason='invalid_or_stale_frame')
        # One slot only: overwrite unprocessed frame; never queue or wait.
        if stamp != self.last_stamp and self.input_lock.acquire(False):
            try:
                np.frombuffer(self.frame, np.uint8, count=pixels.size)[:] = pixels.reshape(-1)
                self.seq += 1
                if not self.zero_sent and self.zero_encoded is not None and len(self.zero_encoded) <= self.CAPACITY:
                    self.zero[:len(self.zero_encoded)] = self.zero_encoded
                    self.zero_size.value = len(self.zero_encoded)
                    self.zero_sent = True
                    self.zero_encoded = None
                self.header[:] = [self.seq, self.generation, stamp, pixels.shape[0], pixels.shape[1],
                                  self.hold_epoch, mode == 'policy', self.opening]
                self.last_stamp = stamp
            finally:
                self.input_lock.release()
        if self.output_lock.acquire(False):
            try:
                data = bytes(self.result).split(b'\0',1)[0]
            finally:
                self.output_lock.release()
            if data:
                self.last = json.loads(data)
        if self.last is None or self.last['generation'] != self.generation:
            return fallback
        if self.config.get('method') in ('energy', 'local_normal') and self.last.get('hold_epoch') != self.hold_epoch:
            return dict(fallback, reason='waiting_hold_calibration')
        age = now-self.last['capture_time']
        if not 0 <= age <= self.config['max_sample_gap_s']:
            return dict(fallback, reason='stale_result', result_age_ms=age*1000)
        if self.config.get('method') in ('energy', 'local_normal') and (mode != 'policy' or (self.opening and self.config.get('method') == 'energy')):
            return dict(self.last, stuck=False, reason='gripper_opening' if self.opening else 'non_policy', result_age_ms=age*1000)
        return dict(self.last, result_age_ms=age*1000)

    def close(self):
        if self.process is not None:
            self.stop.set()
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            self.process = None
