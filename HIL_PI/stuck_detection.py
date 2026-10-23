"""Passive translation-stall detection. No hardware or action mutation."""
from collections import deque
import math
import numpy as np


class StuckDetector:
    DEFAULTS = {
        "enabled": False,
        "window_mode": "time",
        "confirm_chunks": 1,
        "feedback_samples": 5,
        "feedback_max_excursion_m": 0.0003,
        "min_error_fraction": 0.8,
        "log_interval_s": 2.0,
        "window_s": 0.5,
        "confirm_s": 0.25,
        "min_tracking_error_m": 0.003,
        "release_tracking_error_m": None,
        "max_position_excursion_m": 0.001,
        "command_timeout_s": 1.0,
        "max_sample_gap_s": 0.2,
        "min_samples": 5,
        "min_direction_cosine": 0.7,
    }

    def __init__(self, config=None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("hil.stuck_detection must be a mapping")
        unknown = set(config) - set(self.DEFAULTS)
        if unknown:
            raise ValueError("Unknown stuck_detection options: " + str(sorted(unknown)))
        self.config = dict(self.DEFAULTS, **config)
        if self.config["release_tracking_error_m"] is None:
            self.config["release_tracking_error_m"] = self.config["min_tracking_error_m"]
        self.enabled = self.config["enabled"]
        if type(self.enabled) is not bool:
            raise ValueError("stuck_detection.enabled must be boolean")
        for key in ("release_tracking_error_m", "log_interval_s", "window_s", "confirm_s", "min_tracking_error_m",
                    "max_position_excursion_m", "command_timeout_s", "max_sample_gap_s"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid stuck_detection." + key)
        if self.config["release_tracking_error_m"] > self.config["min_tracking_error_m"]:
            raise ValueError("Release error must not exceed entry error")
        n = self.config["min_samples"]
        if type(n) is not int or n < 2:
            raise ValueError("stuck_detection.min_samples must be integer >= 2")
        cosine = self.config["min_direction_cosine"]
        if isinstance(cosine, bool) or not isinstance(cosine, (int, float)) or not math.isfinite(cosine) or not 0 <= cosine <= 1:
            raise ValueError("min_direction_cosine must be in [0, 1]")
        if self.config["max_position_excursion_m"] >= self.config["min_tracking_error_m"]:
            raise ValueError("Position excursion threshold must be below tracking-error threshold")
        if self.config["window_mode"] == "time" and self.config["command_timeout_s"] < self.config["window_s"] + self.config["confirm_s"]:
            raise ValueError("command_timeout_s must cover window_s + confirm_s")
        if self.config['window_mode'] not in ('time', 'executed_chunk', 'feedback'):
            raise ValueError('Invalid stuck_detection.window_mode')
        if type(self.config['confirm_chunks']) is not int or self.config['confirm_chunks'] < 1:
            raise ValueError('confirm_chunks must be a positive integer')
        fraction = self.config['min_error_fraction']
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError('min_error_fraction must be in (0, 1]')
        if type(self.config['feedback_samples']) is not int or self.config['feedback_samples'] < 2:
            raise ValueError('feedback_samples must be integer >= 2')
        excursion = self.config['feedback_max_excursion_m']
        if isinstance(excursion, bool) or not isinstance(excursion, (int, float)) or not math.isfinite(excursion) or not 0 < excursion < self.config['min_tracking_error_m']:
            raise ValueError('Invalid feedback_max_excursion_m')
        from HIL_PI.feedback_stuck_detection import FeedbackStuckDetector
        self.feedback_detector = FeedbackStuckDetector(self.config)
        from HIL_PI.chunk_stuck_detection import ChunkStuckDetector
        self.chunk_detector = ChunkStuckDetector(self.config)
        self.reset()

    def reset(self):
        self.chunk_detector.reset()
        self.feedback_detector.reset()
        self.samples = deque()
        self.direction = None
        self.suspect_since = None
        self.last_time = None
        self.last_fresh_time = None
        self.previous_target = None
        self.stuck = False

    def _clear_window(self):
        self.samples.clear()
        self.direction = None
        self.suspect_since = None
        self.stuck = False

    def update(self, timestamp, measured_xyz, published_xyz, *, mode="policy", fresh_command=True,
               source=None, generation=None, chunk_end=False):
        """Compare current measured position against the PREVIOUS published target.

        Inputs use the same robot base frame, meters, and a monotonic timestamp.
        fresh_command means a new policy sample was actually dispatched this tick.
        A held target remains eligible only within command_timeout_s.
        """
        if not self.enabled:
            return None
        if self.config['window_mode'] == 'feedback':
            result = self.feedback_detector.update(timestamp, measured_xyz, published_xyz,
                mode=mode, fresh_command=fresh_command, source=source, generation=generation, chunk_end=chunk_end)
            self.stuck = result['stuck']
            return result
        if self.config['window_mode'] == 'executed_chunk':
            result = self.chunk_detector.update(timestamp, measured_xyz, published_xyz,
                mode=mode, fresh_command=fresh_command, source=source, generation=generation, chunk_end=chunk_end)
            self.stuck = result['stuck']
            return result
        was_stuck = self.stuck

        def report(reason, error=None, excursion=None, span=0.0):
            event = "entered" if self.stuck and not was_stuck else "cleared" if was_stuck and not self.stuck else None
            return dict(stuck=self.stuck, event=event, reason=reason,
                        tracking_error_m=error, position_excursion_m=excursion,
                        window_span_s=span, samples=len(self.samples),
                        suspect_duration_s=(max(0.0, timestamp-self.suspect_since)
                                            if self.suspect_since is not None else 0.0),
                        detector="translation_stall_v2", monitor_only=True)

        try:
            timestamp = float(timestamp)
            measured = np.asarray(measured_xyz, dtype=float)
            target = np.asarray(published_xyz, dtype=float)
            valid = math.isfinite(timestamp) and measured.shape == (3,) and target.shape == (3,)
            valid = valid and np.isfinite(measured).all() and np.isfinite(target).all()
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self.reset()
            return report("invalid_sample")
        if mode != "policy":
            self.reset()
            return report("non_policy")
        if self.last_time is not None and (timestamp <= self.last_time or
                timestamp-self.last_time > self.config["max_sample_gap_s"]):
            self.reset()
            self.last_time = timestamp
            self.previous_target = target.copy()
            self.last_fresh_time = timestamp if fresh_command else None
            return report("sample_gap")
        previous = self.previous_target
        self.previous_target = target.copy()
        self.last_time = timestamp
        if fresh_command:
            self.last_fresh_time = timestamp
        if self.last_fresh_time is None or timestamp-self.last_fresh_time > self.config["command_timeout_s"]:
            self._clear_window()
            return report("stale_command")
        if previous is None:
            return report("warming_up")
        delta = previous-measured
        error = float(np.linalg.norm(delta))
        # Hysteresis only after confirmed stall; keep legacy behavior when omitted.
        threshold = self.config["release_tracking_error_m" if self.stuck else "min_tracking_error_m"]
        if error < threshold:
            self._clear_window()
            return report("below_error_threshold", error)
        direction = delta/error
        # Reset on an actual change in intended movement, not just noisy measured drift.
        current_delta = target-measured
        norm = float(np.linalg.norm(current_delta))
        if norm < threshold:
            self._clear_window()
            return report("below_error_threshold", error)
        if float(np.dot(current_delta/norm, direction)) < self.config["min_direction_cosine"]:
            self._clear_window()
            return report("direction_changed", error)
        if self.direction is not None and float(np.dot(direction, self.direction)) < self.config["min_direction_cosine"]:
            self._clear_window()
        if self.direction is None:
            self.direction = direction.copy()
        self.samples.append((timestamp, measured.copy()))
        cutoff = timestamp-self.config["window_s"]
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        span = timestamp-self.samples[0][0]
        if span + 1e-9 < self.config["window_s"] or len(self.samples) < self.config["min_samples"]:
            return report("warming_up", error, span=span)
        # Full-window excursion prevents oscillation/return-to-start being called a stall.
        positions = np.stack([p for _, p in self.samples])
        excursion = float(np.linalg.norm(np.ptp(positions, axis=0)))
        if excursion > self.config["max_position_excursion_m"]:
            self.suspect_since = None
            self.stuck = False
            return report("moving", error, excursion, span)
        if self.suspect_since is None:
            self.suspect_since = timestamp
        self.stuck = timestamp-self.suspect_since + 1e-9 >= self.config["confirm_s"]
        return report("stuck" if self.stuck else "suspected", error, excursion, span)
