"""Causal tactile-change detector adapted from tactile_monitor_20261012 replay.

Online reference is the first policy-segment frame (not offline frame at 1 s).
Full decoded frames are differenced before INTER_AREA resize and center crop.
Only distinct capture timestamps count; normal robot motion is not inferred here.
"""
from collections import deque
import math
import time
import cv2
import numpy as np
from HIL.stuck_detection import StuckDetector


class TactileStuckDetector:
    DEFAULTS = dict(slip_enabled=False,slip_speed_px_s=8.,slip_min_coherence=.65,slip_min_tracks=6,
                    slip_confirm_frames=3,slip_min_energy=.15,slip_min_contact_ratio=.85,slip_arm_s=.3,normal_data_dir='tactile_normal_data', normal_min_samples=10, normal_min_blocks=2,
                    normal_server='http://127.0.0.1:5558', normal_token_file='Config/correction.token',
                    normal_corrector_data_dir='', method='change', energy_margin=0.15, energy_confirm_frames=3,
                    energy_min_hold_score=1.0, opening_delta_m=0.002, enabled=False, image_key='observation/left_tactile',
                    calibration_start_s=0.5, calibration_end_s=1.5, arm_after_s=2.0,
                    threshold_floor=0.75, increase_threshold=0.5, max_sample_gap_s=0.2)

    def __init__(self, config=None):
        config = config or {}
        config = dict(config)
        monitor = config.pop('monitor', {})
        method = config.get('method')
        if method in ('rule_based', 'monitor'):
            config['method'] = 'local_normal'
        if method == 'monitor':
            from HIL.monitor_client import validate_config
            monitor = validate_config(monitor)
        if not isinstance(config, dict) or set(config) - set(self.DEFAULTS):
            raise ValueError('Invalid hil.tactile_stuck_detection options')
        self.config = dict(self.DEFAULTS, **config)
        if type(self.config['enabled']) is not bool or type(self.config['slip_enabled']) is not bool:
            raise ValueError('tactile_stuck_detection.enabled must be boolean')
        if not isinstance(self.config['image_key'], str) or not self.config['image_key']:
            raise ValueError('Invalid tactile image_key')
        for key, value in self.config.items():
            if key not in ('enabled', 'slip_enabled', 'image_key', 'method', 'normal_data_dir', 'normal_server', 'normal_token_file', 'normal_corrector_data_dir'):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid tactile_stuck_detection.' + key)
        if not self.config['calibration_start_s'] < self.config['calibration_end_s'] <= self.config['arm_after_s']:
            raise ValueError('Invalid tactile calibration/arming times')
        if self.config['method'] not in ('change', 'energy', 'local_normal'):
            raise ValueError('Invalid tactile method')
        if type(self.config['energy_confirm_frames']) is not int:
            raise ValueError('energy_confirm_frames must be integer')
        for key in ('normal_min_samples', 'normal_min_blocks','slip_min_tracks','slip_confirm_frames'):
            if type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError('Invalid ' + key)
        if self.config['normal_min_blocks'] > 40:
            raise ValueError('normal_min_blocks exceeds grid size')
        if self.config['slip_min_coherence']>1 or self.config['slip_min_contact_ratio']>1:raise ValueError('invalid slip ratio')
        if self.config['slip_enabled'] and self.config['method']!='local_normal':raise ValueError('slip requires local_normal')
        self.enabled = self.config['enabled']
        if method == 'monitor':
            self.config.update(method='monitor', monitor=monitor)
        self.reset()

    def reset(self):
        self.reference = None
        self.started = self.last_stamp = None
        self.window = deque(maxlen=10)
        self.calibration_window = deque(maxlen=10)
        self.calibration = []
        self.threshold = None
        self.stuck = False
        self.last_report = dict(stuck=False, reason='reset', score=None, increase=None, threshold=None)

    @staticmethod
    def scores(window):
        x = np.stack(window)
        score = float(np.mean(np.abs(np.median(x[5:], axis=0) - np.median(x[:5], axis=0))))
        q = np.mean(np.abs(x), axis=(1, 2, 3))
        return score, float(np.median(q[5:]) - np.median(q[:5]))

    def update(self, encoded, stamp, mode='policy'):
        def report(reason, score=None, increase=None):
            self.last_report = dict(stuck=self.stuck, reason=reason, score=score,
                                    increase=increase, threshold=self.threshold)
            return dict(self.last_report)
        if not self.enabled or mode != 'policy':
            self.reset()
            return report('disabled' if not self.enabled else 'non_policy')
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not isinstance(encoded, (bytes, np.ndarray)):
            self.reset()
            return report('invalid_input')
        if self.stuck:
            return report('latched')
        if self.last_stamp is not None:
            if stamp == self.last_stamp:
                return dict(self.last_report, reason='duplicate_frame')
            if not 0 < stamp - self.last_stamp <= self.config['max_sample_gap_s']:
                self.reset()
        frame = encoded if isinstance(encoded, np.ndarray) else cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.reset()
            return report('invalid_image')
        if self.reference is not None and frame.shape != self.reference.shape:
            self.reset()
        if self.reference is None:
            self.reference = frame.copy()
            self.started = stamp
        self.last_stamp = stamp
        elapsed = stamp - self.started
        diff = cv2.absdiff(frame, self.reference)
        small = cv2.resize(diff, (80, 45), interpolation=cv2.INTER_AREA)[5:40, 12:68].astype(np.float32)
        self.window.append(small)
        if self.config['calibration_start_s'] <= elapsed <= self.config['calibration_end_s']:
            self.calibration_window.append(small)
            if len(self.calibration_window) == 10:
                self.calibration.append(self.scores(self.calibration_window)[0])
        if elapsed <= self.config['calibration_end_s']:
            return report('calibrating')
        if self.threshold is None:
            if len(self.calibration) < 10:
                self.reset()
                return report('insufficient_calibration_restart')
            med = float(np.median(self.calibration))
            mad = float(np.median(np.abs(np.asarray(self.calibration) - med)))
            self.threshold = max(med + 6 * 1.4826 * mad, 3 * med, self.config['threshold_floor'])
        if elapsed <= self.config['arm_after_s'] or len(self.window) < 10:
            return report('arming')
        score, increase = self.scores(self.window)
        self.stuck = score >= self.threshold and increase >= self.config['increase_threshold']
        return report('tactile_change' if self.stuck else 'below_threshold', score, increase)


class CombinedStuckDetector:
    """OR merger retaining state-action report fields and existing recovery contract."""
    def __init__(self, state_config=None, tactile_config=None):
        self.state = StuckDetector(state_config)
        self.tactile = TactileStuckDetector(tactile_config)
        self.enabled = self.state.enabled or self.tactile.enabled
        self.config = self.state.config
        self.worker = None
        self.raw_frame = self.raw_stamp = None
        self.reset()

    def start(self):
        if self.tactile.enabled and self.worker is None:
            if self.tactile.config['method'] == 'monitor':
                from HIL.monitor_client import MonitorClient
                self.worker = MonitorClient(self.tactile.config)
            else:
                from HIL.tactile_process import TactileProcess
                self.worker = TactileProcess(self.tactile.config)
            self.worker.start()

    def calibrate_hold(self, width):
        if self.worker is not None:
            self.worker.calibrate_hold(width)

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def reset(self):
        if self.worker is not None:
            self.worker.reset()
        self.raw_frame = self.raw_stamp = None
        self.state.reset()
        self.tactile.reset()
        self.stuck = False
        self.payload, self.stamps = {}, {}

    def set_observation(self, payload, stamps, raw_frame=None, raw_stamp=None, opening_width=None):
        self.payload, self.stamps = payload, stamps
        self.raw_frame, self.raw_stamp = raw_frame, raw_stamp
        if self.worker is not None:
            if self.tactile.config['method'] == 'monitor':
                self.worker.set_observation(payload, stamps)
            else:
                self.worker.set_context(payload.get("extra", {}).get("tactile_baseline", {}).get(self.tactile.config["image_key"]), opening_width)

    def update(self, *args, **kwargs):
        state = self.state.update(*args, **kwargs)
        if not self.tactile.enabled:
            return state  # Preserve baseline report exactly when disabled.
        if state is None:
            state = dict(stuck=False, event=None, reason='state_action_disabled',
                         tracking_error_m=None, position_excursion_m=None,
                         window_span_s=0., suspect_duration_s=0.)
        key = self.tactile.config['image_key']
        if self.worker is not None:
            tactile = self.worker.update(self.raw_frame, self.raw_stamp, kwargs.get('mode', 'policy'), time.monotonic())
        else:
            tactile = self.tactile.update(self.payload.get(key), self.stamps.get(key), kwargs.get('mode', 'policy'))
        active = state['stuck'] or tactile['stuck']
        event = 'entered' if active and not self.stuck else 'cleared' if self.stuck and not active else None
        self.stuck = active
        sources = [name for name, fired in [('state_action', state['stuck']), ('tactile', tactile['stuck'])] if fired]
        return dict(state, stuck=active, event=event,
                    reason='or:' + '+'.join(sources) if active else state['reason'],
                    trigger_sources=sources, state_action=state, tactile=tactile)
