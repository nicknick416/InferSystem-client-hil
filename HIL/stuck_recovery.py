"""Recovery to a measured pre-stall pose, with no hardware or networking I/O."""
from collections import deque
import math
import numpy as np
from Inference.action_processing import rotation6d_to_quaternion
from Inference.dispatch import _clamp_quaternion_step, quaternion_to_rotation6d


class StuckRecovery:
    DEFAULTS = dict(enabled=False, motion_mode='limited', arrival_confirm_samples=1, target_mode='motion_window', history_s=5.0, checkpoint_window_s=0.5, min_motion_m=0.001, max_distance_m=0.02,
                    max_rotation_rad=0.15, linear_speed_m_s=0.005,
                    angular_speed_rad_s=0.1, max_tracking_error_m=0.002,
                    max_tracking_rotation_rad=0.03, position_tolerance_m=0.0005,
                    rotation_tolerance_rad=0.01, timeout_s=8.0, max_sample_gap_s=0.2)

    def __init__(self, config=None, fps=30.):
        config = {} if config is None else config
        if not isinstance(config, dict) or set(config)-set(self.DEFAULTS):
            raise ValueError('Invalid hil.stuck_recovery options')
        self.config = dict(self.DEFAULTS, **config)
        self.enabled = self.config['enabled']
        if type(self.enabled) is not bool:
            raise ValueError('stuck_recovery.enabled must be boolean')
        if self.config['target_mode'] not in ('motion_window', 'before_last_moving_chunk'):
            raise ValueError('Invalid stuck_recovery.target_mode')
        if self.config['motion_mode'] not in ('limited', 'inference'):
            raise ValueError('Invalid stuck_recovery.motion_mode')
        if type(self.config['arrival_confirm_samples']) is not int or self.config['arrival_confirm_samples'] < 1:
            raise ValueError('arrival_confirm_samples must be positive integer')
        for key, value in self.config.items():
            if key not in ('enabled', 'target_mode', 'motion_mode') and (isinstance(value, bool) or not isinstance(value, (int, float))
                                    or not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid stuck_recovery.'+key)
        if not (self.config['position_tolerance_m'] < self.config['max_tracking_error_m'] <= self.config['max_distance_m']
                and self.config['rotation_tolerance_rad'] < self.config['max_tracking_rotation_rad'] <= self.config['max_rotation_rad']):
            raise ValueError('Invalid recovery distance/rotation bounds')
        if self.config['history_s'] <= self.config['checkpoint_window_s']:
            raise ValueError('Recovery history must cover checkpoint window')
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Recovery fps must be positive')
        self.reset()

    def reset(self):
        self.history = deque()
        self.chunks = deque(maxlen=2)
        self.checkpoint_source = None
        self.motion_source = None
        self.checkpoint = None
        self.status = 'idle'
        self.reason = None
        self.command = self.target = None
        self.started = self.last_time = None
        self.target_time = None
        self.arrival_samples = 0

    def observe(self, timestamp, state, detection, mode, *, source=None, generation=None):
        if not self.enabled or self.status != 'idle':
            return
        if mode != 'policy':
            self.history.clear()
            self.chunks.clear()
            self.checkpoint_source = self.motion_source = None
            self.checkpoint = None
            return
        if self.history and not 0 < timestamp-self.history[-1][0] <= self.config['max_sample_gap_s']:
            self.history.clear()
            self.chunks.clear()
            self.checkpoint_source = self.motion_source = None
            self.checkpoint = None
        self.history.append((timestamp, np.array(state, dtype=float)))
        while self.history and timestamp-self.history[0][0] > self.config['history_s']:
            self.history.popleft()
        if self.config['target_mode'] == 'before_last_moving_chunk':
            # Observation precedes dispatch: on a boundary it closes the OLD chunk.
            self._update_moving_chunk(state, detection)
            if source is not None and source.get('request_id') is not None:
                identity = (generation, source['request_id'])
                if not self.chunks or self.chunks[-1]['identity'] != identity:
                    self.chunks.append(dict(identity=identity, time=timestamp,
                        state=np.array(state, dtype=float), source=dict(generation=generation,
                        request_id=source['request_id'], first_executed_chunk_index=source.get('chunk_index'))))
            return
        # Keep the start of the latest window with real displacement, including
        # well-tracked motion that the stall detector calls below_error_threshold.
        cutoff = timestamp-self.config['checkpoint_window_s']
        eligible = [entry for entry in self.history if entry[0] <= cutoff+1e-8]
        if not detection['stuck'] and eligible:
            stamp, pose = eligible[-1]
            if np.linalg.norm(state[:3]-pose[:3]) >= self.config['min_motion_m']:
                self.checkpoint = (stamp, pose.copy())

    def _update_moving_chunk(self, state, detection):
        if not self.chunks or detection['stuck']:
            return
        current = self.chunks[-1]
        if np.linalg.norm(state[:3]-current['state'][:3]) < self.config['min_motion_m']:
            return
        self.motion_source = dict(current['source'])
        # Moving chunk N selects measured start of N-1, not the trigger chunk.
        if len(self.chunks) >= 2:
            previous = self.chunks[-2]
            self.checkpoint = (previous['time'], previous['state'].copy())
            self.checkpoint_source = dict(previous['source'])
        else:
            self.checkpoint = self.checkpoint_source = None

    @staticmethod
    def angle(a, b):
        qa, qb = rotation6d_to_quaternion(a[3:9]), rotation6d_to_quaternion(b[3:9])
        return 2*math.acos(float(np.clip(abs(np.dot(qa, qb)), 0, 1)))

    def start(self, timestamp, state, published):
        self.started = self.last_time = timestamp
        self.command = np.array(state, dtype=float)
        self.command[9] = published[9]  # Keep current gripper command, never replay historical opening.
        self.status, self.reason = 'holding', 'no_valid_history'
        if self.checkpoint is not None:
            stamp, pose = self.checkpoint
            self.target = pose.copy()
            self.target[9] = self.command[9]
            self.target_time = stamp
            distance = float(np.linalg.norm(self.target[:3]-self.command[:3]))
            if timestamp-stamp > self.config['history_s']:
                self.reason = 'history_expired'
            elif distance > self.config['max_distance_m']:
                self.reason = 'target_too_far'
            elif self.angle(self.target, self.command) > self.config['max_rotation_rad']:
                self.reason = 'target_rotation_too_large'
            elif self.at_target(state):
                self.reason = 'already_at_target'
            else:
                self.status, self.reason = 'retreating', 'stuck_triggered'
        return self.report()

    def at_target(self, state):
        return (np.linalg.norm(self.target[:3]-state[:3]) <= self.config['position_tolerance_m']
                and self.angle(self.target, state) <= self.config['rotation_tolerance_rad'])

    def report(self):
        return dict(status=self.status, reason=self.reason, target_time=self.target_time,
                    target_mode=self.config['target_mode'], motion_mode=self.config['motion_mode'],
                    target_source=self.checkpoint_source,
                    last_moving_chunk=self.motion_source, arrival_samples=self.arrival_samples,
                    target_state=None if self.target is None else self.target.tolist())

    def step(self, timestamp, state):
        if self.status == 'idle':
            raise RuntimeError('Recovery has not started')
        if self.status == 'retreating':
            dt = timestamp-self.last_time
            failure = ('sample_gap' if not 0 < dt <= self.config['max_sample_gap_s'] else
                       'timeout' if timestamp-self.started >= self.config['timeout_s'] else None)
            self.arrival_samples = self.arrival_samples + 1 if self.at_target(state) else 0
            if failure or self.arrival_samples >= self.config['arrival_confirm_samples']:
                self.status, self.reason = 'holding', failure or 'completed'
                if failure or self.config['motion_mode'] == 'limited':
                    self.command[:9] = state[:9]
                else:
                    # Keep the same reference after arrival; measured-pose reseeding
                    # changes the equilibrium in a compliant Cartesian controller.
                    self.command = self.target.copy()
            elif self.config['motion_mode'] == 'inference':
                # Same Cartesian dispatch path as policy; controller plans motion.
                # start() already rejected targets outside recovery distance/angle bounds.
                # No recovery-specific speed or lead clamp in this explicit mode.
                self.command = self.target.copy()
            else:
                dt = min(dt, 1/self.fps)
                proposal = self.command.copy()
                delta = self.target[:3]-self.command[:3]
                step = self.config['linear_speed_m_s']*dt
                proposal[:3] += delta*min(1., step/max(float(np.linalg.norm(delta)), 1e-12))
                q = _clamp_quaternion_step(rotation6d_to_quaternion(self.command[3:9]),
                                          rotation6d_to_quaternion(self.target[3:9]),
                                          self.config['angular_speed_rad_s']*dt)
                proposal[3:9] = quaternion_to_rotation6d(q)
                # Stop advancing the setpoint if the real arm cannot follow it.
                if (np.linalg.norm(proposal[:3]-state[:3]) <= self.config['max_tracking_error_m']
                        and self.angle(proposal, state) <= self.config['max_tracking_rotation_rad']):
                    self.command = proposal
        self.last_time = timestamp
        report = self.report()
        report.update(elapsed_s=timestamp-self.started,
            position_error_m=None if self.target is None else float(np.linalg.norm(self.target[:3]-state[:3])),
            rotation_error_rad=None if self.target is None else self.angle(self.target, state))
        return self.command.tolist(), report
