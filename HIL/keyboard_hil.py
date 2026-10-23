"""Two-key HIL rollback using only executed policy commands and measured feedback."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from HIL.stuck_recovery import StuckRecovery


@dataclass
class ExecutedChunk:
    identity: tuple[int, int]
    source: dict
    start_time: float
    start_state: np.ndarray
    last_state: np.ndarray
    max_width_m: float
    last_width_m: float
    closing_start: tuple[float, np.ndarray] | None = None
    closing_anchor: tuple[float, np.ndarray] | None = None


class KeyboardHIL:
    DEFAULTS = dict(enabled=False, close_threshold_m=0.003, open_tolerance_m=0.002,
                    open_confirm_samples=3, open_timeout_s=3.0)

    def __init__(self, config: dict | None, recovery_config: dict | None, fps: float):
        value = {} if config is None else config
        if not isinstance(value, dict) or set(value) - set(self.DEFAULTS):
            raise ValueError('Invalid hil.keyboard_hil options')
        self.config = dict(self.DEFAULTS, **value)
        if type(self.config['enabled']) is not bool:
            raise ValueError('hil.keyboard_hil.enabled must be boolean')
        for name in ('close_threshold_m', 'open_tolerance_m', 'open_timeout_s'):
            number = self.config[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f'Invalid hil.keyboard_hil.{name}')
        count = self.config['open_confirm_samples']
        if type(count) is not int or count < 1:
            raise ValueError('Invalid hil.keyboard_hil.open_confirm_samples')
        motion = dict(recovery_config or {})
        motion.update(enabled=True, target_mode='before_last_moving_chunk', motion_mode='inference')
        self.motion = StuckRecovery(motion, fps)
        self.reset()

    def reset(self):
        self.chunks: list[ExecutedChunk] = []
        self.phase = 'none'
        self.hil_id: int | None = None
        self.attempts = getattr(self, 'attempts', 0)
        self.branch = None
        self.reason = None
        self.anchor = None
        self.command = None
        self.open_started = None
        self.open_samples = 0
        self.last_feedback_time = None
        self.motion.reset()

    @property
    def active(self) -> bool:
        return self.phase not in ('none', 'human')

    def feedback(self, timestamp: float, state):
        if self.phase != 'none':
            return
        if self.last_feedback_time is not None and not 0 < timestamp - self.last_feedback_time <= self.motion.config['max_sample_gap_s']:
            self.chunks.clear()
        self.last_feedback_time = timestamp
        if self.chunks:
            self.chunks[-1].last_state = np.asarray(state, dtype=float).copy()
            cutoff = timestamp - self.motion.config['history_s']
            self.chunks = [c for c in self.chunks if c.start_time >= cutoff]

    def executed(self, timestamp: float, state, published, source: dict | None, generation: int,
                 previous_width_m: float, gripper_scale: float):
        if self.phase != 'none' or source is None or source.get('request_id') is None:
            return
        identity = (generation, source['request_id'])
        current = np.asarray(state, dtype=float).copy()
        width = float(published[9]) / gripper_scale
        if not self.chunks or self.chunks[-1].identity != identity:
            self.chunks.append(ExecutedChunk(identity, dict(source), timestamp, current, current.copy(),
                                             float(previous_width_m), float(previous_width_m)))
        chunk = self.chunks[-1]
        if width < chunk.last_width_m - 1e-9 and chunk.closing_start is None:
            chunk.closing_start = (timestamp, current.copy())
        if width > chunk.last_width_m + 1e-9:
            chunk.max_width_m = width
            chunk.closing_start = None
            chunk.closing_anchor = None
        if chunk.closing_start is not None and chunk.max_width_m - width >= self.config['close_threshold_m']:
            chunk.closing_anchor = chunk.closing_start
        chunk.last_width_m = width

    def trigger(self, timestamp: float, state, published, *, gripper_scale: float):
        if self.phase != 'none':
            return None
        self.attempts += 1
        self.hil_id = self.attempts
        self.gripper_scale = gripper_scale
        current = np.asarray(state, dtype=float).copy()
        self.command = current.copy()
        self.command[9] = float(published[9])
        self.branch = 'none'
        self.reason = 'no_classifiable_motion'
        self.phase = 'await_human'
        latest = self.chunks[-1] if self.chunks else None
        chunk = None
        selected_index = None
        for index in range(len(self.chunks) - 1, -1, -1):
            item = self.chunks[index]
            if item.closing_anchor is not None or np.linalg.norm(item.last_state[:3] - item.start_state[:3]) >= self.motion.config['min_motion_m']:
                chunk, selected_index = item, index
                break
        if chunk is not None and chunk.closing_anchor is not None:
            self.branch = 'gripper_close'
            self.anchor = chunk.closing_anchor
            self._prepare_anchor(timestamp, current)
            if self.phase != 'await_human':
                # Never issue a new closing target while recovering an open grip.
                self.command[9] = max(float(self.anchor[1][9]), float(current[9])) * gripper_scale
                self.open_started = timestamp
                self.open_samples = 0
                self.phase = 'opening'
                self.reason = None
        elif chunk is not None:
            self.branch = 'position_move'
            if selected_index >= 1:
                prior = self.chunks[selected_index - 1]
                self.anchor = (prior.start_time, prior.start_state.copy())
                self._prepare_anchor(timestamp, current)
                if self.phase != 'await_human':
                    self._start_motion(timestamp, current)
            else:
                self.reason = 'no_valid_history'
        return dict(hil_id=self.hil_id, branch=self.branch, reason=self.reason,
                    phase=self.phase, source=None if chunk is None else chunk.source,
                    latest_source=None if latest is None else latest.source,
                    skipped_chunks=0 if selected_index is None else len(self.chunks) - selected_index - 1,
                    anchor_time=None if self.anchor is None else self.anchor[0],
                    anchor_state=None if self.anchor is None else self.anchor[1].tolist())

    def _prepare_anchor(self, timestamp: float, state: np.ndarray):
        stamp, pose = self.anchor
        if timestamp - stamp > self.motion.config['history_s']:
            self.reason = 'history_expired'
        elif np.linalg.norm(pose[:3] - state[:3]) > self.motion.config['max_distance_m']:
            self.reason = 'target_too_far'
        elif self.motion.angle(pose, state) > self.motion.config['max_rotation_rad']:
            self.reason = 'target_rotation_too_large'
        else:
            self.phase = 'retreating'

    def _start_motion(self, timestamp: float, state: np.ndarray):
        self.motion.reset()
        self.motion.checkpoint = self.anchor
        result = self.motion.start(timestamp, state, self.command)
        self.phase = 'retreating' if result['status'] == 'retreating' else 'await_human'
        self.reason = None if result['status'] == 'retreating' else result['reason']

    def step(self, timestamp: float, state, measured_width_m: float):
        event = None
        if self.phase == 'opening':
            if timestamp - self.open_started >= self.config['open_timeout_s']:
                self.phase, self.reason = 'await_human', 'open_timeout'
                self.command = np.asarray(state, dtype=float).copy()
                self.command[9] = measured_width_m * self.gripper_scale
                event = 'rollback_failed'
            else:
                target_width = self.command[9] / self.gripper_scale
                self.open_samples = self.open_samples + 1 if abs(measured_width_m - target_width) <= self.config['open_tolerance_m'] else 0
                if self.open_samples >= self.config['open_confirm_samples']:
                    self._start_motion(timestamp, np.asarray(state, dtype=float))
                    if self.phase == 'await_human':
                        event = 'rollback_finished' if self.reason == 'already_at_target' else 'rollback_failed'
        if self.phase == 'retreating':
            if timestamp <= self.motion.started:
                return self.motion.command.tolist(), event
            command, report = self.motion.step(timestamp, state)
            self.command = np.asarray(command, dtype=float)
            if report['status'] == 'holding':
                self.phase, self.reason = 'await_human', report['reason']
                event = 'rollback_finished' if report['reason'] in ('completed', 'already_at_target') else 'rollback_failed'
        return self.command.tolist(), event

    def request_human(self):
        if self.phase != 'await_human':
            return False
        self.phase = 'await_neutral'
        return True

    def neutral(self, axes, buttons):
        if self.phase == 'await_neutral' and not np.any(axes) and not buttons:
            self.phase = 'human'
            return True
        return False
