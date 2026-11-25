"""Thin settled SpaceMouse holds in the recording writer, never in robot control."""
from __future__ import annotations

from math import cos, isfinite, radians

import numpy as np


def enabled(hil):
    value = hil.get('omit_settled_human_frames', False)
    if type(value) is not bool:
        raise ValueError('hil.omit_settled_human_frames must be boolean')
    if value and (hil.get('recording', {}).get('format') != 'collector_h265'
                  or hil.get('server_capture') or hil.get('expo_online')
                  or hil.get('keyboard_hil', {}).get('enabled')):
        raise ValueError('Idle filtering requires local SpaceMouse collector_h265 recording')
    return value


class SettledHumanFrames:
    """Keep boundaries, require live measured feedback and fixed published targets."""

    def __init__(self, writer, fps):
        self.writer = writer
        self.fps = fps
        self.segment = None
        self.armed = False
        self.anchor = None
        self.last_state_time = None
        self.seen = 0
        self.gaps = []
        self.pending = None
        self.after = []

    def _stationary(self, row):
        seg = row.get('segment')
        if row.get('mode') != 'human' or seg != self.segment:
            self.segment = seg
            self.anchor = None
            self.armed = False
            self.last_state_time = None
        if row.get('mode') != 'human':
            return False
        try:
            axes = np.asarray(row['spacemouse_axes'], dtype=float)
            buttons = row['spacemouse_buttons']
            if axes.shape != (6,) or not np.isfinite(axes).all() or type(buttons) is not int:
                raise ValueError('missing operator feedback')
            if np.any(axes) or buttons:
                self.armed = True
                self.anchor = None
                return False
            if not self.armed:
                return False
            measured = np.asarray(row['state'][:10], dtype=float)
            command = np.asarray(row['published_action'][:10], dtype=float)
            joints = np.asarray(row['robot_state']['joint_positions'], dtype=float)
            ts = float(row['state_time'])
            age = float(row['publish_start']) - ts
            if (measured.shape != (10,) or command.shape != (10,) or joints.shape != (6,)
                    or not all(np.isfinite(v).all() for v in (measured, command, joints))
                    or not isfinite(ts) or not 0 <= age <= .1):
                raise ValueError('missing or stale measurement')
        except (TypeError, KeyError, ValueError, OverflowError):
            self.anchor = None
            return False
        prev = self.last_state_time
        self.last_state_time = ts
        if prev is None or not 0 < ts-prev <= 2/self.fps:
            self.anchor = None
        if self.anchor is not None:
            s, a, j, since, first_tick = self.anchor
            # Fixed anchor, not consecutive differences: slow drift resets the dwell.
            stable = (np.linalg.norm(measured[:3]-s[:3]) <= .0005
                      and abs(measured[9]-s[9]) <= .0005
                      and np.max(abs(joints-j)) <= radians(.15)
                      and all(np.dot(measured[k:k+3], s[k:k+3]) >= cos(radians(.15))
                              for k in (3, 6))
                      and np.max(abs(command-a)) <= 1e-6)
            if stable:
                return ts-since >= .30 and row['tick']-first_tick >= self.fps*.30
        self.anchor = (measured, command, joints, ts, row['tick'])
        return False

    def _keep(self, row):
        self.writer.add(dict(row, frame_index=self.writer.tick_count))

    def _resolve(self, drop):
        if self.pending is None:
            return
        row = self.pending
        if drop:
            tick = row['tick']
            if self.gaps and self.gaps[-1]['last_tick'] == tick-1:
                self.gaps[-1]['last_tick'] = tick
                self.gaps[-1]['last_time'] = row['time']
            else:
                self.gaps.append(dict(first_tick=tick, last_tick=tick, first_time=row['time'],
                                      last_time=row['time'], segment=row['segment']))
        else:
            self._keep(row)
        for event in self.after:
            self.writer.add(event)
        self.pending = None
        self.after.clear()

    def add(self, row):
        if row['kind'] == 'tick':
            if row['tick'] != self.seen:
                raise ValueError('Non-contiguous source control ticks before idle filtering')
            self.seen += 1
            idle = self._stationary(row)
            self._resolve(drop=idle)
            if idle:
                self.pending = row  # one-frame lookahead retains the last hold before resume/end
            else:
                self._keep(row)
        elif row['kind'] == 'episode_end':
            self._resolve(drop=False)
            self.writer.add(row)
        elif self.pending is None:
            self.writer.add(row)
        else:
            self.after.append(row)

    def finish(self):
        self._resolve(drop=False)
        removed = sum(r['last_tick']-r['first_tick']+1 for r in self.gaps)
        if self.seen != self.writer.tick_count + removed:
            raise ValueError('Unaccounted control frames')
        return dict(source_tick_count=self.seen, omitted_human_idle_frames=removed,
                    omitted_human_idle_ranges=self.gaps)
