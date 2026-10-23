"""HID input only; local Flexiv mapping with yaw reversed after operator feedback."""
from __future__ import annotations

import threading
import time
import numpy as np


class SpaceMouse:
    def __init__(self, deadzone=0.1, vendor_id=0x256F, product_id=None,
                 rotation_roll_enabled=True, rotation_pitch_enabled=True, rotation_yaw_enabled=True):
        self.deadzone, self.vendor_id, self.product_id = deadzone, vendor_id, product_id
        self.rotation_roll_enabled = rotation_roll_enabled
        self.rotation_pitch_enabled = rotation_pitch_enabled
        self.rotation_yaw_enabled = rotation_yaw_enabled
        self.lock = threading.Lock()
        self.axes = np.zeros(6)
        self.buttons = 0
        self.activity = threading.Event()
        self.error = None
        self.device = None
        self.running = False
        self.thread = None

    def feed(self, report):
        def axis(i):
            return np.clip(int.from_bytes(bytes(report[i:i+2]), 'little', signed=True) / 350., -1., 1.)
        with self.lock:
            if report and report[0] == 1 and len(report) >= 13:
                self.axes[:] = [axis(3), axis(1), -axis(5), -axis(9), axis(7), axis(11)]
            elif report and report[0] == 1 and len(report) >= 7:
                self.axes[:3] = [axis(3), axis(1), -axis(5)]
            elif report and report[0] == 2 and len(report) >= 7:
                self.axes[3:] = [-axis(3), axis(1), axis(5)]
            elif report and report[0] == 3 and len(report) >= 2:
                self.buttons = report[1] & 3
            # Mask before the activity latch: disabled rotation cannot trigger takeover.
            if not self.rotation_roll_enabled:
                self.axes[3] = 0.    # tool Rx: roll
            if not self.rotation_pitch_enabled:
                self.axes[4] = 0.    # tool Ry: pitch
            if not self.rotation_yaw_enabled:
                self.axes[5] = 0.    # tool Rz: left/right twist
            if np.any(np.abs(self.axes) > self.deadzone) or self.buttons:
                self.activity.set()  # latch even brief input between control ticks

    def sample(self):
        if self.error:
            raise RuntimeError('SpaceMouse reader failed') from self.error
        with self.lock:
            return np.where(np.abs(self.axes) > self.deadzone, self.axes, 0.).copy(), self.buttons

    def open(self):
        import hid  # delayed: offline tests never enumerate/open HID
        devices = hid.enumerate(self.vendor_id, self.product_id or 0)
        # Composite HID devices can have multiple interfaces; use the motion interface.
        devices = [d for d in devices if d.get('usage_page', 1) == 1 and d.get('usage', 8) == 8] or devices
        if not devices:
            raise RuntimeError('No matching SpaceMouse; check HID permissions/device configuration')
        self.device = hid.device()
        try:
            self.device.open_path(devices[0]['path'])
            self.device.set_nonblocking(True)
        except Exception:
            self.device.close()
            self.device = None
            raise
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while self.running:
                report = self.device.read(13)
                if report:
                    self.feed(report)
                else:
                    time.sleep(.002)
        except Exception as exc:
            self.error = exc
            self.running = False

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(1)
        if self.device:
            self.device.close()
            self.device = None


def human_action(state, axes, buttons, dt, *, linear_speed=.06, angular_speed=.21,
                 gripper_speed=.03, gripper_max=.1, gripper_scale=10.):
    """Absolute 32D target from measured state: no accumulated unreachable target.

    Translation in base frame, rotation in tool frame. Buttons: 1 close, 2 open,
    both hold (never implicitly return to an old pose). State gripper is meters.
    """
    out = np.zeros(32, dtype=np.float64)
    out[:10] = state[:10]
    out[:3] += np.asarray(axes[:3]) * linear_speed * dt
    a, b = np.asarray(state[3:6]), np.asarray(state[6:9])
    a = a / np.linalg.norm(a)
    b = b - a * np.dot(a, b)
    b = b / np.linalg.norm(b)
    rot = np.column_stack([a, b, np.cross(a, b)])
    w = np.asarray(axes[3:]) * angular_speed * dt
    theta = np.linalg.norm(w)
    if theta > 1e-12:
        x, y, z = w / theta
        k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rot = rot @ (np.eye(3) + np.sin(theta)*k + (1.-np.cos(theta))*(k@k))
    out[3:9] = np.concatenate([rot[:, 0], rot[:, 1]])
    direction = -1 if buttons == 1 else 1 if buttons == 2 else 0
    out[9] = np.clip(state[9] + direction * gripper_speed * dt, 0., gripper_max) * gripper_scale
    return out.tolist()


class HumanController:
    """Measured-relative motion while active; fixed targets while released.

    Translation and rotation latch independently, so gripper-only input never
    follows TCP feedback noise. Gripper target integrates commands, not lagging
    feedback, and therefore can cross dispatcher hysteresis/mechanical deadband.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.position_active = False
        self.rotation_active = False

    def step(self, state, axes, buttons, dt, *, linear_speed=.06, angular_speed=.21,
             gripper_speed=.03, gripper_max=.1, gripper_scale=10.):
        if self.target is None:
            self.target = np.asarray(state, dtype=float).copy()
        position_active = bool(np.any(axes[:3]))
        rotation_active = bool(np.any(axes[3:]))
        # Release: latch measured TCP once to cancel a residual ahead-of-robot target.
        if self.position_active and not position_active:
            self.target[:3] = state[:3]
        if self.rotation_active and not rotation_active:
            self.target[3:9] = state[3:9]
        proposed = human_action(state, axes, 0, dt, linear_speed=linear_speed,
                                angular_speed=angular_speed, gripper_max=gripper_max,
                                gripper_scale=1.)
        if position_active:
            self.target[:3] = proposed[:3]
        if rotation_active:
            self.target[3:9] = proposed[3:9]
        direction = -1 if buttons == 1 else 1 if buttons == 2 else 0
        self.target[9] = np.clip(self.target[9]+direction*gripper_speed*dt, 0., gripper_max)
        self.position_active, self.rotation_active = position_active, rotation_active
        out = np.zeros(32)
        out[:10] = self.target[:10]
        out[9] *= gripper_scale
        return out.tolist()
