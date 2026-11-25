"""Pure preview of the existing dispatcher, with all hardware handles replaced."""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace
import numpy as np
from Inference.action_processing import canonicalize_action_values, max_eef_action_delta
from Inference.dispatch import quaternion_to_rotation6d


class _RobotSink:
    def act(self, action, state=None):
        self.action = action


class _GripperSink:
    def __init__(self, max_width, width):
        self.max_width = max_width
        self.width = width

    def observe(self):
        return SimpleNamespace(max_width=self.max_width, width=self.width)

    def move(self, width, wait=False):
        self.width = width


def command32(dispatcher, requested, scale):
    pose = dispatcher.last_robot_action_values
    if pose is None or len(pose) != 7:
        raise RuntimeError('Expected a single published pose7')
    width = dispatcher._last_gripper_width_cmd
    if width is None:
        raise RuntimeError('Missing dispatcher gripper command')
    return [*pose[:3], *quaternion_to_rotation6d(pose[3:7]), width*scale] + [0.]*22


def preview_candidate(dispatcher, requested, arm, config, gripper_max, scale):
    """Use the SAME clamp/gripper hysteresis with a copy of pre-selection state.

    Calling this never writes hardware or advances the live dispatcher's memory.
    An unsafe policy candidate is recorded as rejected and cannot form a valid pair.
    """
    action = canonicalize_action_values(requested, 'cartesian', policy_format='canonical',
                                         canonical_dim=32, effective_action_dim=10)
    pos, rot = max_eef_action_delta(arm.eef_pose, action)
    if not np.isfinite(action).all() or pos > config.max_eef_delta_m or rot > config.max_eef_rotation_delta_rad:
        return dict(valid=False, action=None, pose=None, gripper_m=None,
                    rejection=f'safety: {pos:.6f} m / {rot:.6f} rad')
    preview = copy(dispatcher)
    preview._robot = _RobotSink()
    width = dispatcher._last_gripper_width_cmd
    if width is None and dispatcher._max_eef_gripper_step > 0:
        width = dispatcher._gripper.observe().width
    preview._gripper = _GripperSink(gripper_max, width or 0.)
    preview._gripper_max_width = gripper_max
    preview.dispatch(action, state=arm)
    return dict(valid=True, action=command32(preview, requested, scale),
                pose=list(preview.last_robot_action_values), gripper_m=preview._last_gripper_width_cmd,
                rejection=None)
