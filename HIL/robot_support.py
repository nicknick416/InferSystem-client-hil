"""Shared hardware contracts for both HIL runtimes; no device connections here."""
from __future__ import annotations

import importlib.util
import math


def gripper_action_scale(cfg):
    """Convert measured meters to model action units, including human/hold actions."""
    inf = cfg.inference
    if inf.gripper_mode.value == 'normalized_width':
        width = float(cfg.robot.gripper.max_width)
        if not math.isfinite(width) or width <= 0:
            raise ValueError('normalized_width requires positive finite gripper.max_width')
        if inf.gripper_action_is_10x_meters:
            raise ValueError('normalized_width cannot also use gripper_action_is_10x_meters')
        return 1. / width
    return 10. if inf.gripper_action_is_10x_meters else 1.


def validate_robot(cfg, raw):
    inf = cfg.inference
    if cfg.robot.type not in ('flexiv', 'ur') or inf.action_space.value != 'cartesian':
        raise ValueError('HIL supports single-arm Flexiv/UR Cartesian only')
    if inf.gripper_mode.value not in ('width', 'normalized_width') or inf.hold_gripper_closed_during_inference:
        raise ValueError('HIL requires width or normalized_width without forced gripper hold')
    gripper_action_scale(cfg)
    if cfg.robot.type != 'ur':
        return
    if cfg.robot.dof != 6 or not cfg.robot.gripper.enabled:
        raise ValueError('UR HIL requires six joints and an enabled Robotiq gripper')
    if not math.isfinite(inf.fps) or not math.isclose(cfg.robot.control.frequency_hz, inf.fps):
        raise ValueError('UR control.frequency_hz must equal inference.fps')
    if cfg.robot.control.wait_step_reached:
        raise ValueError('UR HIL requires control.wait_step_reached=false')
    home = inf.inference_home
    if home is None:
        home = cfg.robot.control.home_position_deg
    if home is None or len(home) != 6 or not all(math.isfinite(v) for v in home):
        raise ValueError('UR HIL requires a finite six-joint home position')
    if inf.inference_home_gripper_width is not None and not (
        0 <= inf.inference_home_gripper_width <= cfg.robot.gripper.max_width
    ):
        raise ValueError('UR inference_home_gripper_width must be meters within gripper range')
    if inf.gripper_mode.value == 'normalized_width' and raw.get('inference', {}).get('gripper_state_is_10x_meters', False):
        raise ValueError('UR normalized_width contract requires gripper state in meters')


def live_dependencies(cfg):
    modules = ['hid', 'numpy', 'zmq', 'cv2']
    modules += ['rtde_control', 'rtde_receive'] if cfg.robot.type == 'ur' else ['flexivrdk']
    enabled = cfg.inference.enabled_cameras
    if any(camera.type == 'realsense' for name, camera in cfg.cameras.items()
           if enabled is None or name in enabled):
        modules.append('pyrealsense2')
    return {name: importlib.util.find_spec(name) is not None for name in modules}
