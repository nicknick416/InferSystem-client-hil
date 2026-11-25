"""Offline episode loading and UR Cartesian replay. Importing never opens hardware."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import msgpack
import numpy as np
import yaml


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return a


def pose7(value, name):
    a = vector(value, 7, name).copy()
    norm = np.linalg.norm(a[3:])
    if abs(norm - 1) > 0.01:
        raise ValueError(f"{name}: quaternion must have unit length")
    a[3:] /= norm
    return a


def pose_error(target, actual):
    actual = pose7(actual, 'actual eef_pose')
    distance = float(np.linalg.norm(target[:3] - actual[:3]))
    angle = 2 * math.acos(float(np.clip(abs(np.dot(target[3:], actual[3:])), 0, 1)))
    return distance, angle


@dataclass
class Trajectory:
    episode: Path
    source: str
    times: np.ndarray
    ticks: np.ndarray
    poses: np.ndarray
    widths: np.ndarray
    initial_joints: np.ndarray
    recorded_config: dict
    fps: float

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, timestamp):
        """Linear xyz, shortest-path quaternion SLERP, held gripper commands."""
        t = float(np.clip(timestamp, 0, self.duration))
        i = min(int(np.searchsorted(self.times, t, side='right')) - 1, len(self.times) - 1)
        if i == len(self.times) - 1:
            return self.poses[i].copy(), float(self.widths[i]), int(self.ticks[i])
        u = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        a, b = self.poses[i], self.poses[i + 1]
        qa, qb = a[3:], b[3:].copy()
        dot = float(np.dot(qa, qb))
        if dot < 0:
            qb, dot = -qb, -dot
        if dot > 0.9995:
            q = qa + u * (qb - qa)
        else:
            angle = math.acos(float(np.clip(dot, 0, 1)))
            q = (math.sin((1-u)*angle)*qa + math.sin(u*angle)*qb) / math.sin(angle)
        q /= np.linalg.norm(q)
        return np.r_[a[:3] + u*(b[:3]-a[:3]), q], float(self.widths[i]), int(self.ticks[i])


def load_episode(path, *, source='command', start=0, end=None, max_gap_s=0.25):
    path = Path(path).expanduser().resolve()
    if path.name == 'control.msgpack':
        path = path.parent
    if source not in ('command', 'measured'):
        raise ValueError('source must be command or measured')
    max_gap_s = positive(max_gap_s, 'max_gap_s')
    from HIL.recording import read_metadata, compact_records
    if path.name == 'ticks.csv' and path.parent.name == 'hil':
        path = path.parent.parent
    manifest = read_metadata(path)
    if path.name.startswith('.pending_') or manifest.get('status') != 'kept':
        raise ValueError('Replay requires a completed, kept episode')
    config = manifest.get('config', {})
    if config.get('robot', {}).get('type') != 'ur':
        raise ValueError('Episode was not recorded with a UR robot')
    fps = positive(manifest.get('fps'), 'recorded fps')
    rows = [r for r in compact_records(path) if r.get('kind') == 'tick']
    end = len(rows) if end is None else end
    if not 0 <= start < end <= len(rows):
        raise ValueError(f'Invalid frame range [{start}, {end}); episode has {len(rows)} ticks')
    rows = rows[start:end]
    times, ticks, poses, widths = [], [], [], []
    for r in rows:
        tick = r['tick']
        if type(tick) is not int or tick < 0:
            raise ValueError('tick must be a nonnegative integer')
        ticks.append(tick)
        times.append(float(r['publish_start'] if source == 'command' else r['state_time']))
        poses.append(pose7(r['published_pose'] if source == 'command'
                           else r['robot_state']['eef_pose'], f'tick {tick} pose'))
        widths.append(float(r['commanded_gripper_m'] if source == 'command' else r['state'][9]))
        vector(r['robot_state']['joint_positions'], 6, f'tick {tick} joints')
    times, poses, widths = np.asarray(times), np.asarray(poses), np.asarray(widths)
    max_width = positive(config['robot']['gripper']['max_width'], 'recorded gripper max_width')
    if not np.isfinite(times).all() or not np.isfinite(widths).all():
        raise ValueError('Non-finite timestamp or gripper width')
    if np.any(widths < 0) or np.any(widths > max_width + 1e-8):
        raise ValueError('Gripper width outside recorded limits (expected meters)')
    if np.any(np.diff(ticks) != 1):
        raise ValueError('Missing or out-of-order control ticks')
    gaps = np.diff(times)
    if np.any(gaps <= 0) or np.any(gaps > max_gap_s):
        raise ValueError(f'Invalid time sequence or recording gap > {max_gap_s}s')
    return Trajectory(path, source, times-times[0], np.asarray(ticks), poses, widths,
                      vector(rows[0]['robot_state']['joint_positions'], 6, 'initial joints'), config, fps)


def validate_config(config_path, trajectory):
    """Read only: no robot factory, SDK, camera or inference connection."""
    config = yaml.safe_load(Path(config_path).read_text())
    robot = config.get('robot', {})
    if robot.get('type') != 'ur' or robot.get('dof', 6) != 6:
        raise ValueError('Replay requires a six-joint UR configuration')
    control = robot.get('control', {})
    hz = positive(control.get('frequency_hz', control.get('hz', 30)), 'control.frequency_hz')
    grip = robot.get('gripper', {})
    recorded = trajectory.recorded_config['robot'].get('gripper', {})
    if not grip.get('enabled'):
        raise ValueError('Replay requires the recorded gripper to be enabled')
    for key, default in [('max_width', 0.085), ('min_width', 0.), ('open_raw', 0), ('close_raw', 255)]:
        if not math.isclose(float(grip.get(key, default)), float(recorded.get(key, default)), abs_tol=1e-8):
            raise ValueError(f'Gripper {key} differs from the recorded configuration')
    if grip.get('auto_calibrate_on_connect', grip.get('auto_calibrate', False)):
        raise ValueError('Disable automatic gripper calibration before replay')
    minimum, maximum = float(grip.get('min_width', 0)), float(grip.get('max_width', .085))
    if np.any(trajectory.widths < minimum-1e-8) or np.any(trajectory.widths > maximum+1e-8):
        raise ValueError('Trajectory exceeds configured gripper limits')
    return hz


def describe(trajectory, hz, speed):
    speed = positive(speed, 'speed')
    return dict(episode=str(trajectory.episode), source=trajectory.source,
                frames=len(trajectory.times), first_tick=int(trajectory.ticks[0]),
                last_tick=int(trajectory.ticks[-1]), recorded_fps=trajectory.fps,
                duration_s=trajectory.duration, replay_duration_s=trajectory.duration/speed,
                control_hz=hz, speed=speed, initial_joints_rad=trajectory.initial_joints.tolist(),
                first_pose=trajectory.poses[0].tolist(),
                gripper_range_m=[float(trajectory.widths.min()), float(trajectory.widths.max())])


def replay(config_path, trajectory, *, speed=1., live=False, log_dir=None,
           move_to_start=True, start_velocity=0.15, max_position_error=.05,
           max_rotation_error=.35, max_lag_s=.25, settle_timeout=3.,
           robot_factory=None, clock=time):
    speed = positive(speed, 'speed')
    start_velocity = positive(start_velocity, 'start_velocity')
    max_position_error = positive(max_position_error, 'max_position_error')
    max_rotation_error = positive(max_rotation_error, 'max_rotation_error')
    max_lag_s = positive(max_lag_s, 'max_lag_s')
    settle_timeout = positive(settle_timeout, 'settle_timeout')
    hz = validate_config(config_path, trajectory)
    summary = describe(trajectory, hz, speed)
    # Reject discontinuities before any hardware is created, including near-antipodal quaternions.
    for a, b in zip(trajectory.poses[:-1], trajectory.poses[1:]):
        distance, angle = pose_error(a, b)
        if distance > max_position_error or angle > max_rotation_error:
            raise ValueError('Recorded trajectory contains a pose jump exceeding replay limits')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not live:
        print('[DRY RUN] 数据检查通过；未连接机械臂、夹爪、相机或推理服务。')
        return summary

    from Core import Action, ActionSpace
    if robot_factory is None:
        from Robot import BaseRobot
        robot_factory = BaseRobot.from_config
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    out = Path(log_dir) if log_dir else trajectory.episode / 'replay_tracking' / stamp
    out.mkdir(parents=True, exist_ok=False)
    summary.update(status='starting', sent_frames=0, log_dir=str(out), move_to_start=move_to_start,
                   max_position_error_m=max_position_error, max_rotation_error_rad=max_rotation_error)
    robot = gripper = None
    peak_position = peak_rotation = 0.
    try:
        with (out / 'tracking.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample', 'phase', 'elapsed_s', 'source_time_s', 'source_tick',
                             'position_error_m_before_send', 'rotation_error_rad_before_send',
                             'target_gripper_m', 'actual_gripper_m_before_send'] +
                            ['target_'+x for x in ('x','y','z','qw','qx','qy','qz')] +
                            ['actual_'+x for x in ('x','y','z','qw','qx','qy','qz')])
            robot = robot_factory(config_path)
            if robot.robot_type != 'ur':
                raise ValueError('Robot factory returned a non-UR robot')
            if not math.isclose(robot.get_params().control_frequency_hz, hz):
                raise ValueError('UR driver frequency differs from replay scheduler frequency')
            robot.connect()
            robot.enable()
            if not robot.is_operational() or robot.dof != 6:
                raise RuntimeError('UR is not operational or does not have six joints')
            # Require real servoL, never the driver\'s asynchronous moveL fallback.
            if not callable(getattr(robot._rtde_c, 'servoL', None)):
                raise RuntimeError('UR replay requires servoL support')
            if move_to_start:
                print('[LIVE] MoveJ 到所选片段首帧的实测关节位置。')
                if not robot.move_joint_position(trajectory.initial_joints.tolist(), velocity=start_velocity):
                    raise RuntimeError('Failed to reach the recorded starting joints')
            else:
                q = vector(robot.observe().joint_positions, 6, 'current joints')
                if np.max(np.abs(q-trajectory.initial_joints)) > .05:
                    raise RuntimeError('--no-move-to-start requires joints within 0.05 rad of the start')
            if not robot._rtde_c.isProgramRunning():
                raise RuntimeError('UR RTDE control script is not running')
            initial = robot.observe()
            error = pose_error(trajectory.poses[0], initial.eef_pose)
            if error[0] > max_position_error or error[1] > max_rotation_error:
                raise RuntimeError('Starting TCP differs from first target; check TCP setup and start pose')
            gripper = robot.create_gripper()
            if gripper is None:
                raise RuntimeError('Recorded gripper is unavailable')
            gripper.connect()
            origin = clock.monotonic()
            period = 1 / hz
            duration = trajectory.duration / speed
            trajectory_steps = math.ceil(duration / period)
            steps = trajectory_steps + math.ceil(settle_timeout / period) + 1
            final_sent = False
            settled_samples = 0
            print('[LIVE] 开始回放，Ctrl+C 停止；结束后保持夹爪，不自动回 Home。')
            for n in range(steps):
                deadline = origin + n*period
                clock.sleep(max(0., deadline-clock.monotonic()))
                now = clock.monotonic()
                if now-deadline > max_lag_s:
                    raise RuntimeError('Replay control loop missed its deadline')
                if not robot.is_operational() or not robot._rtde_c.isProgramRunning():
                    raise RuntimeError('UR fault, disconnection or stopped control script')
                source_time = min(n*period*speed, trajectory.duration)
                target, width, tick = trajectory.sample(source_time)
                actual = robot.observe()
                actual_width = float(gripper.observe().width)
                distance, angle = pose_error(target, actual.eef_pose)
                peak_position, peak_rotation = max(peak_position, distance), max(peak_rotation, angle)
                phase = 'settling' if final_sent else 'replay'
                writer.writerow([n, phase, now-origin, source_time, tick, distance, angle, width,
                                 actual_width, *target.tolist(), *actual.eef_pose])
                if n % max(1, round(hz)) == 0:
                    f.flush()
                    print(f'[{n}] t={source_time:.2f}/{trajectory.duration:.2f}s '
                          f'error={distance*1000:.2f}mm/{math.degrees(angle):.2f}deg', flush=True)
                if distance > max_position_error or angle > max_rotation_error:
                    raise RuntimeError(f'Tracking error exceeded limit: {distance:.4f}m, {angle:.4f}rad')
                # Check again after observation I/O, before sending any stale target.
                if clock.monotonic()-deadline > max_lag_s:
                    raise RuntimeError('Observation I/O exceeded replay deadline')
                # Validate acknowledgement; URRobotiqGripper.move() discards its bool result.
                raw = gripper._width_to_raw(width)
                if gripper.move_raw(raw) is False:
                    raise RuntimeError('Robotiq rejected replay command')
                if clock.monotonic()-deadline > max_lag_s:
                    raise RuntimeError('Gripper I/O exceeded replay deadline')
                robot.act(Action(ActionSpace.CARTESIAN, target.tolist()), state=actual)
                summary['sent_frames'] += 1
                settled_samples = (settled_samples + 1 if final_sent
                    and distance <= min(.005, max_position_error)
                    and angle <= min(.05, max_rotation_error) else 0)
                if settled_samples >= 3:
                    summary.update(status='completed', final_position_error_m=distance,
                                   final_rotation_error_rad=angle)
                    break
                if source_time >= trajectory.duration:
                    final_sent = True
            else:
                raise RuntimeError('Final pose did not settle within timeout')
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Every exit stops the arm; failures are not cleared or retried automatically.
        cleanup_errors = []
        for obj, method in ((robot, 'stop'), (gripper, 'disconnect'), (robot, 'disconnect')):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception as exc:
                    cleanup_errors.append(f'{method}: {exc}')
        summary.update(peak_position_error_m=peak_position, peak_rotation_error_rad=peak_rotation,
                       cleanup_errors=cleanup_errors)
        if cleanup_errors and summary['status'] == 'completed':
            summary['status'] = 'cleanup_failed'
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
        print(f'回放结果：{summary["status"]}；日志：{out}')
    if summary['status'] != 'completed':
        raise RuntimeError('Replay cleanup failed; inspect summary.json')
    return summary
