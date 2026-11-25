"""No hardware: episode validation, interpolation and failure-path control tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import yaml

from HIL.ur_replay import load_episode, replay


@pytest.fixture
def recording(tmp_path):
    ep = tmp_path / 'episode_000001'
    ep.mkdir()
    config = {'robot': {'type': 'ur', 'dof': 6, 'control': {'frequency_hz': 30},
                        'gripper': {'enabled': True, 'max_width': .085}},
              'inference': {'gripper_mode': 'normalized_width'}}
    manifest = dict(status='kept', fps=30, config=config)
    rows = []
    for i in range(4):
        pose = [.4+i*.001, .1, .3, 1., 0., 0., 0.]
        measured = [.399+i*.001, .1, .3, 1., 0., 0., 0.]
        rows.append(dict(kind='tick', tick=i, publish_start=10+i/30, state_time=9.99+i/30,
                         published_pose=pose, commanded_gripper_m=.02+i*.001,
                         state=[0.]*9+[.015]+[0.]*22,
                         published_action=[999.]*32, policy_raw_action=[-999.]*32,
                         robot_state=dict(joint_positions=[.1]*6, eef_pose=measured)))
    (ep/'manifest.json').write_text(json.dumps(manifest))
    write_rows(ep, rows)
    config_path = tmp_path/'ur.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return ep, config_path, rows


def write_rows(ep, rows):
    (ep/'control.msgpack').write_bytes(b''.join(msgpack.packb(r) for r in rows))


class Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, duration): self.now += duration


class Gripper:
    def __init__(self): self.width = .02; self.connected = False; self.ack = True; self.commands = []
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def observe(self): return SimpleNamespace(width=self.width)
    def _width_to_raw(self, width): return width  # Identity conversion in the fake driver.
    def move_raw(self, raw):
        self.width = raw
        self.commands.append(raw)
        return self.ack


class Robot:
    robot_type = 'ur'
    dof = 6
    def __init__(self, clock):
        self.clock = clock
        self.pose = [.4, .1, .3, 1., 0., 0., 0.]
        self.joints = [.1]*6
        self.calls = []
        self.gripper = Gripper()
        self.connected = False
        self.stopped = False
        self.reached = True
        self.failure = None
        self.follow = True
        self._rtde_c = SimpleNamespace(servoL=lambda: None, isProgramRunning=lambda: True)
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def enable(self): pass
    def is_operational(self): return self.connected
    def get_params(self): return SimpleNamespace(control_frequency_hz=30)
    def stop(self): self.stopped = True
    def create_gripper(self): return self.gripper
    def move_joint_position(self, q, **kwargs): self.joints = q; return self.reached
    def observe(self): return SimpleNamespace(eef_pose=self.pose, joint_positions=self.joints)
    def act(self, action, **kwargs):
        if self.failure: raise self.failure
        self.calls.append((self.clock.monotonic(), action.values))
        if self.follow: self.pose = action.values


def live(recording, tmp_path, *, robot=None, **kwargs):
    ep, config, _ = recording
    clock = robot.clock if robot is not None else Clock()
    robot = robot or Robot(clock)
    result = replay(config, load_episode(ep), live=True, robot_factory=lambda _: robot,
                    clock=clock, log_dir=tmp_path/'logs', **kwargs)
    return robot, result


def test_command_and_measured_use_physical_units_and_actual_published_commands(recording):
    ep, _, _ = recording
    a = load_episode(ep)
    assert a.poses[0, 0] == .4
    assert a.widths[0] == .02
    b = load_episode(ep, source='measured')
    assert b.poses[0, 0] == .399
    assert b.widths[0] == .015
    assert a.duration == pytest.approx(.1)


def test_slice_and_single_frame(recording):
    ep, _, _ = recording
    a = load_episode(ep/'control.msgpack', start=1, end=2)
    assert a.duration == 0
    assert a.sample(10)[2] == 1
    with pytest.raises(ValueError): load_episode(ep, end=8)
    with pytest.raises(ValueError): load_episode(ep, start=-1)


@pytest.mark.parametrize('fault', ['nan', 'zero_quaternion', 'gap', 'duplicate_time', 'missing_tick', 'width', 'joints'])
def test_bad_recording_refused(recording, fault):
    ep, _, rows = recording
    if fault == 'nan': rows[1]['published_pose'][0] = float('nan')
    if fault == 'zero_quaternion': rows[1]['published_pose'][3:] = [0]*4
    if fault == 'gap': rows[-1]['publish_start'] += 1
    if fault == 'duplicate_time': rows[1]['publish_start'] = rows[0]['publish_start']
    if fault == 'missing_tick': rows[1]['tick'] = 5
    if fault == 'width': rows[1]['commanded_gripper_m'] = .9
    if fault == 'joints': rows[1]['robot_state']['joint_positions'] = [0]*7
    write_rows(ep, rows)
    with pytest.raises(ValueError): load_episode(ep)


@pytest.mark.parametrize('field,value', [('status', 'pending'), ('robot_type', 'flexiv')])
def test_incomplete_or_wrong_robot_refused(recording, field, value):
    ep, _, _ = recording
    m = json.loads((ep/'manifest.json').read_text())
    if field == 'status': m['status'] = value
    else: m['config']['robot']['type'] = value
    (ep/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): load_episode(ep)


def test_interpolation_uses_shortest_quaternion_path_and_holds_gripper(recording):
    ep, _, rows = recording
    rows[1]['published_pose'][3:] = [-1., 0., 0., 0.]
    write_rows(ep, rows)
    t = load_episode(ep)
    pose, width, tick = t.sample(1/60)
    assert pose[0] == pytest.approx(.4005)
    assert abs(pose[3]) == pytest.approx(1.)
    assert width == .02
    assert tick == 0


def test_slerp_rotating_quaternion(recording):
    ep, _, rows = recording
    rows[1]['published_pose'][3:] = [np.cos(.2), 0., 0., np.sin(.2)]
    write_rows(ep, rows)
    pose, _, _ = load_episode(ep).sample(1/60)
    assert pose[3:] == pytest.approx([np.cos(.1), 0., 0., np.sin(.1)])


def test_dry_run_cannot_construct_hardware_or_create_logs(recording, tmp_path):
    ep, config, _ = recording
    def forbidden(_): pytest.fail('Hardware created during dry run')
    result = replay(config, load_episode(ep), robot_factory=forbidden, log_dir=tmp_path/'logs')
    assert result['frames'] == 4
    assert not (tmp_path/'logs').exists()


@pytest.mark.parametrize('speed', [0, -1, float('nan'), float('inf')])
def test_invalid_speed_never_connects(recording, speed):
    ep, config, _ = recording
    with pytest.raises(ValueError):
        replay(config, load_episode(ep), speed=speed, live=True,
               robot_factory=lambda _: pytest.fail('connected'))


def test_half_speed_keeps_servo_frequency_and_cleanup(recording, tmp_path):
    robot, result = live(recording, tmp_path, speed=.5)
    times = [t for t, _ in robot.calls]
    assert np.diff(times) == pytest.approx(np.full(len(times)-1, 1/30))
    assert times[-1] >= .2
    assert robot.calls[-1][1][0] == pytest.approx(.403)
    assert robot.gripper.commands[-1] == pytest.approx(.023)
    assert result['status'] == 'completed'
    assert robot.stopped and not robot.connected and not robot.gripper.connected
    assert (tmp_path/'logs'/'tracking.csv').is_file()


@pytest.mark.parametrize('failure', [RuntimeError('servoL rejected'), KeyboardInterrupt()])
def test_send_failure_and_interrupt_stop_and_preserve_logs(recording, tmp_path, failure):
    robot = Robot(Clock())
    robot.failure = failure
    with pytest.raises(type(failure)):
        live(recording, tmp_path, robot=robot)
    result = json.loads((tmp_path/'logs'/'summary.json').read_text())
    assert result['status'] in ('failed', 'interrupted')
    assert robot.stopped and not robot.connected and not robot.gripper.connected


def test_start_move_failure_sends_no_replay(recording, tmp_path):
    robot = Robot(Clock()); robot.reached = False
    with pytest.raises(RuntimeError, match='starting joints'):
        live(recording, tmp_path, robot=robot)
    assert not robot.calls and robot.stopped


def test_start_mismatch_refuses_servo_and_gripper(recording, tmp_path):
    robot = Robot(Clock()); robot.pose[0] = .9
    with pytest.raises(RuntimeError, match='Starting TCP'):
        live(recording, tmp_path, robot=robot)
    assert not robot.calls and not robot.gripper.commands and robot.stopped


def test_skip_start_requires_nearby_joints(recording, tmp_path):
    robot = Robot(Clock()); robot.joints = [1.]*6
    with pytest.raises(RuntimeError, match='no-move-to-start'):
        live(recording, tmp_path, robot=robot, move_to_start=False)
    assert not robot.calls


def test_gripper_rejection_stops_before_arm_send(recording, tmp_path):
    robot = Robot(Clock()); robot.gripper.ack = False
    with pytest.raises(RuntimeError, match='Robotiq rejected'):
        live(recording, tmp_path, robot=robot)
    assert not robot.calls and robot.stopped


def test_observation_delay_aborts_before_stale_command(recording, tmp_path):
    robot = Robot(Clock())
    original = robot.gripper.observe
    def delayed():
        robot.clock.now += .3
        return original()
    robot.gripper.observe = delayed
    with pytest.raises(RuntimeError, match='Observation I/O'):
        live(recording, tmp_path, robot=robot)
    assert not robot.calls and robot.stopped


def test_tracking_error_aborts(recording, tmp_path):
    robot = Robot(Clock()); robot.follow = False
    with pytest.raises(RuntimeError, match='Tracking error'):
        live(recording, tmp_path, robot=robot, max_position_error=.0015)
    assert robot.stopped


def test_config_gripper_mismatch_refused_before_hardware(recording):
    ep, config, _ = recording
    c = yaml.safe_load(config.read_text()); c['robot']['gripper']['max_width'] = .1
    config.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='max_width differs'):
        replay(config, load_episode(ep), live=True,
               robot_factory=lambda _: pytest.fail('connected'))
