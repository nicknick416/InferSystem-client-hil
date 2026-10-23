from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import pytest

from Core import RobotParams
from Core.config_schema import FlexivRobotConfig
from Robot.flexiv import FlexivGripper, FlexivRobot


def test_flexiv_gripper_connect_initializes_before_opening(monkeypatch):
    events = []

    class FakeGripperHandle:
        def Enable(self, name):
            events.append(("Enable", name))

        def Init(self):
            events.append(("Init",))

        def Move(self, width, velocity, force):
            events.append(("Move", width, velocity, force))

    class FakeToolHandle:
        def Switch(self, name):
            events.append(("Switch", name))

    class FakeRDK:
        def Gripper(self, robot_handle):
            events.append(("Gripper", robot_handle))
            return FakeGripperHandle()

        def Tool(self, robot_handle):
            events.append(("Tool", robot_handle))
            return FakeToolHandle()

    robot_handle = object()
    gripper = FlexivGripper(
        FakeRDK(),
        robot_handle,
        name="Flexiv-GN01",
        config={
            "max_width": 0.1,
            "default_velocity": 0.2,
            "default_force": 15.0,
        },
    )
    monkeypatch.setattr(
        gripper,
        "_wait_init",
        lambda: events.append(("wait_init",)),
    )
    monkeypatch.setattr(
        gripper,
        "_wait_done",
        lambda: events.append(("wait_done",)),
    )

    gripper.connect()

    assert events == [
        ("Gripper", robot_handle),
        ("Enable", "Flexiv-GN01"),
        ("Tool", robot_handle),
        ("Switch", "Flexiv-GN01"),
        ("Init",),
        ("wait_init",),
        ("Move", 0.1, 0.2, 15.0),
        ("wait_done",),
    ]
    assert gripper._connected is True


def test_flexiv_gripper_wait_init_requires_stable_not_moving(monkeypatch):
    import Robot.flexiv as flexiv_module

    clock = [0.0]
    moving_states = iter([True, False, False, False, False, False, False])

    class FakeGripperHandle:
        def states(self):
            return SimpleNamespace(is_moving=next(moving_states, False))

    gripper = FlexivGripper(object(), object())
    gripper._gripper = FakeGripperHandle()
    monkeypatch.setattr(flexiv_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        flexiv_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    gripper._wait_init(timeout_s=2.0)

    assert clock[0] >= 1.1


def test_flexiv_gripper_wait_init_raises_on_timeout(monkeypatch):
    import Robot.flexiv as flexiv_module

    clock = [0.0]

    class FakeGripperHandle:
        def states(self):
            return SimpleNamespace(is_moving=True)

    gripper = FlexivGripper(object(), object())
    gripper._gripper = FakeGripperHandle()
    monkeypatch.setattr(flexiv_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        flexiv_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    with pytest.raises(RuntimeError, match="夹爪初始化未在 0.3s 内完成"):
        gripper._wait_init(timeout_s=0.3)


def test_flexiv_build_params_uses_configured_home_position_deg():
    cfg = FlexivRobotConfig.model_validate(
        {
            "type": "flexiv",
            "serial_number": "Rizon4-test",
            "control": {
                "home_position_deg": [0, -10, 20, 30, 40, 50, 60],
            },
        }
    )

    params = FlexivRobot._build_params(cfg)

    assert params.home_position == pytest.approx(
        [math.radians(v) for v in [0, -10, 20, 30, 40, 50, 60]]
    )


def test_flexiv_go_home_moves_to_configured_home_position():
    robot = FlexivRobot.__new__(FlexivRobot)
    robot._connected = False
    robot._robot = None
    robot._params = RobotParams(
        dof=7,
        joint_position_min=[-10.0] * 7,
        joint_position_max=[10.0] * 7,
        joint_velocity_max=[1.0] * 7,
        joint_acceleration_max=[1.0] * 7,
        joint_torque_max=[1.0] * 7,
        home_position=[0.0, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    robot._control_config = {"home_velocity_scale": 30}
    calls = []

    def fake_move_joint_position(self, positions, *, velocity=None, timeout_s=30.0):
        calls.append((list(positions), velocity, timeout_s))
        return True

    robot.move_joint_position = MethodType(fake_move_joint_position, robot)

    assert FlexivRobot.go_home(robot, timeout_s=12.0) is True
    assert calls == [
        ([0.0, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.3, 12.0),
    ]


def test_flexiv_go_home_explicit_velocity_overrides_home_velocity_scale():
    robot = FlexivRobot.__new__(FlexivRobot)
    robot._connected = False
    robot._robot = None
    robot._params = RobotParams(
        dof=7,
        joint_position_min=[-10.0] * 7,
        joint_position_max=[10.0] * 7,
        joint_velocity_max=[1.0] * 7,
        joint_acceleration_max=[1.0] * 7,
        joint_torque_max=[1.0] * 7,
        home_position=[0.0] * 7,
    )
    robot._control_config = {"home_velocity_scale": 30}
    calls = []

    def fake_move_joint_position(self, positions, *, velocity=None, timeout_s=30.0):
        calls.append((list(positions), velocity, timeout_s))
        return True

    robot.move_joint_position = MethodType(fake_move_joint_position, robot)

    assert FlexivRobot.go_home(robot, velocity=0.8, timeout_s=5.0) is True
    assert calls == [([0.0] * 7, 0.8, 5.0)]
