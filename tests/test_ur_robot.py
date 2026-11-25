from __future__ import annotations

from Core import Action, ActionSpace
from Core.config_schema import SystemConfig
from Robot import BaseRobot, URRobot


class _FakeRTDEControl:
    def __init__(self):
        self.servoj_calls = []
        self.servol_calls = []

    def isConnected(self):
        return True

    def servoJ(self, *args):
        self.servoj_calls.append(args)

    def servoL(self, *args):
        self.servol_calls.append(args)


class _FakeRTDEReceive:
    def getActualQ(self):
        return [0.0] * 6

    def getActualQd(self):
        return [0.0] * 6

    def getActualCurrentAsTorque(self):
        return [0.0] * 6

    def getTargetQ(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

    def getActualTCPSpeed(self):
        return [0.0] * 6

    def getActualTCPForce(self):
        return [0.0] * 6


def _connected_robot() -> URRobot:
    robot = URRobot(
        "192.0.2.10",
        rtde_control_handle=_FakeRTDEControl(),
        rtde_receive_handle=_FakeRTDEReceive(),
        control_config={"wait_step_reached": False},
    )
    robot._connected = True
    return robot


def test_ur_config_is_accepted_and_default_import_registers_robot():
    cfg = SystemConfig.model_validate(
        {
            "robot": {
                "type": "ur",
                "robot_ip": "192.0.2.10",
                "control": {"frequency_hz": 125},
                "gripper": {"enabled": True, "max_width": 0.085},
            },
            "inference": {
                "action_space": "cartesian",
                "smooth": {"enabled": True},
            },
        }
    )

    assert cfg.robot.type == "ur"
    assert "ur" in BaseRobot._registry
    robot = BaseRobot._registry["ur"]._from_config_dict(cfg.robot)
    assert isinstance(robot, URRobot)
    assert robot.robot_type == "ur"
    assert robot.get_params().control_frequency_hz == 125


def test_ur_example_config_loads():
    cfg = SystemConfig.from_yaml("Config/ur_example.yaml")

    assert cfg.robot.type == "ur"
    assert cfg.inference is not None
    assert cfg.inference.action_space.value == "joint_position"


def test_ur_config_accepts_legacy_ip_and_hz_aliases():
    cfg = SystemConfig.model_validate(
        {
            "robot": {
                "type": "ur",
                "ip": "192.0.2.10",
                "control": {"hz": 250},
            }
        }
    )

    assert cfg.robot.robot_ip == "192.0.2.10"
    robot = BaseRobot._registry["ur"]._from_config_dict(cfg.robot)
    assert robot.robot_ip == "192.0.2.10"
    assert robot.get_params().control_frequency_hz == 250


def test_ur_joint_and_cartesian_actions_call_rtde_interfaces():
    robot = _connected_robot()

    robot.act(Action(ActionSpace.JOINT_POSITION, [0.0] * 6))
    robot.act(Action(ActionSpace.CARTESIAN, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))

    assert robot._rtde_c.servoj_calls
    assert robot._rtde_c.servol_calls

def test_ur_packaged_gripper_activation_and_open_direction(monkeypatch):
    from Robot import robotiq_gripper
    from Robot.ur import URRobotiqGripper
    class Fake:
        def connect(self, *args, **kwargs): pass
        def activate(self, **kwargs): self.activation = kwargs
        def move(self, raw, **kwargs):
            self.raw = raw
            return True, raw
    monkeypatch.setattr(robotiq_gripper, "RobotiqGripper", Fake)
    gripper = URRobotiqGripper("192.0.2.2", config={"auto_calibrate_on_connect": False})
    gripper.connect()
    assert gripper._gripper.activation["auto_calibrate"] is False
    gripper.set(True)
    assert gripper._gripper.raw == 0
    gripper.move(0.0)
    assert gripper._gripper.raw == 255


def test_ur_rejected_servo_is_not_success():
    import pytest
    robot = _connected_robot()
    robot._rtde_c.servoL = lambda *args: False
    with pytest.raises(RuntimeError, match="rejected"):
        robot.act(Action(ActionSpace.CARTESIAN, [0.1, 0.2, 0.3, 1, 0, 0, 0]))


def test_ur_normalized_gripper_state_and_dispatch():
    import pytest
    from Inference.dispatch import ActionDispatcher, _state_gripper_scalar
    from Robot.gripper import GripperState
    class Gripper:
        def observe(self): return GripperState(width=0.0425, max_width=0.085)
        def move(self, width, **kwargs): self.width = width
    gripper = Gripper()
    dispatcher = ActionDispatcher.__new__(ActionDispatcher)
    dispatcher._gripper = gripper
    dispatcher._gripper_mode = "normalized_width"
    dispatcher._gripper_max_width = None
    dispatcher._last_gripper_width_cmd = None
    dispatcher._gripper_action_is_10x_meters = False
    dispatcher._gripper_wait_during_inference = False
    dispatcher._max_eef_gripper_step = 0.005
    value = _state_gripper_scalar(0.0425, mode="normalized_width", threshold=0.5, max_width=0.085)
    assert value == pytest.approx(0.0425)
    dispatcher._dispatch_gripper_width(0.5)
    assert gripper.width == pytest.approx(0.0425)
    dispatcher._dispatch_gripper_width(1.0)
    assert gripper.width == pytest.approx(0.0475)
    dispatcher._dispatch_gripper_width(0.0)
    assert gripper.width == pytest.approx(0.0425)
