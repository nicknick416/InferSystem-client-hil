"""RTDE upload acknowledgements must not be mistaken for program readiness."""
from test_ur_robot import _connected_robot


class DelayedControl:
    def __init__(self, clock, *, delay=.03, upload_ok=True, fault=False):
        self.clock = clock
        self.delay = delay
        self.upload_ok = upload_ok
        self.started_at = None
        self.moves = []
        self.stop_calls = 0
        self.fault = fault

    def isConnected(self):
        return True

    def reuploadScript(self):
        self.started_at = self.clock[0]
        return self.upload_ok

    def isProgramRunning(self):
        return self.started_at is not None and self.clock[0] >= self.started_at + self.delay

    def moveJ(self, *args):
        self.moves.append((self.clock[0], args))
        return self.isProgramRunning()

    def isSteady(self):
        return True

    def stopJ(self, *args):
        self.stop_calls += 1


def setup_robot(monkeypatch, **kwargs):
    import Robot.ur as ur
    clock = [0.]
    monkeypatch.setattr(ur.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(ur.time, 'sleep', lambda delta: clock.__setitem__(0, clock[0] + delta))
    robot = _connected_robot()
    robot._rtde_c = DelayedControl(clock, **kwargs)
    robot._rtde_r.isProtectiveStopped = lambda: robot._rtde_c.fault
    robot._rtde_r.isEmergencyStopped = lambda: False
    return robot, clock


def test_home_waits_for_program_after_successful_upload(monkeypatch):
    robot, clock = setup_robot(monkeypatch)
    assert robot.move_joint_position([0.] * 6, timeout_s=.2)
    assert len(robot._rtde_c.moves) == 1
    assert robot._rtde_c.moves[0][0] >= .03


def test_home_never_sends_motion_when_program_does_not_start(monkeypatch, caplog):
    robot, clock = setup_robot(monkeypatch, delay=100.)
    assert robot.move_joint_position([0.] * 6, timeout_s=.05) is False
    assert robot._rtde_c.moves == []
    assert robot._rtde_c.stop_calls == 0
    assert 'control script did not start' in caplog.text


def test_home_rejects_fault_before_upload_or_motion(monkeypatch):
    robot, clock = setup_robot(monkeypatch, fault=True)
    assert robot.move_joint_position([0.] * 6, timeout_s=.2) is False
    assert robot._rtde_c.started_at is None
    assert robot._rtde_c.moves == []


def test_home_aborts_fault_while_waiting_for_script(monkeypatch):
    robot, clock = setup_robot(monkeypatch, delay=100.)
    robot._rtde_r.isProtectiveStopped = lambda: clock[0] >= .02
    assert robot.move_joint_position([0.] * 6, timeout_s=.2) is False
    assert robot._rtde_c.moves == []
    assert clock[0] < .2


def test_home_rejected_upload_never_sends_motion(monkeypatch):
    robot, clock = setup_robot(monkeypatch, upload_ok=False)
    assert robot.move_joint_position([0.] * 6, timeout_s=.2) is False
    assert robot._rtde_c.moves == []


def test_home_lost_script_during_motion_does_not_retry(monkeypatch, caplog):
    robot, clock = setup_robot(monkeypatch, delay=0.)
    def move(*args):
        robot._rtde_c.moves.append((clock[0], args))
        robot._rtde_c.delay = 100.
        return True
    robot._rtde_c.moveJ = move
    robot._rtde_r.getActualQ = lambda: [1.] * 6
    assert robot.move_joint_position([0.] * 6, timeout_s=.2) is False
    assert len(robot._rtde_c.moves) == 1
    assert clock[0] < .2
    assert 'control script stopped during' in caplog.text
