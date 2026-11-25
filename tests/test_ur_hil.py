"""UR HIL contract checks: fake SDKs only, never open devices."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from Core import ActionSpace, ArmState
from HIL.robot_support import gripper_action_scale, live_dependencies
from Inference.action_processing import canonicalize_action_values
from Inference.dispatch import ActionDispatcher
from test_ur_robot import _connected_robot


@pytest.mark.parametrize('package,filename', [
    ('HIL', 'ur_vtla_rtc_hil.yaml'),
    ('HIL', 'ur_vtla_nonrtc_hil.yaml'),
    ('HIL_PI', 'ur_pi_rtc_hil.yaml'),
])
def test_configs_select_ur_dependencies_and_units(package, filename, monkeypatch):
    runtime = importlib.import_module(package + '.runtime')
    cfg, raw = runtime.check_config(Path('Config') / filename)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None if name == 'flexivrdk' else object())
    assert all(live_dependencies(cfg).values())
    assert 'flexivrdk' not in live_dependencies(cfg)
    assert {'rtde_control', 'rtde_receive'} <= live_dependencies(cfg).keys()
    assert gripper_action_scale(cfg) == pytest.approx(1/.085)
    assert raw['hil']['stuck_recovery']['enabled'] is False


@pytest.mark.parametrize('package', ['HIL', 'HIL_PI'])
@pytest.mark.parametrize('field,value,message', [
    ('gripper_action_is_10x_meters', True, 'cannot also'),
    ('gripper_state_is_10x_meters', True, 'state in meters'),
    ('fps', 20, 'frequency_hz'),
    ('inference_home', [0]*7, 'six-joint'),
    ('inference_home_gripper_width', .1, 'gripper range'),
])
def test_invalid_ur_contract_rejected(package, field, value, message, tmp_path):
    runtime = importlib.import_module(package + '.runtime')
    filename = 'ur_pi_rtc_hil.yaml' if package == 'HIL_PI' else 'ur_vtla_rtc_hil.yaml'
    raw = yaml.safe_load((Path('Config')/filename).read_text())
    raw['inference'][field] = value
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        runtime.check_config(path)


@pytest.mark.parametrize('package', ['HIL', 'HIL_PI'])
def test_human_preview_publish_and_rtc_keep_normalized_action(package):
    execution = importlib.import_module(package + '.execution')
    mouse = importlib.import_module(package + '.spacemouse')
    rtc = importlib.import_module(package + '.rtc')
    state = [.4, 0., .3, 1., 0., 0., 0., 1., 0., .0425] + [0.]*22
    robot = _connected_robot()
    class Grip:
        width = .0425
        writes = 0
        def observe(self):
            return SimpleNamespace(width=self.width, max_width=.085)
        def move(self, width, **kwargs):
            self.width = width
            self.writes += 1
    grip = Grip()
    dispatch = ActionDispatcher(robot, grip, arm_dim=7, gripper_index=7,
                               gripper_mode='normalized_width', action_space=ActionSpace.CARTESIAN,
                               max_eef_gripper_step=.005)
    arm = ArmState(timestamp=0, eef_pose=[.4, 0, .3, 1, 0, 0, 0])
    # A human opening at .03 m/s for .1s requests .0455 m, not .535 m.
    human = mouse.human_action(state, np.zeros(6), 2, .1,
                              gripper_max=.085, gripper_scale=1/.085)
    assert human[9] == pytest.approx(.0455/.085)
    preview = execution.preview_candidate(dispatch, human, arm,
        SimpleNamespace(max_eef_delta_m=.3, max_eef_rotation_delta_rad=1), .085, 1/.085)
    assert preview['valid'] and grip.writes == 0
    assert robot._rtde_c.servol_calls == []
    dispatch.dispatch(canonicalize_action_values(human, 'cartesian', policy_format='canonical',
                                                effective_action_dim=10), state=arm)
    published = execution.command32(dispatch, human, 1/.085)
    assert published == pytest.approx(preview['action'])
    assert grip.width == pytest.approx(.0455)
    assert len(robot._rtde_c.servol_calls[-1][0]) == 6
    request = rtc.make_request({'state': state}, [published])
    assert request['state'][9] == .0425
    assert request['rtc']['prefix_actions'][0][9] == pytest.approx(.0455/.085)


@pytest.mark.parametrize('outcome', ['success', 'rejected', 'timeout', 'script_rejected'])
def test_ur_home_restores_script_and_checks_actual_completion(outcome):
    robot = _connected_robot()
    calls = []
    control = robot._rtde_c
    control.isProgramRunning = lambda: True
    control.reuploadScript = lambda: calls.append('script') or (outcome != 'script_rejected')
    control.moveJ = lambda *args: calls.append(('moveJ', args)) or (outcome != 'rejected')
    control.isSteady = lambda: outcome != 'timeout'
    control.stopJ = lambda *args: calls.append('stopJ')
    assert robot.move_joint_position([0]*6, velocity=.2, timeout_s=.001) is (outcome == 'success')
    assert calls[0] == 'script'
    if outcome == 'script_rejected':
        assert calls == ['script']
    else:
        assert calls[1][1] == ([0.]*6, .2, .3, True)
    assert ('stopJ' in calls) is (outcome == 'timeout')
