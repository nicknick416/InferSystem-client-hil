"""Collector-rate translation without devices, including delayed robot feedback."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

STATE = np.array([.4, .1, .3, 1., 0., 0., 0., 1., 0., .04] + [0.] * 22)


@pytest.fixture(params=['HIL', 'HIL_PI'])
def controller(request):
    return importlib.import_module(request.param + '.spacemouse').HumanController


def test_command_increments_match_collector_and_do_not_follow_lag(controller):
    c = controller(translation_mode='integrated')
    targets = [c.step(STATE, [1, 0, 0, 0, 0, 0], 0, 1/30, linear_speed=.21)
               for _ in range(4)]
    np.testing.assert_allclose(np.array(targets)[:, 0], .4 + .007 * np.arange(1, 5))


def test_blocked_robot_never_accumulates_beyond_lead_bound(controller):
    c = controller(translation_mode='integrated', max_translation_lead_m=.04)
    for _ in range(300):
        target = c.step(STATE, [1, 1, 1, 0, 0, 0], 0, 1/30, linear_speed=.21)
        assert np.linalg.norm(np.asarray(target[:3]) - STATE[:3]) <= .04 + 1e-12
    # Release cancels the ahead-of-robot target, even after sustained obstruction.
    held = c.step(STATE, [0]*6, 0, 1/30, linear_speed=.21)
    np.testing.assert_allclose(held[:3], STATE[:3])


def test_release_hold_and_new_gesture_anchor_to_current_feedback(controller):
    c = controller(translation_mode='integrated')
    c.step(STATE, [1, 0, 0, 0, 0, 0], 0, 1/30, linear_speed=.21)
    measured = STATE.copy(); measured[0] += .002
    held = c.step(measured, [0]*6, 0, 1/30)
    drift = measured.copy(); drift[0] += .001
    held_again = c.step(drift, [0]*6, 2, 1/30)
    np.testing.assert_allclose(held_again[:3], held[:3])
    assert held_again[9] > held[9]  # Gripper-only input keeps the arm fixed.
    new = c.step(drift, [-1, 0, 0, 0, 0, 0], 0, 1/30, linear_speed=.21)
    assert new[0] == pytest.approx(drift[0] - .007)
    c.reset()
    restarted = c.step(STATE, [1, 0, 0, 0, 0, 0], 0, 1/30, linear_speed=.21)
    assert restarted[0] == pytest.approx(.407)


def test_default_measured_mode_preserves_existing_behavior(controller):
    c = controller()
    for _ in range(10):
        target = c.step(STATE, [1, 0, 0, 0, 0, 0], 0, 1/30)
        assert target[0] == pytest.approx(.402)


def test_delayed_feedback_tracks_collector_command_rate(controller):
    c = controller(translation_mode='integrated', max_translation_lead_m=.04)
    state = STATE.copy()
    speeds = []
    for _ in range(120):
        target = c.step(state, [1, 0, 0, 0, 0, 0], 0, 1/30, linear_speed=.21)
        delta = .2 * (target[0] - state[0])
        state[0] += delta
        speeds.append(delta * 30)
    assert np.median(speeds[-30:]) == pytest.approx(.21, rel=1e-5)


@pytest.mark.parametrize('bound', [0, -1, float('nan'), float('inf'), True])
def test_invalid_translation_bound_rejected(controller, bound):
    with pytest.raises(ValueError, match='max_translation_lead_m'):
        controller(translation_mode='integrated', max_translation_lead_m=bound)


@pytest.mark.parametrize('package', ['HIL', 'HIL_PI'])
def test_runtime_rejects_lead_bound_above_existing_position_guard(package, tmp_path):
    runtime = importlib.import_module(package + '.runtime')
    example = 'ur_vtla_rtc_hil.yaml' if package == 'HIL' else 'ur_pi_rtc_hil.yaml'
    raw = yaml.safe_load((Path('Config') / example).read_text())
    raw['hil'].update(translation_mode='integrated', max_translation_lead_m=.04)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    runtime.check_config(path)
    raw['hil']['max_translation_lead_m'] = raw['inference']['max_eef_delta_m'] + .01
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='must not exceed'):
        runtime.check_config(path)
