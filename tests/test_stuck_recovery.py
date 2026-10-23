"""Hardware-free recovery tests: timing, pose tracking, stale actions, and bounds."""
import numpy as np
import pytest
from HIL.stuck_recovery import StuckRecovery
from HIL.control import Arbiter
from Inference.dispatch import quaternion_to_rotation6d


def state(x=0., angle=0.):
    out = np.zeros(32)
    out[0] = x
    out[3:9] = quaternion_to_rotation6d([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    out[9] = .04
    return out


def ready(**config):
    r = StuckRecovery(dict(enabled=True, **config))
    r.checkpoint = (0., state(0., .04))
    current = state(.006)
    published = current.copy(); published[9] = .25
    assert r.start(1., current, published)['status'] == 'retreating'
    return r, current


def test_checkpoint_is_measured_start_of_last_motion_window_not_last_false():
    r = StuckRecovery({'enabled': True})
    for i in range(61):
        t = i/30
        r.observe(t, state(min(t, 1.)*.01), {'stuck': False, 'reason':'below_error_threshold'}, 'policy')
    stamp, saved = r.checkpoint
    assert stamp < 1. and .008 <= saved[0] <= .0091
    assert r.start(2., state(.01), state(.01))['status'] == 'retreating'
    np.testing.assert_equal(r.target[:9], saved[:9])


def test_full_pose_speed_bounds_gripper_preserved_and_hold():
    r, current = ready()
    last = current.copy(); last[9] = .25
    for i in range(1, 180):
        action, report = r.step(1+i/30, current)
        action = np.array(action)
        assert np.linalg.norm(action[:3]-last[:3]) <= .005/30+1e-10
        assert r.angle(action,last) <= .1/30+1e-7
        assert action[9] == .25
        current = action.copy(); current[9] = .04
        last = action
        if report['status'] == 'holding':break
    assert report['reason'] == 'completed'
    assert r.at_target(current)
    held, _ = r.step(9, state(.01))
    np.testing.assert_allclose(held, action)


def test_nonfollowing_arm_has_bounded_lead_then_times_out_to_measured_hold():
    r,current = ready(timeout_s=2.)
    for i in range(1,61):
        action, report = r.step(1+i/30,current)
        assert np.linalg.norm(np.array(action[:3])-current[:3]) <= .002+1e-10
    assert report['reason'] == 'timeout'
    np.testing.assert_allclose(action[:9],current[:9])
    assert action[9] == .25


def test_gap_holds_and_invalid_history_never_moves():
    r,current = ready()
    action, report = r.step(1.3,current)
    assert report['reason'] == 'sample_gap'
    np.testing.assert_allclose(action[:9],current[:9])
    for checkpoint, reason in [(None,'no_valid_history'), ((-10,state()),'history_expired'),
                               ((0,state(.1)),'target_too_far'), ((0,state(0, .5)),'target_rotation_too_large')]:
        r=StuckRecovery({'enabled':True});r.checkpoint=checkpoint
        assert r.start(1,current,current)['reason']==reason
        action,_=r.step(1.1,current)
        np.testing.assert_allclose(action,current)


def test_human_and_sample_gap_invalidate_checkpoint():
    r,current=ready();r.reset()
    r.checkpoint=(0,state());r.observe(.1,current,{},'human')
    assert r.checkpoint is None
    r.observe(.2,current,{'stuck':False},'policy');r.checkpoint=(0,state())
    r.observe(1.,current,{'stuck':False},'policy')
    assert r.checkpoint is None


def test_recovery_invalidates_queue_inflight_response_and_resume():
    a=Arbiter();a.switch('policy');old=a.generation
    response={'actions':[state().tolist()]*50}
    assert a.accept(response,old,30)
    a.switch('recovery')
    assert not a.actions and a.last_action is None
    assert not a.accept(response,old,30)
    assert not a.accept(response,a.generation,30)
    a.switch('recovery_hold');a.switch('policy')
    assert not a.accept(response,old,30)
    assert a.accept(response,a.generation,30)
    a.switch('recovery');a.switch('human')
    assert not a.actions


@pytest.mark.parametrize('config',[{'enabled':'yes'},{'timeout_s':0},{'max_distance_m':float('nan')},
                                  {'typo':1},{'max_distance_m':.0001},{'history_s':.2}])
def test_invalid_config(config):
    with pytest.raises(ValueError):StuckRecovery(config)


def test_disabled_default_has_no_history():
    r=StuckRecovery();r.observe(0,None,None,'policy')
    assert not r.enabled and not r.history
