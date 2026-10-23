import numpy as np
import pytest
from test_stuck_recovery import ready,state
from HIL.stuck_recovery import StuckRecovery


def test_inference_mode_sends_exact_validated_target_and_keeps_gripper():
    r,current=ready(motion_mode='inference')
    action,report=r.step(1+1/30,current)
    np.testing.assert_allclose(action,r.target)
    assert action[9]==.25 and report['status']=='retreating'
    assert report['motion_mode']=='inference'
    # A command is not proof of arrival; still retreating without measured progress.
    _,report=r.step(1+2/30,current)
    assert report['status']=='retreating'
    _,report=r.step(1+3/30,np.asarray(action))
    assert report['reason']=='completed'


def test_inference_timeout_and_gap_hold_measured_pose():
    for gap in (True,False):
        r,current=ready(motion_mode='inference',timeout_s=.1)
        r.step(1+1/30,current)
        if not gap:r.step(1+2/30,current)
        action,report=r.step(1.3 if gap else 1.11,current)
        assert report['reason']==('sample_gap' if gap else 'timeout')
        np.testing.assert_allclose(action[:9],current[:9])
        assert action[9]==.25


def test_inference_does_not_relax_historical_target_bounds():
    r=StuckRecovery({'enabled':True,'motion_mode':'inference'})
    r.checkpoint=(0,state(.1))
    assert r.start(1,state(),state())['reason']=='target_too_far'
    action,_=r.step(1.03,state())
    np.testing.assert_allclose(action,state())
    with pytest.raises(ValueError):StuckRecovery({'motion_mode':'invalid'})
