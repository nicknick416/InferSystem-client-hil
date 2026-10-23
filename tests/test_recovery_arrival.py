import numpy as np
import pytest
from HIL.stuck_recovery import StuckRecovery
from test_stuck_recovery import ready,state


def test_near_target_requires_five_feedbacks_then_preserves_reference():
    r,current=ready(motion_mode='inference',position_tolerance_m=.001,arrival_confirm_samples=5)
    r.step(1+1/30,current)
    measured=r.target.copy();measured[0]+=.00055;measured[9]=.04
    for i in range(2,7):
        action,report=r.step(1+i/30,measured)
        np.testing.assert_allclose(action,r.target)
        assert report['status']==('holding' if i==6 else 'retreating')
    assert report['reason']=='completed' and report['arrival_samples']==5
    assert report['position_error_m']==pytest.approx(.00055)
    held,_=r.step(2.,measured)
    np.testing.assert_allclose(held,r.target)


def test_leaving_tolerance_resets_consecutive_arrival():
    r,current=ready(motion_mode='inference',position_tolerance_m=.001,arrival_confirm_samples=5)
    r.step(1+1/30,current)
    for i in range(2,6):r.step(1+i/30,r.target)
    r.step(1+6/30,current)
    assert r.arrival_samples==0
    for i in range(7,11):assert r.step(1+i/30,r.target)[1]['status']=='retreating'
    assert r.step(1+11/30,r.target)[1]['reason']=='completed'


@pytest.mark.parametrize('value',[0,True,1.5])
def test_arrival_config(value):
    with pytest.raises(ValueError):StuckRecovery({'arrival_confirm_samples':value})
