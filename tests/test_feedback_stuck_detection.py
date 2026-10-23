import numpy as np
import pytest
from HIL.stuck_detection import StuckDetector


def detector(**kw):return StuckDetector(dict(enabled=True,window_mode='feedback',**kw))

def feed(d,i,x=0,target=.004,request=1,**kwargs):
    return d.update(i/30,[x,0,0],[target,0,0],source=None if request is None else {'request_id':request,'chunk_index':i},
                    fresh_command=request is not None,generation=1,**kwargs)


def test_five_actual_feedbacks_trigger_without_chunk_end():
    d=detector()
    rows=[feed(d,i) for i in range(6)]
    assert not any(r['stuck'] for r in rows[:5])
    assert rows[5]['event']=='entered' and rows[5]['samples']==5
    assert rows[5]['window_span_s']==pytest.approx(5/30)


def test_tracking_previous_published_target_not_current_new_target():
    d=detector()
    for i in range(10):
        r=feed(d,i,x=i*.002,target=(i+1)*.002)
        assert not r['stuck']
        if i:assert r['tracking_error_m']==pytest.approx(0)


def test_movement_then_stall_inside_same_chunk():
    d=detector()
    for i in range(12):assert not feed(d,i,x=i*.001,target=.03)['stuck']
    rows=[feed(d,i,x=.011,target=.03) for i in range(12,17)]
    assert rows[-1]['event']=='entered'
    assert rows[-1]['evaluated_chunk']['request_id']==1


def test_moving_arm_and_oscillation_do_not_trigger():
    for motion in [lambda i:i*.0001,lambda i:.0004*(i%2)]:
        d=detector()
        for i in range(40):assert not feed(d,i,x=motion(i),target=.02)['stuck']


def test_chunk_boundary_does_not_mix_four_feedbacks_from_each_chunk():
    d=detector()
    for i in range(9):assert not feed(d,i,request=1 if i<4 else 2)['stuck']
    assert feed(d,9,request=2)['event']=='entered'


def test_low_error_and_reversal_break_confirmation():
    d=detector()
    for i in range(4):feed(d,i)
    feed(d,4,target=0)
    assert feed(d,5)['reason']=='below_error_threshold'
    assert not any(feed(d,i)['stuck'] for i in range(6,10))
    assert feed(d,10)['stuck']
    assert feed(d,11,target=-.004)['stuck']  # observation still belongs to old command
    assert not feed(d,12,target=-.004)['stuck']


def test_hold_expires_gap_human_and_invalid_samples_reset():
    d=detector(command_timeout_s=.1)
    feed(d,0)
    for i in range(1,9):r=feed(d,i,request=None)
    assert r['reason']=='stale_command' and not r['stuck']
    d=detector()
    for i in range(4):feed(d,i)
    assert feed(d,20)['reason']=='sample_gap'
    assert not any(feed(d,i)['stuck'] for i in range(21,25))
    assert feed(d,25,mode='human')['reason']=='non_policy'
    assert d.update(float('nan'),[0]*3,[0]*3)['reason']=='invalid_sample'


@pytest.mark.parametrize('kw',[{'feedback_samples':1},{'feedback_samples':True},
 {'feedback_max_excursion_m':0},{'feedback_max_excursion_m':float('nan')},{'feedback_max_excursion_m':.1}])
def test_invalid_config(kw):
    with pytest.raises(ValueError):detector(**kw)
