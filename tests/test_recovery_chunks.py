import pytest
from HIL.stuck_recovery import StuckRecovery
from test_stuck_recovery import state


def recovery():return StuckRecovery({'enabled':True,'target_mode':'before_last_moving_chunk'})

def observe(r,t,x,request=None,index=0,mode='policy',stuck=False):
    r.observe(t,state(x),{'stuck':stuck},mode,
              source=None if request is None else dict(request_id=request,chunk_index=index),generation=3)


def test_last_moving_chunk_anchor_is_its_predecessor_not_trigger_predecessor():
    r=recovery()
    observe(r,0,0,10,7)
    observe(r,.1,.001,11,5)
    observe(r,.2,.003,11,6)  # chunk 11 moved; target is chunk 10 start
    observe(r,.3,.003,12,9)
    observe(r,.4,.003,13,4)  # multiple stationary chunks must not advance anchor
    observe(r,.5,.003,13,5,stuck=True)
    report=r.start(.5,state(.003),state(.003))
    assert report['target_time']==0
    assert report['target_state'][0]==0
    assert report['last_moving_chunk']['request_id']==11
    assert report['target_source']==dict(generation=3,request_id=10,first_executed_chunk_index=7)
    assert report['status']=='retreating'


def test_boundary_observation_is_previous_chunk_result_and_hold_is_not_chunk():
    r=recovery()
    observe(r,0,0,1);observe(r,.1,0,2)
    observe(r,.2,.002,3)  # closes moving chunk 2 before starting 3
    assert r.motion_source['request_id']==2 and r.checkpoint_source['request_id']==1
    observe(r,.3,.004)  # motion while holding chunk 3 command; same actual chunk
    assert r.motion_source['request_id']==3 and r.checkpoint_source['request_id']==2
    assert r.checkpoint[0]==.1


def test_no_time_window_threshold_chunk_can_move_in_less_than_half_second():
    r=recovery()
    observe(r,0,0,1);observe(r,.03,0,2);observe(r,.06,.002,2)
    assert r.checkpoint[0]==0
    assert r.checkpoint_source['request_id']==1


@pytest.mark.parametrize('reset',['human','gap'])
def test_human_and_gap_clear_motion_and_chunk_history(reset):
    r=recovery()
    observe(r,0,0,1);observe(r,.1,0,2);observe(r,.2,.002,2)
    if reset=='human':observe(r,.3,.002,mode='human')
    observe(r,.6,.003,3);observe(r,.7,.005,3)
    assert r.start(.7,state(.005),state(.005))['reason']=='no_valid_history'


def test_first_chunk_has_no_predecessor_and_disable_is_noop():
    r=recovery();observe(r,0,0,1);observe(r,.1,.002,1)
    assert r.checkpoint is None
    r=StuckRecovery({'target_mode':'before_last_moving_chunk'})
    observe(r,0,0,1);observe(r,.1,.002,2)
    assert not r.chunks
