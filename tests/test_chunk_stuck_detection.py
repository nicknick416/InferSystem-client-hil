import numpy as np
import pytest
from HIL.stuck_detection import StuckDetector


def detector(**kw):
    return StuckDetector(dict(enabled=True,window_mode='executed_chunk',**kw))


def run_chunk(d,request=1,start=0,n=8,dt=.02,moving=False,target=.004,end=True):
    rows=[]
    for i in range(n):
        x=i*.001 if moving else 0
        rows.append(d.update(start+i*dt,[x,0,0],[target,0,0],source={'request_id':request,'chunk_index':i+5},
                             generation=2,chunk_end=end and i==n-1))
    x=n*.001 if moving else 0
    rows.append(d.update(start+n*dt,[x,0,0],[target,0,0],fresh_command=False))
    return rows


def test_full_chunk_and_last_action_feedback_no_fixed_time_window():
    d=detector()
    rows=run_chunk(d)
    assert not any(r['stuck'] for r in rows[:-1])
    r=rows[-1]
    assert r['event']=='entered' and r['samples']==8 and r['window_span_s']==pytest.approx(.16)
    assert r['evaluated_chunk']['request_id']==1
    assert r['evaluated_chunk']['first_executed_chunk_index']==5
    assert d.update(.18,[0]*3,[.004,0,0],fresh_command=False)['evaluated_chunk'] is None


def test_new_chunk_boundary_closes_only_executed_old_prefix():
    d=detector()
    run_chunk(d,n=6,end=False)
    r=d.update(.14,[0]*3,[.004,0,0],source={'request_id':2,'chunk_index':9},generation=2)
    assert r['event']=='entered' and r['evaluated_chunk']['request_id']==1
    assert d.chunk_detector.chunk['source']['request_id']==2


def test_motion_return_to_start_not_mistaken_for_stall():
    d=detector()
    for i,x in enumerate([0,.002,0,.002,0,.002]):
        d.update(i*.02,[x,0,0],[.01,0,0],source={'request_id':1},chunk_end=i==5)
    r=d.update(.12,[0]*3,[.01,0,0],fresh_command=False)
    assert r['reason']=='moving' and not r['stuck']


def test_confirmation_is_consecutive_chunks_and_short_chunk_resets():
    d=detector(confirm_chunks=2)
    assert run_chunk(d)[-1]['reason']=='suspected'
    assert run_chunk(d,request=2,start=.18)[-1]['event']=='entered'
    assert run_chunk(d,request=3,start=.36,n=2)[-1]['reason']=='short_chunk'
    assert not d.stuck
    assert run_chunk(d,request=4,start=.42)[-1]['reason']=='suspected'


def test_fraction_direction_and_low_error_guards():
    d=detector()
    assert run_chunk(d,target=.0001)[-1]['reason']=='below_error_threshold'
    d=detector()
    for i in range(8):
        d.update(i*.02,[0]*3,[.01*(-1)**i,0,0],source={'request_id':1},chunk_end=i==7)
    assert d.update(.16,[0]*3,[.01,0,0],fresh_command=False)['reason']=='direction_changed'


def test_gap_human_and_unknown_source_cannot_confirm():
    for mode in ('human','policy'):
        d=detector();run_chunk(d,n=6,end=False)
        r=d.update(1.,[0]*3,[.004,0,0],mode=mode,source={'request_id':1})
        assert r['reason']==('non_policy' if mode=='human' else 'sample_gap')
        assert not r['stuck']
    d=detector()
    for i in range(50):
        assert not d.update(i*.02,[0]*3,[.004,0,0])['stuck']


@pytest.mark.parametrize('config',[{'window_mode':'bad'},{'confirm_chunks':0}, {'confirm_chunks':True},
                                  {'min_error_fraction':0},{'min_error_fraction':float('nan')}])
def test_invalid_chunk_config(config):
    with pytest.raises(ValueError):StuckDetector(config)
