import json,time
from concurrent.futures import Future
import numpy as np
import pytest
from HIL.correction import CorrectionBridge,config
from HIL.correction_geometry import absolute,relative

STATE=np.array([.4,0,.3,1,0,0,0,1,0,.04]+[0.]*22)


def bridge(tmp_path,**kw):
    (tmp_path/'token').write_text('test-token')
    b=CorrectionBridge(dict(enabled=True,token_file='token',data_dir='data',**kw),tmp_path)
    b.request=lambda path,payload:dict(status='ok')
    return b


def capture(b):
    return b.capture(STATE,{'observation/wrist_image':b'png'},1,{'rgb_read_complete':.99},STATE,'episode',4)


def test_correct_frame_pair_and_no_human_no_label(tmp_path):
    b=bridge(tmp_path)
    try:
        capture(b)
        assert b.resume(STATE,{},2,{},5)['reason']=='no_human_intervention'
        cycle=capture(b);b.human()
        target=STATE.copy();target[0]+=.005
        result=b.resume(target,{'observation/wrist_image':b'target'},2,{},8)
        assert result['status']=='queued'
        deadline=time.monotonic()+3
        while not list((tmp_path/'data/sent').glob('*.json')) and time.monotonic()<deadline:time.sleep(.02)
        sample=json.loads(next((tmp_path/'data/sent').glob('*.json')).read_text())
        assert sample['sample_id']==cycle and sample['anchor_tick']==4 and sample['resume_tick']==8
        assert sample['anchor_pose']==STATE[:9].tolist() and sample['target_pose']==target[:9].tolist()
    finally:b.close()


def prediction(cycle,anchor):
    return dict(status='ok',cycle_id=cycle,bounds=dict(max_translation_m=.02,max_rotation_rad=.15),
        poses=[absolute(anchor,[.002,0,0,0,0,0]),absolute(anchor,[.003,0,0,0,0,0])],scores=[.2,.7],selected=1,steps=200)


def set_result(b,r):
    b.future=Future();b.future.set_result(r);b.future_cycle=b.anchor['sample_id']


def test_selection_rejects_out_of_bounds_stale_human_and_anchor_drift(tmp_path):
    b=bridge(tmp_path,auto_execute=True)
    try:
        cycle=capture(b);r=prediction(cycle,STATE[:9]);set_result(b,r)
        assert b.poll(STATE)['selected']==1
        for problem in ('bounds','human','drift','wrong_cycle'):
            cycle=capture(b);r=prediction(cycle,STATE[:9]);current=STATE.copy()
            if problem=='bounds':r['poses'][0][0]+=.1
            if problem=='wrong_cycle':r['cycle_id']='old'
            set_result(b,r)
            if problem=='human':b.human()
            if problem=='drift':current[0]+=.005
            assert b.poll(current) is None
    finally:b.close()


def test_out_of_bounds_demo_is_saved_rejected_not_clipped(tmp_path):
    b=bridge(tmp_path)
    try:
        capture(b);b.human();target=STATE.copy();target[0]+=.03
        assert b.resume(target,{},2,{},7)['status']=='rejected'
    finally:b.close()
    sample=json.loads(next((tmp_path/'data/rejected').glob('*.json')).read_text())
    assert sample['target_pose'][0]==pytest.approx(.43)


def test_default_disabled_and_config_errors():
    assert not config()['enabled']
    for c in [{'enabled':'yes'},{'auto_execute':True},{'max_translation_m':float('nan')},{'unknown':1}]:
        with pytest.raises(ValueError):config(c)


def test_missing_anchor_is_explicit_and_durable(tmp_path):
    b=bridge(tmp_path)
    try:
        result=b.resume(STATE,{},2,{},10)
        assert result['status']=='skipped' and result['reason']=='missing_anchor'
    finally:b.close()
    events=[json.loads(line) for line in (b.data/'capture_audit.jsonl').read_text().splitlines()]
    assert any(e['event']=='resume_skipped' and e['reason']=='missing_anchor' for e in events)
