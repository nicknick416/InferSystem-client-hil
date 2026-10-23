import json
import numpy as np
from test_correction import bridge,capture,prediction,set_result,STATE
from HIL.correction_geometry import absolute


def test_xyz_demo_keeps_raw_rotation_but_only_checks_translation(tmp_path):
    b=bridge(tmp_path,action_mode='xyz')
    try:
        capture(b);b.human()
        target=STATE.copy()
        target[:9]=absolute(STATE[:9],[.003,0,0,.2,0,0],max_rotation=.3)
        assert b.resume(target,{},2,{},8)['status']=='queued'
    finally:b.close()
    paths=list((b.data/'outbox').glob('*.json'))+list((b.data/'sent').glob('*.json'))
    sample=json.loads(paths[0].read_text())
    assert sample['action_mode']=='xyz'
    np.testing.assert_allclose(sample['target_pose'],target[:9])


def test_xyz_rejects_rotating_predictions_and_mode_mismatch(tmp_path):
    b=bridge(tmp_path,action_mode='xyz',auto_execute=True)
    try:
        for fault in ('none','rotation','mode'):
            cycle=capture(b);r=prediction(cycle,STATE[:9]);r['action_mode']='xyz'
            if fault=='rotation':
                r['poses'][0]=absolute(STATE[:9],[.002,0,0,.01,0,0])
            if fault=='mode':r['action_mode']='se3'
            set_result(b,r)
            result=b.poll(STATE)
            assert (result is not None)==(fault=='none')
    finally:b.close()
