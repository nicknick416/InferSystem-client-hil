import cv2
import numpy as np
from HIL.tactile_slip import TactileSlip
from HIL.tactile_stuck_detection import TactileStuckDetector

def detector():
    return TactileSlip(dict(TactileStuckDetector.DEFAULTS,slip_enabled=True,slip_arm_s=.1))

def frame(offset):
    gray=np.random.default_rng(3).integers(40,220,(45,80),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(3,3),.5)
    moved=cv2.warpAffine(gray,np.float32([[1,0,offset],[0,1,0]]),(80,45),borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(moved,cv2.COLOR_GRAY2BGR)

def test_consistent_texture_motion_and_static():
    d=detector()
    for i in range(15):
        assert not d.update(frame(0),i/30,.5,True)['slipping']
    d.reset()
    for i in range(15):r=d.update(frame(i*.6),i/30,.5,True)
    assert r['slipping'] and r['speed_px_s']>8 and r['coherence']>.65

def test_coldstart_release_gap_duplicate_reset():
    d=detector()
    for i in range(15):
        assert not d.update(frame(i*.6),i/30,.5,False)['slipping']
    for i in range(15):r=d.update(frame(i*.6),1+i/30,.5,True)
    assert r['slipping']
    count=d.count
    assert d.update(frame(8.4),1+14/30,.5,True)['reason']=='duplicate_frame'
    assert d.count==count
    assert not d.update(frame(9),1.5,.1,True)['slipping']
    assert not d.update(frame(0),3,.5,True)['slipping']
    assert not d.update(frame(.6),3+1/30,.3,True)['slipping']

def test_localnormal_slip_merges_with_energy_and_coldstart(tmp_path):
    from HIL.tactile_normal import LocalNormal
    c=dict(TactileStuckDetector.DEFAULTS,normal_data_dir=str(tmp_path),slip_enabled=True)
    d=LocalNormal(c);d.poll_ready=lambda:None
    d.set_zero(cv2.imencode('.png',np.zeros((45,80,3),np.uint8))[1].tobytes())
    d.samples=[[130.]*40]*10;d.corrector_ready=True
    for i in range(20):r=d.update(frame(i*.6),i/30,0,True,False)
    assert r['stuck'] and 'tactile_slip' in r['trigger_reasons'] and not r['energy_stuck']
    d.corrector_ready=False
    r=d.update(frame(12),20/30,0,True,False)
    assert not r['stuck'] and not r['slip']['slipping']
