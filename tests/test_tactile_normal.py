import numpy as np
import cv2
from HIL.tactile_normal import LocalNormal
from HIL.tactile_stuck_detection import TactileStuckDetector


def image(v):return np.full((45,80,3),v,np.uint8)
def make(tmp_path):
    c=dict(TactileStuckDetector.DEFAULTS,normal_data_dir=str(tmp_path),normal_min_samples=2)
    d=LocalNormal(c);d.poll_ready=lambda:None
    d.set_zero(cv2.imencode('.png',image(20))[1].tobytes())
    return d


def test_cold_start_never_alarms_persistence_and_release(tmp_path):
    d=make(tmp_path)
    for i in range(6):d.update(image(40),i/30,0,False,False)
    d.update(image(40),6/30,1,True,False)
    d.update(image(40),7/30,2,True,False)
    assert len(d.samples)==2
    for i in range(8,15):
        r=d.update(image(100),i/30,2,True,False)
        assert not r['stuck'] and not r['ready']
    assert len(list(tmp_path.glob('sample_*.json')))==2
    d=make(tmp_path);assert len(d.samples)==2
    d.corrector_ready=True
    for i in range(3):r=d.update(image(50),1+i/30,0,True,False)
    assert r['stuck'] and r['exceeded_blocks']==40
    d.reset()
    for i in range(4):
        r=d.update(image(20),2+i/30,0,True,True)
        assert not r['stuck'] and r['energy']==0
    # Opening command itself does not mask a high score.
    for i in range(3):r=d.update(image(50),3+i/30,0,True,True)
    assert r['stuck']


def test_local_change_without_global_increase(tmp_path):
    d=make(tmp_path)
    for i in range(7):d.update(image(40),i/30,0,False,False)
    d.update(image(40),7/30,1,True,False);d.update(image(40),8/30,2,True,False)
    d.corrector_ready=True
    a=image(40)
    a[5:12,12:26]=60  # two neighboring blocks increase
    a[12:19,12:26]=20 # two others release, global mean unchanged
    for i in range(3):r=d.update(a,1+i/30,2,True,False)
    assert r['stuck'] and r['exceeded_blocks']==2
    assert abs(r['energy']-.5)<1e-6


def test_no_p_no_normal_samples(tmp_path):
    d=make(tmp_path);d.corrector_ready=True
    for i in range(10):r=d.update(image(60),i/30,0,True,False)
    assert not r['ready'] and not r['stuck'] and not d.samples
