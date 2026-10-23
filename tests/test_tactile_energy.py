import time
import cv2
import numpy as np
from HIL.tactile_energy import TactileEnergy
from HIL.tactile_stuck_detection import TactileStuckDetector
from HIL.tactile_process import TactileProcess

CFG=dict(TactileStuckDetector.DEFAULTS,enabled=True,method='energy')
def im(v):return np.full((90,160,3),v,np.uint8)
def zero():return cv2.imencode('.png',im(20))[1].tobytes()


def test_two_point_squeeze_release_and_recalibration():
    d=TactileEnergy(CFG);d.set_zero(zero())
    assert d.update(im(20),0,0,True,False)['score']==0
    for i in range(1,6):assert not d.update(im(40),i/30,0,False,False)['stuck']
    r=d.update(im(40),6/30,1,True,False)
    assert r['energy']==.5 and r['threshold']==.65
    assert not d.update(im(50),7/30,1,True,False)['stuck']
    assert not d.update(im(50),8/30,1,True,False)['stuck']
    assert d.update(im(50),9/30,1,True,False)['stuck']
    d.reset()
    assert d.update(im(20),10/30,1,True,False)['energy']==0
    assert not d.update(im(20),11/30,1,True,True)['stuck']
    assert d.update(im(50),12/30,1,True,False)['reason']=='waiting_hold_p'


def test_weak_hold_cannot_arm_and_gap_breaks_confirmation():
    d=TactileEnergy(CFG);d.set_zero(zero())
    assert d.update(im(20),0,1,True,False)['reason']=='waiting_hold_p'
    for i in range(1,7):d.update(im(40),i/30,1,False,False)
    d.update(im(40),7/30,2,True,False)
    for i in (8,9):assert not d.update(im(60),i/30,2,True,False)['stuck']
    assert not d.update(im(60),1.,2,True,False)['stuck']


def test_energy_process_p_and_opening_gate():
    w=TactileProcess(CFG);w.start();w.set_context(zero(),.02)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            now=time.monotonic();r=w.update(im(40),now,'human',now)
            if r.get('score') is not None:break
            time.sleep(.034)
        assert r.get('score')==20,r
        w.calibrate_hold(.02)
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            now=time.monotonic();r=w.update(im(40),now,'policy',now)
            if r.get('energy')==.5:break
            time.sleep(.034)
        assert r.get('energy')==.5,r
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            now=time.monotonic();r=w.update(im(60),now,'policy',now)
            if r['stuck']:break
            time.sleep(.034)
        assert r['stuck'],r
        w.set_context(None,.04)
        now=time.monotonic()
        assert not w.update(im(60),now,'policy',now)['stuck']
    finally:w.close()
