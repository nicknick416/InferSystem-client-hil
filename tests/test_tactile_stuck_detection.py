import cv2
import numpy as np
import pytest
from HIL.tactile_stuck_detection import TactileStuckDetector, CombinedStuckDetector


def image(value):
    return cv2.imencode('.png', np.full((90,160,3), value, np.uint8))[1].tobytes()


def calibrated():
    d=TactileStuckDetector(dict(enabled=True))
    for i in range(65):
        assert not d.update(image(40), i/30)['stuck']
    return d


def test_increase_triggers_and_latches_then_human_resets():
    d=calibrated()
    rows=[d.update(image(60), i/30) for i in range(65,70)]
    assert rows[-1]['stuck'] and rows[-1]['threshold']==.75
    assert d.update(image(40), 70/30)['stuck']
    assert not d.update(image(40), 71/30, 'human')['stuck']
    assert d.update(image(40), 72/30)['reason']=='calibrating'


def test_no_trigger_stable_or_departure_decrease():
    d=calibrated()
    for i in range(65,80):
        assert not d.update(image(40), i/30)['stuck']
    d=calibrated()
    # Build deformation before arming, then unload: S high but G negative.
    d.window.clear()
    for _ in range(10):d.window.append(np.full((35,56,3),20,np.float32))
    for i in range(65,75):assert not d.update(image(40),i/30)['stuck']


def test_duplicates_gap_invalid_and_no_early_alarm():
    d=calibrated()
    for _ in range(20):assert not d.update(image(80),64/30)['stuck']
    assert d.update(image(80),4.)['reason']=='calibrating'
    assert d.update(b'bad',4.01)['reason']=='invalid_image'
    assert d.update(None,None)['reason']=='invalid_input'


def test_or_tactile_without_state_error_and_single_entry():
    d=CombinedStuckDetector(dict(enabled=True,window_mode='feedback'),dict(enabled=True))
    rows=[]
    for i in range(75):
        d.set_observation({'observation/left_tactile':image(40 if i<65 else 60)}, {'observation/left_tactile':i/30})
        rows.append(d.update(i/30,[0]*3,[0]*3,source={'request_id':1},generation=1))
    entered=[r for r in rows if r['event']=='entered']
    assert len(entered)==1 and entered[0]['trigger_sources']==['tactile']
    assert not entered[0]['state_action']['stuck']
    d.reset()
    assert not d.stuck and d.tactile.reference is None


def test_or_state_action_during_tactile_calibration():
    d=CombinedStuckDetector(dict(enabled=True,window_mode='feedback',feedback_samples=3),dict(enabled=True))
    for i in range(4):
        d.set_observation({'observation/left_tactile':image(40)}, {'observation/left_tactile':i/30})
        r=d.update(i/30,[0]*3,[.004,0,0],source={'request_id':1},generation=1)
    assert r['event']=='entered' and r['trigger_sources']==['state_action']


def test_tactile_only_and_disabled_baseline():
    d=CombinedStuckDetector({},dict(enabled=True))
    assert d.update(0,[0]*3,[0]*3)['reason']=='state_action_disabled'
    d=CombinedStuckDetector(dict(enabled=True,window_mode='feedback'),{})
    assert 'tactile' not in d.update(0,[0]*3,[0]*3)


@pytest.mark.parametrize('cfg',[dict(enabled=1),dict(arm_after_s=1),dict(threshold_floor=-1),dict(foo=1)])
def test_config_rejects_invalid(cfg):
    with pytest.raises(ValueError):TactileStuckDetector(cfg)
