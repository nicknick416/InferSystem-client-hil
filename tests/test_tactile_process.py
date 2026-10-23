import time
import numpy as np
from HIL.tactile_process import TactileProcess
from HIL.tactile_stuck_detection import TactileStuckDetector


def test_process_trigger_reset_nonblocking_and_close():
    worker=TactileProcess(dict(TactileStuckDetector.DEFAULTS,enabled=True))
    worker.start()
    pixels=np.full((480,640,3),40,np.uint8)
    try:
        # Real capture clock; covers spawn startup and per-segment calibration.
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            now=time.monotonic()
            r=worker.update(pixels,now,'policy',now)
            if r.get('reason')=='below_threshold':break
            time.sleep(1/30)
        assert r.get('reason')=='below_threshold',r
        durations=[]
        pixels[:]=60
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            now=time.monotonic()
            r=worker.update(pixels,now,'policy',now)
            durations.append(time.monotonic()-now)
            if r['stuck']:break
            time.sleep(1/30)
        assert r['stuck'],r
        assert r['processing_ms']>=0
        # Keep old positive result in shared memory: reset must reject it.
        worker.reset()
        now=time.monotonic()
        r=worker.update(pixels,now,'policy',now)
        assert not r['stuck']
        # Simulate worker holding both locks: control call returns immediately.
        worker.input_lock.acquire();worker.output_lock.acquire()
        try:
            now=time.monotonic()
            r=worker.update(pixels,now,'policy',now)
            assert time.monotonic()-now<.05
        finally:
            worker.input_lock.release();worker.output_lock.release()
        assert not worker.update(pixels,time.monotonic(),'human',time.monotonic())['stuck']
        now=time.monotonic()
        worker.update(pixels,now,'policy',now)
        worker.last=dict(stuck=True,generation=worker.generation,capture_time=time.monotonic()-1)
        worker.output_lock.acquire()
        try:
            now=time.monotonic()
            assert not worker.update(pixels,now,'policy',now)['stuck']
        finally:worker.output_lock.release()
        print('control submission max ms',max(durations)*1000)
    finally:
        process=worker.process
        worker.close()
        assert not process.is_alive()
