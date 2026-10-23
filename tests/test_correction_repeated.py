from test_correction import bridge, capture, STATE
import json,time

def test_auto_limit_does_not_limit_later_manual_uploads(tmp_path):
    b=bridge(tmp_path,auto_execute=True,max_auto_per_episode=1)
    try:
        b.auto_count=1
        ids=[]
        for i in range(3):
            ids.append(capture(b))
            assert b.predict() is False
            b.human()
            r=b.resume(STATE,{},2+i,{},8+i)
            assert r['status']=='queued'
        deadline=time.monotonic()+4
        while len(list((b.data/'sent').glob('*.json')))<3 and time.monotonic()<deadline:
            time.sleep(.02)
        sent=[json.loads(p.read_text()) for p in (b.data/'sent').glob('*.json')]
        assert {r['sample_id'] for r in sent}==set(ids)
    finally:b.close()
