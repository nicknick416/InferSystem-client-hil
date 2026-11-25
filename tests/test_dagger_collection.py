"""Continuity rules and capture finalization, without connecting to a robot."""
import json
import os

import numpy as np
import pytest

from HIL.dagger_sequences import policy_tail, segments


def control(i, mode, segment, x=0.4):
    state = [x, .1, .3, 1., 0., 0., 0., 1., 0., .0425]+[0.]*22
    cmd = state.copy(); cmd[9] = .5
    t = 10+i/30
    return dict(tick=i, mode=mode, segment=segment, state=state, state_time=t,
                time=t, publish_start=t+.001, publish_time=t+.002,
                published_action=cmd, human_action=cmd if mode=='human' else None,
                robot_state={'joint_positions':[.1]*6, 'eef_pose':[x,.1,.3,1.,0.,0.,0.]},
                pair_valid=False, policy_source=None, spacemouse_axes=[0.]*6,
                spacemouse_buttons=0, commanded_gripper_m=.0425)


def test_only_trim_terminal_tracked_hold():
    rows=[control(i,'policy',0,.4 if i<22 else .42) for i in range(42)]
    assert policy_tail(rows,30)==23
    for i in range(32,42):rows[i]['published_action'][9]=.8
    assert policy_tail(rows,30)==42  # waiting for gripper
    rows[-1]['mode']='human'
    assert policy_tail(rows,30)==42
    drift=[control(i,'policy',0,.4+i*.00012) for i in range(40)]
    assert policy_tail(drift,30)==40


def test_handover_and_missing_ticks():
    rows=[control(i,m,s) for i,m,s in [(0,'policy',0),(1,'human',1),(9,'human',1),
                                        (10,'policy',2),(11,'human',3)]]
    rows[3]['published_action'][9]=.8  # actual handover command, never smooth or split full
    assert segments(rows,{(2,8)},False)==[[0,1,2,3,4]]
    assert segments(rows,{(2,8)},True)==[[1,2],[4]]
    assert segments(rows,set(),False)==[[0,1],[2,3,4]]
    rows[2]['state'][0]+=.02
    assert segments(rows,{(2,8)},True)==[[1],[2],[4]]


@pytest.mark.skipif(os.getenv('HIL_TEST_VAAPI')!='1',reason='requires local video encoder')
@pytest.mark.parametrize('outcome', ['success', 'failure'])
@pytest.mark.parametrize('whole', [False,True])
def test_saved_capture_and_both_datasets(tmp_path, outcome, whole):
    import cv2
    import pyarrow.parquet as pq
    from HIL.recording import Episode
    from HIL.dagger_export import CAMS, export_episode
    from HIL.dagger_storage import video_info
    from HIL.video_storage import iter_records
    opts=dict(recording={'format':'collector_h265'}, omit_settled_human_frames=True,
              trim_terminal_policy_hold=True,export_dagger_training=True)
    if whole:
        del opts['omit_settled_human_frames'],opts['trim_terminal_policy_hold']
        opts['omit_stationary_frames']=True
    ep=Episode(tmp_path,dict(fps=30,horizon=50,outcome=outcome,prompt='Load tray.',config={'hil':opts}))
    def img(i):return cv2.imencode('.png',np.full((128,256,3),i,np.uint8))[1].tobytes()
    for i in range(75):
        mode,seg=('policy',0) if i<4 else ('human',1) if i<31 else ('policy',2) if i<38 else ('human',3) if i<42 else ('policy',4)
        x=.4 if i<31 else .4+min(i-30,18)*.001
        row=control(i,mode,seg,x)
        if i in (4,38):row['spacemouse_axes'][0]=1.
        row['observation']={k:img(i+30) for k in CAMS}
        row['observation']['extra']={'tactile_baseline':{k:img(8) for k in CAMS if 'tactile' in k}}
        row['sensor_times']={k:row['state_time'] for k in CAMS}
        ep.add('tick',**row)
    ep.add('episode_end',last_tick=74,time=13.)
    if whole:
        from HIL.outcome import choose_episode_decision
        class Keyboard:
            def wait(self,keys,prompt):
                assert 's' in keys and 'f' in keys
                return 's' if outcome=='success' else 'f'
        assert choose_episode_decision(ep,Keyboard()) is True
    kept=ep.decide(True);ep.seal();m=json.loads((kept/'metadata.json').read_text())
    assert m['training_export']=={'full_episode':1,'human_only':2}
    assert m['source_tick_count']==75 and m['omitted_human_idle_frames']>0
    assert m['source_terminal_last_tick']==74 and m['terminal_last_tick']==48
    raw=[r for r in iter_records(kept) if r['kind']=='tick']
    removed=m['omitted_stationary_frames'] if whole else m['omitted_human_idle_frames']+m['omitted_policy_tail_frames']
    assert len(raw)==m['total_frames']==75-removed
    if whole:
        assert [r['tick'] for r in raw]==[0,4,5,*range(31,49)]
        assert [r['frame_index'] for r in raw]==list(range(len(raw)))
        import av
        for cam in CAMS:
            ref=raw[0]['observation'][cam]
            with av.open(str(kept/ref['stream']/'video.mp4')) as f:
                means=[im.to_ndarray(format='rgb24').mean() for im in f.decode(video=0)]
            np.testing.assert_allclose(means,[r['tick']+30 for r in raw],atol=4)
    assert export_episode(kept)=={'full_episode':0,'human_only':0}
    for view,groups in [('full_episode',[raw]),('human_only',[[r for r in raw if r['segment']==s] for s in (1,3)])]:
        root=tmp_path/'training'/view
        entries=[json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
        assert [e['outcome'] for e in entries]==[outcome]*len(groups)
        for p,rows in zip(sorted(root.glob('data/*/*.parquet')),groups,strict=True):
            d=pq.read_table(p).to_pydict();n=len(rows)
            assert d['source_tick']==[r['tick'] for r in rows] and d['frame_index']==list(range(n))
            np.testing.assert_allclose(d['timestamp'],np.arange(n)/30,atol=1e-6)
            np.testing.assert_allclose(d['action'],[r['published_action'] for r in rows])
            assert d['is_human']==[r['mode']=='human' for r in rows]
            vids=list(root.glob(f'videos/*/*/{p.stem}.mp4'));assert len(vids)==4
            assert all(video_info(v)[0]==n for v in vids)


def test_export_requires_kept_labeled_episode(tmp_path):
    from HIL.dagger_export import export_episode
    path=tmp_path/'episode';path.mkdir();meta=path/'metadata.json'
    for status in ('pending','discarded'):
        meta.write_text(json.dumps({'status':status,'outcome':'failure'}))
        assert export_episode(path)=={'reason':'not_kept'}
    meta.write_text(json.dumps({'status':'kept','outcome':'unknown'}))
    with pytest.raises(ValueError,match='labeled'):
        export_episode(path)
    assert not (tmp_path/'training').exists()


def run_idle(rows):
    from HIL.stationary_recording import StationaryFrames
    class Sink:
        def __init__(self): self.events=[];self.tick_count=0
        def add(self,r): self.events.append(r);self.tick_count+=int(r['kind']=='tick')
    sink=Sink();gate=StationaryFrames(sink,30)
    for row in rows:
        gate.add(dict(kind='tick',**row));gate.add(dict(kind='diagnostic',source=row['tick']))
    gate.add(dict(kind='episode_end',last_tick=len(rows)-1));meta=gate.finish()
    assert [r['source'] for r in sink.events if r['kind']=='diagnostic']==list(range(len(rows)))
    saved=[r for r in sink.events if r['kind']=='tick'];end=sink.events[-1]
    assert end['source_last_tick']==len(rows)-1 and end['last_tick']==saved[-1]['tick']
    return saved,meta


def test_whole_idle_across_owners_and_only_certified_join():
    rows=[control(i,'human' if 8<=i<20 or i>=27 else 'policy',0 if i<8 else 1 if i<20 else 2 if i<27 else 3) for i in range(35)]
    saved,m=run_idle(rows);assert [r['tick'] for r in saved]==[0,8,20,27]
    assert m['omitted_stationary_frames']==31 and m['omitted_human_idle_frames']==18
    cert=m['omitted_stationary_ranges'];assert segments(saved,set(),False,cert)==[[0,1,2,3]]
    assert segments(saved,set(),True,cert)==[[1],[3]]
    assert segments(saved,set(),False,[])==[[0],[1],[2],[3]]
    saved,m=run_idle([control(i,'policy',0) for i in range(37)]);assert [r['tick'] for r in saved]==[0]


def test_whole_idle_protects_movement_grip_and_timing():
    rows=[control(i,'policy',0,.4+.00004*i) for i in range(36)]
    saved,_=run_idle(rows);assert len(saved)==len(rows)
    for command in (True,False):
        rows=[control(i,'policy',0) for i in range(24)]
        for i,r in enumerate(rows):
            if command:r['published_action'][9]+=.001*i
            else:r['state'][9]+=.00033*i
        saved,_=run_idle(rows);assert len(saved)==len(rows)
    rows=[control(i,'policy',0) for i in range(24)]
    rows[9]['publish_start']+=.5;rows[15]['state_time']=rows[14]['state_time']
    saved,_=run_idle(rows);assert {9,15} <= {r['tick'] for r in saved}
