"""Offline HIL tests: never instantiate robot SDKs, cameras, or HID devices."""
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest

from HIL.control import Arbiter, PolicyWorker
from HIL.recording import Episode, records
from HIL.spacemouse import SpaceMouse, human_action
from HIL.runtime import check_config, published_human
from Inference.action_processing import canonicalize_action_values
from Inference.dispatch import ActionDispatcher
from Core import ActionSpace, ArmState


STATE = [.4, 0., .3, 1., 0., 0., 0., 1., 0., .05] + [0.]*22


def test_takeover_preserves_shadow_and_resume_invalidates_it():
    a = Arbiter()
    a.switch('policy')
    generation = a.generation
    response = {'actions': [STATE]*50}
    assert a.accept(response, generation, 30)
    a.switch('human')
    assert len(a.actions) == 30
    assert a.generation == generation
    assert a.accept(response, generation, 30)  # shadow only, never selected during HUMAN
    a.switch('policy')
    assert not a.accept(response, generation, 30)
    assert a.accept(response, a.generation, 30)
    a.switch('paused')
    assert not a.actions
    assert not a.accept(response, a.generation, 30)


def test_async_replacement_skips_executed_steps_not_append_stale_tail():
    a = Arbiter()
    a.switch('policy')
    actions = [list(STATE) for _ in range(50)]
    for i, row in enumerate(actions):
        row[0] = i
    a.actions.extend([STATE]*20)
    assert a.accept({'actions': actions}, a.generation, 30, skip_steps=7)
    assert len(a.actions) == 30
    assert a.actions[0]['action'][0] == 7
    assert a.actions[-1]['action'][0] == 36
    with pytest.raises(ValueError):
        a.accept({'actions': [[float('nan')]*32]}, a.generation, 30)


def test_mouse_deadzone_latch_separate_reports_and_buttons():
    m = SpaceMouse()
    m.feed([1, 0, 0, 10, 0, 0, 0])
    assert not m.activity.is_set()
    m.feed([1, 0, 0, 175, 0, 0, 0])
    assert m.activity.is_set()
    assert m.sample()[0][0] == .5
    m.feed([1, 0, 0, 0, 0, 0, 0])
    assert m.activity.is_set()  # brief movement cannot be lost between control ticks
    m.activity.clear()
    m.feed([2, 0, 0, 175, 0, 0, 0])
    assert m.sample()[0][3] == -.5
    m.feed([3, 1])
    assert m.sample()[1] == 1
    m.error = OSError('unplug')
    with pytest.raises(RuntimeError):
        m.sample()


def test_human_absolute_pose_gripper_units_and_no_both_button_home():
    a = human_action(STATE, [1, 0, 0, 0, 0, 1], 1, 1/30)
    assert a[0] == pytest.approx(.402)
    assert a[9] == pytest.approx(.49)
    assert a[10:] == [0.]*22
    pose = canonicalize_action_values(a, 'cartesian', policy_format='canonical', effective_action_dim=10)
    assert np.linalg.norm(pose[3:7]) == pytest.approx(1.)
    held = human_action(STATE, np.zeros(6), 3, 1/30)
    assert held[:9] == STATE[:9]
    assert held[9] == .5


def test_recorded_human_is_post_dispatch_pose_and_gripper_command():
    class Robot:
        dof = 7
        def act(self, action, state=None):
            self.sent = action.values
    class Grip:
        def observe(self):
            class S:
                max_width = .1
            return S()
        def move(self, target, wait=False):
            self.target = target
    r, g = Robot(), Grip()
    d = ActionDispatcher(r, g, arm_dim=7, gripper_index=7, gripper_mode='width',
                         gripper_action_is_10x_meters=True, action_space=ActionSpace.CARTESIAN,
                         max_eef_xyz_step=.001)
    req = human_action(STATE, [1,0,0,0,0,0], 1, 1/30)
    converted = canonicalize_action_values(req, 'cartesian', policy_format='canonical', effective_action_dim=10)
    d.dispatch(converted, state=ArmState(timestamp=0, eef_pose=[.4,0,.3,1,0,0,0]))
    saved = published_human(d, req, 10.)
    assert saved[0] == pytest.approx(.401)
    assert saved[9] == pytest.approx(g.target*10.)
    assert len(saved) == 32


def execution_tick(i, *, request=1, index=None, mode='human', valid=True):
    source = dict(request_id=request, chunk_index=i if index is None else index,
                  tick=0, observation_time=0.) if request is not None else None
    return dict(tick=i, segment=1, mode=mode, state=STATE, state_time=i/30.,
                publish_start=i/30., publish_time=i/30.+.001, human_action=STATE if mode=='human' else None,
                policy_action=STATE if source else None, policy_source=source,
                policy_observation_age_s=i/30. if source else None,
                published_action=STATE, pair_valid=valid and mode=='human' and source is not None,
                observation={'state':STATE, 'observation/image':b'exact-wire'})


def test_recording_execution_pairs_variable_chunk_lengths_and_decisions(tmp_path):
    ep = Episode(tmp_path, {'fps':30, 'horizon':50})
    # Chunk 1 contributes 3 steps, chunk 2 contributes 2, then queue starvation.
    for i in range(3):
        ep.add('tick', **execution_tick(i, request=1))
    for i in range(3, 5):
        ep.add('tick', **execution_tick(i, request=2, index=i-3))
    ep.add('tick', **execution_tick(5, request=None))
    ep.seal()
    assert ep.path.name.startswith('.pending_')
    ep.metadata['outcome'] = 'success'
    path = ep.decide(True)
    with (path/'pairs.msgpack').open('rb') as f:
        pairs=list(msgpack.Unpacker(f,raw=False))
    with (path/'execution_segments.msgpack').open('rb') as f:
        spans=list(msgpack.Unpacker(f,raw=False))
    assert len(pairs)==5
    assert [p['tick'] for p in pairs]==list(range(5))
    assert all(len(p['policy_action'])==32 for p in pairs)
    assert [p['length'] for p in spans]==[3,2]
    assert pairs[0]['observation_ref']['tick']==0
    assert pairs[3]['policy_source']['request_id']==2
    assert not pairs[0]['policy_executed']
    assert next(records(path))['observation']['observation/image']==b'exact-wire'
    discarded=Episode(tmp_path,{})
    discarded.add('event',reason='test')
    discarded_path=discarded.decide(False)
    assert not discarded_path.exists()
    assert (path/'records.msgpack').exists()


def test_invalid_policy_and_timing_gaps_are_explicit(tmp_path):
    ep=Episode(tmp_path,{'fps':30})
    ep.add('tick',**execution_tick(0))
    row=execution_tick(1, valid=False)
    row['publish_start']=2.
    ep.add('tick',**row)
    ep.metadata['outcome'] = 'success'
    path = ep.decide(True)
    with (path/'pairs.msgpack').open('rb') as f:
        assert len(list(msgpack.Unpacker(f,raw=False)))==1
    with (path/'execution_segments.msgpack').open('rb') as f:
        assert len(list(msgpack.Unpacker(f,raw=False)))==2


def test_network_is_async_single_thread_and_reset_serialized():
    worker = PolicyWorker('unused')
    gate = threading.Event()
    ids, calls = [], []
    def fake_call(payload):
        ids.append(threading.get_ident())
        calls.append(payload['cmd'])
        if payload['cmd'] == 'predict':
            gate.wait(2)
            return {'actions':[STATE]*50}
        return {'status':'ok'}
    worker._call = fake_call
    try:
        start = time.monotonic()
        worker.submit({'cmd':'predict'}, tick=12, generation=3)
        assert time.monotonic()-start < .1
        assert worker.poll() is None
        with pytest.raises(RuntimeError):
            worker.reset()
        a = Arbiter()
        a.switch('human')  # succeeds while network is blocked
        assert a.mode == 'human'
        gate.set()
        result = worker.drain()
        assert result['tick'] == 12 and result['generation'] == 3
        worker.reset()
        assert calls == ['predict','reset']
        assert len(set(ids)) == 1 and ids[0] != threading.get_ident()
    finally:
        gate.set()
        worker.close()


def test_config_offline():
    cfg, raw = check_config(Path(__file__).resolve().parents[1]/'Config/flexiv_vtla_hil.yaml')
    assert cfg.inference.async_inference.enabled
    assert raw['hil']['horizon'] == 50


@pytest.mark.parametrize('server_capture', [False, True])
def test_full_mock_runtime_takeover_reset_then_keep(monkeypatch, tmp_path, server_capture):
    """Exercise the real runtime through injected devices, including r -> home -> y."""
    pytest.importorskip("cv2", reason="full mock runtime needs installed camera package, never opens cameras")
    import HIL.runtime as rt
    import HIL.observations as obs
    import Robot
    import Sensor.manager
    import Sensor.tactile.ring_buffer
    from types import SimpleNamespace
    events = []
    mouse = SpaceMouse()
    mouse.open = lambda: events.append('mouse_open_fake')
    mouse.close = lambda: None
    monkeypatch.setattr(rt, 'SpaceMouse', lambda *args: mouse)
    class R:
        dof = 7
        connected = False
        def connect(self): self.connected=True
        def is_connected(self): return self.connected
        def is_fault(self): return False
        def enable(self): pass
        def wait_until_operational(self, **kw): return True
        def observe(self):
            return ArmState(timestamp=time.monotonic(), eef_pose=[.4,0,.3,1,0,0,0], joint_positions=[0.]*7)
        def act(self, action, state=None): events.append(('act', action.values[0]))
        def stop(self): events.append('stop')
        def move_joint_position(self, *args, **kwargs):
            events.append('home')
            events.append(('home_velocity', kwargs['velocity']))
            return True
        def disconnect(self): self.connected=False
        def create_gripper(self): return G()
    class G:
        def connect(self): pass
        def disconnect(self): pass
        def observe(self): return SimpleNamespace(width=.05,max_width=.1)
        def move(self, target, **kwargs): pass
    robot = R()
    monkeypatch.setattr(Robot.BaseRobot, 'from_config', lambda *args: robot)
    def fake_sensor_manager(*args, **kwargs):
        assert kwargs['enabled_names'] == ['third_view', 'left_wrist_view']
        return SimpleNamespace(open_all=lambda:None, close_all=lambda:None)
    monkeypatch.setattr(Sensor.manager.SensorManager, 'from_config', fake_sensor_manager)
    class Ring:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def wait_for_frames(self, *args): return True
        def clear(self): pass
    monkeypatch.setattr(Sensor.tactile.ring_buffer, 'TactileRingBuffer', Ring)
    class C:
        def __init__(self, *args): pass
        def start(self): pass
        def close(self): pass
        def snapshot(self, state, rings, prompt, age, history_frames=30):
            return {'cmd':'predict','state':state,'observation/image':b'fake-png'}, {'rgb':time.monotonic()}
    monkeypatch.setattr(obs, 'CameraWorker', C)
    class W:
        busy = False
        def __init__(self, *args): self.i=0; self.last=None
        def begin_episode(self, eid, metadata):
            events.append('server_begin')
            return dict(episode_id=eid, token='test')
        def end_episode(self): events.append('server_end')
        def control(self, payload):
            events.append(payload['cmd'])
            return {'status':'ok'}
        def reset(self): events.append('policy_reset')
        def submit(self, payload, **ctx):
            self.i+=1; self.busy=True
            self.last=dict(ctx, request_id=self.i, request_time=time.monotonic(),
                           response={'actions':[list(STATE)]*50})
            # Deployed policy deliberately continues a different x target.
            for row in self.last['response']['actions']:
                row[0]=.41
            return self.i
        def poll(self):
            if not self.busy: return None
            self.busy=False
            return dict(self.last, response_time=time.monotonic())
        def drain(self):
            events.append('drain')
            return self.poll()
        def close(self): pass
    monkeypatch.setattr(rt, 'PolicyWorker', W)
    class K:
        def __init__(self): self.answers=iter(['\n','\n','s','q']); self.i=0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, choices, message):
            answer=next(self.answers)
            events.append(('answer',answer))
            return answer
        def read(self):
            self.i+=1
            if self.i==3:
                mouse.feed([1,0,0,175,0,0,0])
                events.append('takeover')
            return 'r' if self.i==58 else ''
    monkeypatch.setattr(rt, 'Keyboard', K)
    path=Path(__file__).resolve().parents[1]/'Config/flexiv_vtla_hil.yaml'
    cfg, raw=check_config(path)
    raw['hil']['output_dir']=str(tmp_path)
    raw['hil']['server_capture']=server_capture
    rt.run_live(path,cfg,raw,'test')
    kept=list(tmp_path.glob('[0-9]*/[0-9]*_episode_*'))
    assert len(kept)==1
    rows=list(records(kept[0]))
    human=[r for r in rows if r['kind']=='tick' and r['mode']=='human']
    assert len(human)==55
    assert all(r['human_action'][0] == pytest.approx(.401) for r in human)
    assert all(r['observation']['state']==r['state'] for r in human)
    pairs=list(msgpack.Unpacker((kept[0]/'pairs.msgpack').open('rb'), raw=False))
    assert len(pairs) == len(human)
    # Smooth-enabled runtime records the actual shadow command, including blending.
    assert all(.4 <= p['policy_action'][0] <= .410001 for p in pairs)
    assert pairs[-1]['policy_action'][0] == pytest.approx(.41)
    assert all(p['policy_action'] == r['policy_action'] for p,r in zip(pairs,human))
    assert all(r['policy_raw_action'][0] == pytest.approx(.41) for r in human)
    assert pairs[0]['tick'] == human[0]['tick']  # takeover tick uses already queued candidate
    assert pairs[0]['policy_source']['tick'] < pairs[0]['tick']
    assert all(r['robot_send_start'] <= r['robot_send_end'] <= r['publish_time'] for r in human)
    takeover_index = events.index('takeover')
    assert all(e[1] == pytest.approx(.401) for e in events[takeover_index:]
               if isinstance(e, tuple) and e[0] == 'act')
    home_indices=[i for i,e in enumerate(events) if e=='home']
    assert len(home_indices)==2
    assert home_indices[-1]<events.index('drain')<events.index(('answer','s'))
    assert ('home_velocity', raw['hil'].get('home_velocity_scale', .30)) in events
    if server_capture:
        assert events.index('server_end') > home_indices[-1]
        assert events.index('hil_upload') > events.index(('answer','s'))
        assert events.index('hil_decide') > events.index('hil_upload')
    end=[r for r in rows if r['kind']=='episode_end']
    assert len(end)==1 and end[0]['reason']=='r' and end[0]['homing_recorded'] is False
    assert end[0]['last_tick']==max(r['tick'] for r in rows if r['kind']=='tick')
    end_index=next(i for i,r in enumerate(rows) if r['kind']=='episode_end')
    assert not any(r['kind']=='tick' for r in rows[end_index+1:])
    assert not any(isinstance(e,tuple) and e[0]=='act' for e in events[home_indices[-1]:])


def test_shadow_preview_is_pure_and_matches_real_dispatch():
    from HIL.execution import preview_candidate
    from types import SimpleNamespace
    class R:
        dof=7
        count=0
        def act(self, action, state=None): self.count+=1
    class G:
        count=0
        def observe(self): return SimpleNamespace(max_width=.1)
        def move(self, width, wait=False): self.count+=1
    r,g=R(),G()
    d=ActionDispatcher(r,g,arm_dim=7,gripper_index=7,gripper_mode='width',
                       gripper_action_is_10x_meters=True, action_space=ActionSpace.CARTESIAN,
                       max_eef_xyz_step=.001)
    arm=ArmState(timestamp=0,eef_pose=[.4,0,.3,1,0,0,0])
    cfg=SimpleNamespace(max_eef_delta_m=.15,max_eef_rotation_delta_rad=.8)
    requested=human_action(STATE,[1,0,0,0,0,0],1,1/30)
    preview=preview_candidate(d,requested,arm,cfg,.1,10.)
    assert r.count==g.count==0
    assert d._last_arm_cmd is None and d._last_gripper_width_cmd is None
    assert preview['action'][0]==pytest.approx(.401)
    converted=canonicalize_action_values(requested,'cartesian',policy_format='canonical',effective_action_dim=10)
    d.dispatch(converted,state=arm)
    assert preview['action']==pytest.approx(published_human(d,requested,10.))
    requested[0]=1.
    bad=preview_candidate(d,requested,arm,cfg,.1,10.)
    assert not bad['valid'] and bad['action'] is None
    assert r.count==g.count==1


def test_human_release_latches_pose_despite_feedback_drift():
    from HIL.spacemouse import HumanController
    c=HumanController()
    c.step(STATE,[1,0,0,0,0,1],0,1/30)
    released=list(STATE)
    released[0]+=.001
    first=c.step(released,np.zeros(6),0,1/30)
    assert first[:9]==released[:9]
    noisy=list(released)
    noisy[0]+=.002
    noisy[2]-=.003
    # Valid changed orientation feedback: hold must not follow it.
    noisy[3:9]=[0.,1.,0.,-1.,0.,0.]
    for _ in range(10):
        held=c.step(noisy,np.zeros(6),2,1/30)
        assert held[:9]==first[:9]
    c.reset()
    new=c.step(noisy,np.zeros(6),0,1/30)
    assert new[:9]==noisy[:9]


def test_gripper_target_accumulates_when_feedback_does_not_move():
    from HIL.spacemouse import HumanController
    c=HumanController()
    state=list(STATE)
    state[9]=.01
    for _ in range(37):
        a=c.step(state,np.zeros(6),2,1/30)
    assert a[9]==pytest.approx(.47)  # 47mm x10, not stuck at feedback+1mm
    released=c.step(state,np.zeros(6),0,1/30)
    assert released[9]==pytest.approx(.47)
    for _ in range(100):
        a=c.step(state,np.zeros(6),1,1/30)
    assert a[9]==0.
    for _ in range(150):
        a=c.step(state,np.zeros(6),2,1/30)
    assert a[9]==1.


def test_server_commit_uploads_only_after_keep_and_pending_refused(tmp_path):
    import json
    from HIL.server_recording import sync_decision
    from HIL.recording import atomic_json
    ep=Episode(tmp_path,{'fps':30})
    atomic_json(ep.path/'server_transaction.json',dict(episode_id=ep.episode_id,token='test',server='unused'))
    class W:
        def __init__(self): self.calls=[]
        def control(self,payload):
            self.calls.append(payload)
            return {'status':'ok'}
    w=W()
    with pytest.raises(ValueError): sync_decision(ep.path,w)
    assert not w.calls
    ep.add('tick',**execution_tick(0))
    ep.metadata['outcome'] = 'success'
    path = ep.decide(True)
    sync_decision(path,w)
    assert [p['cmd'] for p in w.calls]==['hil_end','hil_upload','hil_decide']
    assert w.calls[-1]['keep'] and w.calls[-1]['batches']==1
    assert 'observation' not in w.calls[1]['rows'][0]
    assert json.loads((path/'server_transaction.json').read_text())['sync_status']=='committed'


def test_policy_worker_attaches_and_verifies_hil_receipt():
    w=PolicyWorker('unused')
    captured=[]
    def fake(payload):
        captured.append(payload)
        if payload['cmd']=='hil_begin': return {'status':'ok','token':'test'}
        if payload['cmd']=='predict':
            return dict(actions=[STATE]*50,hil=dict(captured[-1]['hil'],captured=True))
        return {'status':'ok'}
    w._call=fake
    try:
        w.begin_episode('a'*32,{'fps':30,'prompt':'task'})
        w.submit({'cmd':'predict','state':STATE},tick=7,generation=1)
        result=w.drain()
        assert result['response']['hil']['request_id']==1
        assert captured[-1]['hil']['tick']==7
        w.end_episode()
        assert w.hil_session is None
    finally:w.close()


def test_server_kept_end_failure_is_retryable(tmp_path):
    import json
    from HIL.server_recording import sync_decision
    from HIL.recording import atomic_json
    ep=Episode(tmp_path,{'fps':30})
    atomic_json(ep.path/'server_transaction.json',dict(episode_id=ep.episode_id,token='test',server='unused'))
    ep.metadata['outcome'] = 'failure'
    path=ep.decide(True)
    class W:
        def control(self,payload): raise TimeoutError('lost end acknowledgement')
    with pytest.raises(TimeoutError): sync_decision(path,W())
    assert json.loads((path/'server_transaction.json').read_text())['sync_status']=='retry_required'


@pytest.mark.parametrize('choice,outcome', [('s','success'),('f','failure'),('n',None),('q',None)])
def test_pure_policy_explicit_outcome(tmp_path,choice,outcome):
    from HIL.outcome import choose_episode_decision
    ep=Episode(tmp_path,{'fps':30,'horizon':50})
    ep.add('tick',tick=0,mode='policy')
    ep.add('episode_end',last_tick=0)
    class K:
        def wait(self,keys,message):
            assert keys=='sfnq'
            return choice
    result=choose_episode_decision(ep,K())
    assert result==({'s':True,'f':True,'n':False,'q':None}[choice])
    assert ep.metadata.get('outcome')==outcome


def test_client_reports_confirmed_q_switch(capsys):
    w=PolicyWorker('unused')
    replies=iter(['old/critic.pt','old/critic.pt','new/critic.pt',None])
    def fake(payload):
        if payload['cmd']=='hil_begin':
            return dict(status='ok',token='test',serving_critic=next(replies))
        return dict(status='ok')
    w._call=fake
    try:
        w.begin_episode('a'*32,{})
        assert '本轮使用 Q' in capsys.readouterr().out
        w.end_episode();w.begin_episode('b'*32,{})
        assert 'Q 未变' in capsys.readouterr().out
        w.end_episode();w.begin_episode('c'*32,{})
        text=capsys.readouterr().out
        assert 'Q 已切换' in text and '旧 Q：old/critic.pt' in text and '新 Q：new/critic.pt' in text
        w.end_episode();w.begin_episode('d'*32,{})
        text=capsys.readouterr().out
        assert '无法确认' in text and 'Q 已切换' not in text
    finally:w.close()


@pytest.mark.parametrize('roll',[False,True])
@pytest.mark.parametrize('pitch',[False,True])
@pytest.mark.parametrize('yaw',[False,True])
@pytest.mark.parametrize('combined',[False,True])
def test_spacemouse_rotation_switches(roll,pitch,yaw,combined):
    m=SpaceMouse(rotation_roll_enabled=roll,rotation_pitch_enabled=pitch,rotation_yaw_enabled=yaw)
    if combined:
        m.feed([1,0,0,0,0,0,0,175,0,175,0,175,0])
    else:
        m.feed([2,175,0,175,0,175,0])
    a,b=m.sample()
    assert bool(a[3])==roll
    assert bool(a[4])==pitch
    assert bool(a[5])==yaw
    assert m.activity.is_set()==(roll or pitch or yaw)
    m.feed([1,175,0,0,0,0,0])
    assert np.any(m.sample()[0][:3]) and m.activity.is_set()
    m.feed([3,2]);assert m.sample()[1]==2


def test_lossless_blob_dedup_and_compact_reader(tmp_path):
    from HIL.recording import compact_records
    from HIL.outcome import choose_episode_decision
    import shutil
    data=b'image'*1000
    ep=Episode(tmp_path,{'fps':30,'horizon':50})
    for i in range(4):
        ep.add('tick',tick=i,mode='policy',observation={'image':data,'extra':{'history':[data]*30}})
    ep.add('episode_end',last_tick=3)
    ep.seal()
    assert len(list((ep.path/'blobs').glob('*.bin')))==1
    restored=list(records(ep.path))
    assert restored[0]['observation']['image']==data
    assert restored[0]['observation']['extra']['history']==[data]*30
    assert all('observation' not in r for r in compact_records(ep.path))
    assert (ep.path/'control.msgpack').stat().st_size<1000
    # End decisions must not touch any image files.
    shutil.rmtree(ep.path/'blobs')
    class K:
        def wait(self,*args):return 'f'
    assert choose_episode_decision(ep,K()) is True
    assert ep.metadata['outcome']=='failure'


def test_compact_reader_legacy_compatibility(tmp_path):
    from HIL.recording import compact_records
    rows=[{'kind':'tick','tick':0,'observation':{'image':b'old'},'mode':'policy'},
          {'kind':'episode_end','last_tick':0}]
    (tmp_path/'records.msgpack').write_bytes(b''.join(msgpack.packb(r,use_bin_type=True) for r in rows))
    assert list(records(tmp_path))==rows
    assert list(compact_records(tmp_path))[0]=={'kind':'tick','tick':0,'mode':'policy'}


def test_current_tactile_payload_preserves_latest_and_baseline():
    from HIL.observations import CameraWorker
    from types import SimpleNamespace
    import threading
    worker = CameraWorker.__new__(CameraWorker)
    worker.error = None
    worker.lock = threading.Lock()
    worker.latest = ({'observation/image': b'rgb'}, time.monotonic())
    frames = [bytes([i]) for i in range(30)]
    ring = SimpleNamespace(wire_key='observation/left_tactile', name='left',
        snapshot_timed=lambda: (frames, b'baseline', time.monotonic()))
    for count in (1, 30):
        payload, _ = worker.snapshot([0.]*32, [ring], 'test', history_frames=count)
        assert payload['extra']['tactile_history'][ring.wire_key] == frames[-count:]
        assert payload[ring.wire_key] == frames[-1]
        assert payload['extra']['tactile_baseline'][ring.wire_key] == b'baseline'
    assert len(frames) == 30


def test_hil_chunk_smoothing_preserves_raw_and_shadow_alignment():
    a=Arbiter(smooth_steps=8);a.switch('policy')
    old=list(STATE);new=list(STATE);new[2]+=.03
    a.accept({'actions':[old]*50},a.generation,30,source={'request_id':1})
    for _ in range(28):
        row=a.actions.popleft();a.track(row['action'],{'request_id':1})
    a.switch('human')
    a.accept({'actions':[new]*50},a.generation,30,6,source={'request_id':2})
    rows=list(a.actions)
    assert rows[0]['chunk_index']==6
    assert rows[0]['action'][2]==pytest.approx(old[2])
    assert rows[0]['raw_action']==new
    assert rows[0]['blend_source_old']['request_id']==1
    assert rows[8]['action']==new
    xyz=np.asarray([r['action'][:3] for r in rows])
    assert np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()<=.03/8+1e-9
    old_generation=a.generation;a.switch('policy')
    assert a.last_action is None and not a.actions
    assert not a.accept({'actions':[new]*50},old_generation,30)
    assert a.accept({'actions':[new]*50},a.generation,30)
    assert a.actions[0]['action']==new


def test_hil_queue_gap_and_expired_response_do_not_retract_command():
    a=Arbiter(smooth_steps=8);a.switch('policy');a.track(STATE)
    new=list(STATE);new[0]+=.02
    assert a.accept({'actions':[new]*50},a.generation,30,6)
    assert np.allclose(a.actions[0]['action'],STATE)
    previous=list(a.actions)
    assert not a.accept({'actions':[new]*50},a.generation,30,50)
    assert list(a.actions)==previous
