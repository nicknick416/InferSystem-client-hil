"""Offline RTC protocol and takeover tests; no robot/camera access."""
from copy import deepcopy
import numpy as np
import pytest
from HIL.rtc import make_request, validate_response, rtc_config
from HIL.control import Arbiter, PolicyWorker

STATE = [.4, 0., .3, 1., 0., 0., 0., 1., 0., .5] + [0.]*22


def response(request):
    rows = [list(STATE) for _ in range(50)]
    d = request['rtc']['prefix_length']
    rows[:d] = deepcopy(request['rtc']['prefix_actions'])
    return dict(actions=rows, rtc=dict(applied=True, mode='train_time_prefix', prefix_length=d, max_prefix_length=8))


@pytest.mark.parametrize('d', [0, 4, 8])
def test_wire_contract_preserves_png_extra_and_32d(d):
    payload = dict(cmd='predict', state=STATE, image=b'png', extra={'tactile_history': [b'png']})
    req = make_request(payload, [STATE]*d)
    assert req['cmd'] == 'predict_rtc' and payload['cmd'] == 'predict'
    assert req['image'] == b'png' and req['extra'] == payload['extra']
    validate_response(response(req), req)
    req['state'][0] = 99
    assert payload['state'][0] == .4


@pytest.mark.parametrize('fault', ['missing', 'mode', 'length', 'max', 'shape', 'nan', 'prefix', 'padding'])
def test_invalid_server_fails_closed(fault):
    req = make_request(dict(state=STATE), [STATE]*4)
    resp = response(req)
    if fault == 'missing': del resp['rtc']
    if fault == 'mode': resp['rtc']['mode'] = 'ordinary'
    if fault == 'length': resp['rtc']['prefix_length'] = 3
    if fault == 'max': resp['rtc']['max_prefix_length'] = 4
    if fault == 'shape': resp['actions'] = [[0.]*10]*50
    if fault == 'nan': resp['actions'][-1][0] = float('nan')
    if fault == 'prefix': resp['actions'][0][0] += .1
    if fault == 'padding': resp['actions'][0][31] = .1
    with pytest.raises(RuntimeError): validate_response(resp, req)


@pytest.mark.parametrize('mode', ['human', 'recovery', 'recovery_hold', 'correction', 'paused'])
def test_takeover_invalidates_inflight_and_resume_starts_empty(mode):
    arb = Arbiter(rtc_enabled=True)
    arb.switch('policy')
    old = arb.generation
    arb.accept({'actions': [STATE]*50}, old, 30)
    assert len(arb.rtc_prefix(4)) == 4
    arb.switch(mode)
    assert not arb.actions and not arb.rtc_prefix(4)
    assert not arb.accept({'actions': [STATE]*50}, old, 30)
    if mode == 'human':
        assert arb.accept({'actions': [STATE]*50}, arb.generation, 30)
        assert not arb.rtc_prefix(4)  # shadow is not executed prefix
    shadow_generation = arb.generation
    arb.switch('policy')
    assert not arb.rtc_prefix(4)
    assert not arb.accept({'actions': [STATE]*50}, shadow_generation, 30)


def test_snapshot_precedes_pop_and_hold_counts_for_alignment():
    arb = Arbiter(rtc_enabled=True)
    arb.switch('policy')
    rows = [list(STATE) for _ in range(50)]
    for i, row in enumerate(rows): row[0] += i*.001
    arb.accept({'actions': rows}, arb.generation, 2)
    prefix = arb.rtc_prefix(4)
    observed_steps = arb.policy_steps
    for _ in range(5):
        candidate = arb.actions.popleft() if arb.actions else None
        arb.advance(candidate)
    assert prefix == rows[:2]  # short queue never fabricated/padded in time
    arb.accept({'actions': rows}, arb.generation, 30, arb.policy_steps-observed_steps)
    assert arb.actions[0]['chunk_index'] == 5
    prior = deepcopy(arb.actions)
    assert not arb.accept({'actions': rows}, arb.generation, 30, 50)
    assert arb.actions == prior


def test_non_rtc_takeover_and_empty_queue_behavior_unchanged():
    arb = Arbiter()
    arb.switch('policy')
    old = arb.generation
    arb.accept({'actions': [STATE]*50}, old, 30)
    arb.switch('human')
    assert arb.generation == old and len(arb.actions) == 30
    arb.switch('policy')
    arb.advance(None)
    assert arb.policy_steps == 0


def test_worker_validates_without_fallback_and_preserves_hil():
    class Client:
        def __init__(self): self.requests = []
        def _request(self, request):
            self.requests.append(request)
            return {'actions': [STATE]*50}
        def close(self): pass
    worker = PolicyWorker('unused')
    worker.client = Client()
    req = make_request(dict(state=STATE, hil={'episode_id': 'test'}), [])
    try:
        with pytest.raises(RuntimeError): worker._call(req)
        assert len(worker.client.requests) == 1
        assert worker.client.requests[0]['hil']['episode_id'] == 'test'
        assert worker._call(dict(cmd='predict'))['actions']
    finally: worker.close()


def test_config_defaults_and_clipping_guard():
    assert rtc_config({}) == (False, 4)
    raw = {'inference': {'policy_format': 'canonical', 'rtc': {'enabled': True}}}
    assert rtc_config(raw) == (True, 4)
    for value in [True, -1, 9, 4.5]:
        bad = deepcopy(raw); bad['inference']['rtc']['prefix_length'] = value
        with pytest.raises(ValueError): rtc_config(bad)
    raw['hil'] = {'policy_motion_limits': {'enabled': True}}
    with pytest.raises(ValueError): rtc_config(raw)


def test_full_remaining_overlap_matches_reference():
    arb = Arbiter(smooth_steps=2, rtc_enabled=True)
    arb.switch('policy')
    arb.accept({'actions': [STATE]*50}, arb.generation, 10)
    new = list(STATE); new[0] += .1
    arb.accept({'actions': [new]*50}, arb.generation, 30)
    assert arb.actions[9]['blend_weight_old'] == pytest.approx(.1)
    assert 'blend_weight_old' not in arb.actions[10]


def test_zmq_rtc_request_roundtrip():
    import threading
    import msgpack
    import zmq
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    port = socket.bind_to_random_port('tcp://127.0.0.1')
    received = []
    def serve():
        try:
            req = msgpack.unpackb(socket.recv(), raw=False)
            received.append(req)
            socket.send(msgpack.packb(response(req), use_bin_type=True))
        finally: socket.close()
    thread = threading.Thread(target=serve)
    thread.start()
    worker = PolicyWorker(f'127.0.0.1:{port}', timeout_ms=3000)
    try:
        worker.submit(make_request(dict(state=STATE, image=b'png'), [STATE]*4), tick=1, generation=1)
        result = worker.drain()
        assert result['response']['actions'][:4] == [STATE]*4
        assert received[0]['cmd'] == 'predict_rtc'
    finally:
        worker.close(); thread.join(4); ctx.term()


from pathlib import Path
import time
import msgpack
from Core import ArmState
from HIL.spacemouse import SpaceMouse
from HIL.recording import records
from HIL.runtime import check_config

@pytest.mark.parametrize('server_capture', [False, True])
def test_rtc_mock_runtime_takeover_resume_reset(monkeypatch, tmp_path, server_capture):
    """Exercise the real runtime through injected devices, including r -> home -> y."""
    pytest.importorskip("cv2", reason="full mock runtime needs installed camera package, never opens cameras")
    import HIL.runtime as rt
    import HIL.observations as obs
    import Robot
    import Sensor.manager
    import Sensor.tactile.ring_buffer
    from types import SimpleNamespace
    events = []
    requests = []
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
            requests.append((deepcopy(payload), dict(ctx)))
            assert payload['cmd'] == 'predict_rtc'
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
        def __init__(self): self.answers=iter(['\n','\n','y','q']); self.i=0
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
            if self.i == 10:
                mouse.feed([1,0,0,0,0,0,0])
                return 'p'
            return 'r' if self.i==58 else ''
    monkeypatch.setattr(rt, 'Keyboard', K)
    path=Path(__file__).resolve().parents[1]/'Config/flexiv_vtla_hil.yaml'
    cfg, raw=check_config(path)
    raw['inference']['rtc'] = dict(enabled=True, prefix_length=4)
    raw['hil']['output_dir']=str(tmp_path)
    raw['hil']['server_capture']=server_capture
    rt.run_live(path,cfg,raw,'test')
    kept=list(tmp_path.glob('kept_*'))
    assert len(kept)==1
    rows=list(records(kept[0]))
    human=[r for r in rows if r['kind']=='tick' and r['mode']=='human']
    assert len(human) == 7
    assert all(r['human_action'][0] == pytest.approx(.401) for r in human)
    assert requests[0][0]['rtc']['prefix_length'] == 0
    human_ticks = {r['tick'] for r in human}
    assert all(req['rtc']['prefix_length'] == 0 for req, context in requests if context['tick'] in human_ticks)
    resumed = [(req, context) for req, context in requests if context['tick'] > max(human_ticks)]
    assert resumed and resumed[0][0]['rtc']['prefix_length'] == 0
    assert any(req['rtc']['prefix_length'] > 0 for req, _ in resumed[1:])
    assert resumed[0][1]['generation'] > requests[0][1]['generation']
    # Resume sends fresh measured state but holds the previous human command.
    after = [r for r in rows if r['kind']=='tick' and r['tick']>max(human_ticks)]
    first = next(r for r in after if r.get('policy_source'))
    waiting = [r for r in after if r['tick']<first['tick']]
    assert waiting
    assert resumed[0][0]['state'][0] == pytest.approx(.4)
    assert all(r['published_action'] == pytest.approx(human[-1]['published_action']) for r in waiting)
    assert first['policy_source']['chunk_index'] == 0
    home_indices=[i for i,e in enumerate(events) if e=='home']
    assert len(home_indices)==2
    assert home_indices[-1]<events.index('drain')<events.index(('answer','y'))
    assert ('home_velocity', raw['hil'].get('home_velocity_scale', .30)) in events
    if server_capture:
        assert events.index('server_end') > home_indices[-1]
        assert events.index('hil_upload') > events.index(('answer','y'))
        assert events.index('hil_decide') > events.index('hil_upload')
    end=[r for r in rows if r['kind']=='episode_end']
    assert len(end)==1 and end[0]['reason']=='r' and end[0]['homing_recorded'] is False
    assert end[0]['last_tick']==max(r['tick'] for r in rows if r['kind']=='tick')
    end_index=next(i for i,r in enumerate(rows) if r['kind']=='episode_end')
    assert not any(r['kind']=='tick' for r in rows[end_index+1:])
    assert not any(isinstance(e,tuple) and e[0]=='act' for e in events[home_indices[-1]:])



def test_fresh_rtc_hold_does_not_skip_and_blends_over_configured_steps():
    arb = Arbiter(smooth_steps=8, rtc_enabled=True)
    arb.switch('policy')
    generation = arb.generation
    arb.track(STATE)
    for _ in range(12):
        arb.advance(None)
    assert arb.policy_steps == 0
    assert arb.rtc_prefix(4) == []
    target = list(STATE); target[0] += .008
    assert arb.accept({'actions': [target]*50}, generation, 30, arb.policy_steps)
    assert arb.actions[0]['chunk_index'] == 0
    assert [r['action'][0] for r in list(arb.actions)[:9]] == pytest.approx(
        [.4+i*.001 for i in range(9)])
    arb.advance(arb.actions.popleft())
    assert arb.policy_steps == 1
    prefix = arb.rtc_prefix(4)
    assert prefix == [r['action'] for r in list(arb.actions)[:4]]
    arb.switch('human')
    arb.track(target)  # shadow policy is not the actual held command
    arb.switch('policy')
    arb.track(STATE)
    for _ in range(12): arb.advance(None)
    assert arb.policy_steps == 1
    assert not arb.accept({'actions':[target]*50}, generation, 30)
    assert arb.awaiting_first_policy
