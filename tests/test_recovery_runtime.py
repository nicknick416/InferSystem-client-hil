"""Exercise the real HIL loop with fake hardware/network, including mode transitions."""
import sys
import threading
import types
from pathlib import Path
import numpy as np
import pytest
from Core import ArmState
from HIL import runtime


@pytest.mark.parametrize('takeover, enabled', [(False, True), (True, True), (False, False)])
def test_live_loop_recovery_with_fake_hardware(monkeypatch, tmp_path, takeover, enabled):
    cfg, raw = runtime.check_config(Path(runtime.ROOT)/'Config/flexiv_vtla_expo_hil.yaml')
    raw['hil']['output_dir'] = str(tmp_path)
    raw['hil']['server_capture'] = False
    raw['hil']['correction'] = {'enabled': False}
    raw['hil']['stuck_detection']['enabled'] = True
    raw['hil']['stuck_recovery']['enabled'] = enabled
    data = []; requests = []; controls = []; clock = [0.]; resumed = [False]
    def mono():
        clock[0] += .00001
        return clock[0]
    monkeypatch.setattr(runtime.time, 'monotonic', mono)
    monkeypatch.setattr(runtime.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    def ticks():return [r for r in data if r['kind']=='tick']
    def recovery_started():return any(r['kind']=='stuck_recovery' and r.get('event')=='started' for r in data)
    class Keyboard:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def wait(self,*args):return '\n'
        def read(self):
            ts=ticks()
            if len(ts)>240:raise AssertionError('Recovery did not reach expected state')
            if not enabled:
                return 'q' if sum(r.get('stuck_detection',{}).get('stuck',False) for r in ts) >= 5 else ''
            if takeover:
                return 'q' if any(r['mode']=='human' for r in ts) else ''
            held=[r for r in ts if r['mode']=='recovery_hold']
            if len(held)>=3 and not resumed[0]:
                resumed[0]=True;return 'p'
            if resumed[0] and ts[-1]['mode']=='policy' and ts[-1]['request_id'] is not None:return 'q'
            return ''
    class Mouse:
        def __init__(self,*a):self.activity=threading.Event()
        def open(self):pass
        def close(self):pass
        def sample(self):
            if takeover and recovery_started():self.activity.set()
            return np.zeros(6),0
    class Grip:
        def connect(self):pass
        def disconnect(self):pass
        def observe(self):return types.SimpleNamespace(width=.04,max_width=.1)
        def move(self,*a,**kw):pass
    class Robot:
        dof=7
        def __init__(self):self.x=.4
        @staticmethod
        def from_config(path):return robot
        def connect(self):pass
        def disconnect(self):pass
        def is_connected(self):return True
        def is_fault(self):return False
        def enable(self):pass
        def wait_until_operational(self,**kw):return True
        def create_gripper(self):return Grip()
        def stop(self):pass
        def move_joint_position(self,*a,**kw):return True
        def observe(self):return ArmState(timestamp=clock[0],eef_pose=[self.x,0,.3,1,0,0,0])
        def act(self,action,state=None):
            target=action.values[0]
            self.x = min(.412,self.x+.0002) if target>self.x else target
    robot=Robot()
    class Sensors:
        @staticmethod
        def from_config(*a,**kw):return Sensors()
        def open_all(self):pass
        def close_all(self):pass
    class Camera:
        def __init__(self,*a):pass
        def start(self):pass
        def close(self):pass
        def snapshot(self,state,*a,**kw):return {'state':state},{}
    class Ring:
        def __init__(self,*a,**kw):pass
        def start(self):pass
        def stop(self):pass
        def clear(self):pass
        def wait_for_frames(self,*a):return True
    class Episode:
        def __init__(self,*a):self.metadata={};self.path=tmp_path;self.episode_id='fake'
        def add(self,kind,**kw):data.append(dict(kind=kind,**kw))
        def seal(self):pass
        def decide(self,*a):return tmp_path
    class Worker:
        hil_session=None
        def __init__(self,*a):self.pending=None;self.request_id=0
        @property
        def busy(self):return self.pending is not None
        def reset(self):pass
        def submit(self,payload,**kw):
            self.request_id+=1
            requests.append((len(ticks()), kw['generation']))
            self.pending=dict(kw,request_id=self.request_id,request_time=clock[0])
            return self.request_id
        def poll(self):
            if self.pending is None:return None
            context=self.pending;self.pending=None
            action=[.42,0,.3,1,0,0,0,1,0,.4]+[0.]*22
            return dict(context,response_time=clock[0],response={'actions':[action]*50})
        def drain(self):return self.poll()
        def close(self):pass
    def fake_module(name,**attrs):monkeypatch.setitem(sys.modules,name,types.SimpleNamespace(**attrs))
    fake_module('Robot',BaseRobot=Robot)
    fake_module('Sensor.manager',SensorManager=Sensors)
    fake_module('Sensor.tactile.ring_buffer',TactileRingBuffer=Ring)
    fake_module('HIL.observations',CameraWorker=Camera)
    fake_module('HIL.outcome',choose_episode_decision=lambda *a:'failure')
    for name, cls in [('Keyboard',Keyboard),('SpaceMouse',Mouse),('Episode',Episode),('PolicyWorker',Worker)]:
        monkeypatch.setattr(runtime,name,cls)
    runtime.run_live(Path(runtime.ROOT)/'Config/flexiv_vtla_expo_hil.yaml',cfg,raw,'test')
    events=[r for r in data if r['kind']=='stuck_recovery']
    if not enabled:
        assert not events
        assert all(r['mode']=='policy' for r in ticks())
        assert ticks()[-1]['stuck_detection']['stuck']
        return
    assert events[0]['last_moving_chunk']['request_id'] > events[0]['target_source']['request_id']
    assert events[0]['event']=='started' and events[0]['status']=='retreating'
    ts=ticks()
    for r in ts:
        if r['mode'] in ('recovery','recovery_hold'):
            assert r['request_id'] is None and r['policy_action'] is None
            assert not r['pair_valid']
            assert r['published_action'][9] == pytest.approx(.4)
    if takeover:
        assert events[-1]['event']=='cancelled' and ts[-1]['mode']=='human'
    else:
        assert any(r.get('reason')=='completed' for r in events)
        assert sum(r['mode']=='recovery_hold' for r in ts)>=3
        assert ts[-1]['mode']=='policy' and ts[-1]['request_id'] is not None
