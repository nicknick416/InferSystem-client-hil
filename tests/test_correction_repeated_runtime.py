"""Exercise the real HIL loop with fake hardware/network, including mode transitions."""
import sys
import threading
import types
from pathlib import Path
import numpy as np
import pytest
from Core import ArmState
from HIL import runtime


@pytest.mark.parametrize('already_at_target', [False, True, 'history_expired', 'no_valid_history'])
def test_first_auto_then_second_manual_same_episode(monkeypatch, tmp_path, already_at_target):
    automatic, late_takeover, manual = True, False, False
    second_requested = [False]
    OriginalRecovery=runtime.StuckRecovery
    class Recovery(OriginalRecovery):
        def start(self,stamp,state,published):
            if already_at_target and second_requested[0]:
                if already_at_target == 'history_expired':
                    self.checkpoint=(stamp-self.config['history_s']-1,np.array(state))
                elif already_at_target == 'no_valid_history':
                    self.checkpoint=None
                else:
                    self.checkpoint=(stamp,np.array(state))
            return super().start(stamp,state,published)
    monkeypatch.setattr(runtime,'StuckRecovery',Recovery)
    cfg, raw = runtime.check_config(Path(runtime.ROOT)/'Config/flexiv_vtla_expo_hil.yaml')
    raw['hil']['output_dir'] = str(tmp_path)
    raw['hil']['server_capture'] = False
    raw['hil']['stuck_detection']['enabled'] = True
    raw['hil']['stuck_recovery']['enabled'] = True
    raw['hil']['correction'] = dict(enabled=True,auto_execute=automatic)
    raw['hil']['manual_recovery_enabled'] = True
    manual_sent = [False]
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
            if manual and len(ts)>=40 and not manual_sent[0]:
                manual_sent[0]=True
                return 'b'
            if automatic:
                completed=any(r['kind']=='correction_execution' and r.get('event')=='completed' for r in data)
                if completed:
                    end = next(r['tick'] for r in data if r['kind']=='correction_execution' and r.get('event')=='completed')
                    if not second_requested[0] and len(ts)>end+40:
                        second_requested[0]=True
                        return 'b'
                    if sum(r['mode']=='human' for r in ts)>=3 and not resumed[0]:
                        resumed[0]=True
                        return 'p'
                    if resumed[0] and any(r['kind']=='policy_resume' and r.get('event')=='request_submitted' for r in data):
                        return 'q'
            else:
                human_ticks=sum(r['mode']=='human' for r in ts)
                if human_ticks>=3 and not resumed[0]:resumed[0]=True;return 'p'
                if resumed[0] and any(r['kind']=='policy_resume' and r.get('event')=='request_submitted' for r in data):return 'q'
            return ''

    mice=[]
    class Mouse:
        def __init__(self,*a):self.activity=threading.Event();mice.append(self)
        def open(self):pass
        def close(self):pass
        def sample(self):
            if second_requested[0] and not resumed[0] and (sum(r['kind']=='correction_anchor' for r in data)>=2 or
                    (already_at_target in ('history_expired','no_valid_history') and
                     any(r['kind']=='stuck_recovery' and r.get('trigger')=='manual_b' for r in data))):self.activity.set()
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
            if late_takeover and any(r['kind']=='correction_execution' and r.get('event')=='completed' for r in data):
                mice[0].activity.set()
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
        def snapshot(self,state,*a,**kw):return {'state':state,'observation/wrist_image':b'fake_png'},{}
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
    class Bridge:
        def __init__(self,*a):self.anchor=None;self.pending=False;self.labels=[];self.human_seen=False;bridges.append(self)
        def new_episode(self):self.clear()
        def clear(self):self.anchor=None;self.pending=False
        def capture(self,state,payload,stamp,sensor_times,published,episode_id,tick):
            self.anchor=dict(sample_id='cycle',anchor_pose=list(state[:9]),gripper_command=published[9],tick=tick)
            self.human_seen=False
            return 'cycle'
        def human(self):self.human_seen=True;self.pending=False
        def resume(self,state,payload,stamp,sensor_times,tick):
            assert self.human_seen and self.anchor is not None
            self.labels.append(dict(anchor=dict(self.anchor),target=list(state[:9]),tick=tick))
            self.clear()
            return dict(sample_id='cycle',status='queued')
        def predict(self):self.pending=automatic and not second_requested[0]
        def poll(self,state):
            if not self.pending:return None
            self.pending=False
            target=list(self.anchor['anchor_pose']);target[0]-=.003
            return dict(status='ok',selected=0,poses=[target],scores=[-.01],steps=300)
        def close(self):pass
    bridges=[]
    monkeypatch.setattr(runtime,'CorrectionBridge',Bridge)
    def fake_module(name,**attrs):monkeypatch.setitem(sys.modules,name,types.SimpleNamespace(**attrs))
    fake_module('Robot',BaseRobot=Robot)
    fake_module('Sensor.manager',SensorManager=Sensors)
    fake_module('Sensor.tactile.ring_buffer',TactileRingBuffer=Ring)
    fake_module('HIL.observations',CameraWorker=Camera)
    fake_module('HIL.outcome',choose_episode_decision=lambda *a:'failure')
    for name, cls in [('Keyboard',Keyboard),('SpaceMouse',Mouse),('Episode',Episode),('PolicyWorker',Worker)]:
        monkeypatch.setattr(runtime,name,cls)
    runtime.run_live(Path(runtime.ROOT)/'Config/flexiv_vtla_expo_hil.yaml',cfg,raw,'test')
    assert second_requested[0]
    assert len(bridges[0].labels)==1
    anchors=[r for r in data if r['kind']=='correction_anchor']
    assert len(anchors)==2
    label=bridges[0].labels[0]
    assert label['anchor']['tick']==anchors[-1]['tick']
    assert label['tick']>label['anchor']['tick']
