"""Keyboard HIL uses executed commands, measured anchors and explicit transitions."""
import json
import sys
import threading
import types
from copy import deepcopy

import msgpack
import numpy as np
import pytest
import yaml

from Core import ArmState
from HIL_PI import runtime
from HIL_PI.keyboard_hil import KeyboardHIL
from HIL_PI.recording import Episode, compact_records, records
from HIL_PI.runtime import check_config, ROOT


def state(x=0.0, width=0.1):
    return [x, 0., .3, 1., 0., 0., 0., 1., 0., width] + [0.] * 22


def published(x=0.0, width=0.1):
    row = state(x, width)
    row[9] = width * 10.
    return row


def tracker():
    return KeyboardHIL({'enabled': True}, {'max_sample_gap_s': .2}, 30.)


def execute(k, t, x, width, request, previous_width, measured_width=None):
    measured = width if measured_width is None else measured_width
    k.feedback(t, state(x, measured))
    k.executed(t, state(x, measured), published(x, width),
               dict(request_id=request, chunk_index=0), 1, previous_width, 10.)


def test_closing_priority_opens_before_pose_retreat_and_requires_second_key():
    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    execute(k, .03, .005, .1, 2, .1)
    execute(k, .06, .005, .096, 2, .1, measured_width=.1)
    result = k.trigger(.07, state(.006, .04), published(.006, .096), gripper_scale=10.)
    assert result['branch'] == 'gripper_close'
    assert result['anchor_state'][0] == pytest.approx(.005)
    assert k.phase == 'opening'
    command, event = k.step(.07, state(.006, .04), .04)
    assert command[0] == pytest.approx(.006) and command[9] == pytest.approx(1.)
    assert event is None and not k.request_human()
    for t in (.10, .13, .16):
        command, _ = k.step(t, state(.006, .1), .1)
    assert k.phase == 'retreating' and command[9] == pytest.approx(1.)
    events = []
    for t in (.19, .22, .25, .28, .31):
        command, event = k.step(t, state(.005, .1), .1)
        events.append(event)
    assert events.count('rollback_finished') == 1 and k.phase == 'await_human'
    assert k.request_human() and k.phase == 'await_neutral'
    assert not k.neutral(np.array([.5, 0, 0, 0, 0, 0]), 0)
    assert k.neutral(np.zeros(6), 0) and k.phase == 'human'


def test_position_branch_uses_predecessor_start_and_missing_history_holds():
    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    execute(k, .03, .001, .1, 2, .1)
    k.feedback(.06, state(.003, .1))
    result = k.trigger(.06, state(.003, .1), published(.003, .1), gripper_scale=10.)
    assert result['branch'] == 'position_move'
    assert result['anchor_state'][0] == pytest.approx(0.)
    assert k.phase == 'retreating'
    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    result = k.trigger(.03, state(0., .1), published(0., .1), gripper_scale=10.)
    assert result['reason'] == 'no_classifiable_motion'
    assert k.phase == 'await_human'


def test_rollback_uses_recent_classifiable_chunk_after_short_new_chunk():
    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    execute(k, .03, .005, .1, 2, .1)
    execute(k, .06, .005, .09, 2, .1)
    execute(k, .09, .005, .09, 3, .09)
    result = k.trigger(.10, state(.005, .09), published(.005, .09), gripper_scale=10.)
    assert result['branch'] == 'gripper_close'
    assert result['source']['request_id'] == 2
    assert result['latest_source']['request_id'] == 3
    assert result['skipped_chunks'] == 1
    assert result['anchor_state'][0] == pytest.approx(.005)

    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    execute(k, .03, .001, .1, 2, .1)
    k.feedback(.06, state(.003, .1))
    execute(k, .09, .003, .1, 3, .1)
    result = k.trigger(.10, state(.0031, .1), published(.0031, .1), gripper_scale=10.)
    assert result['branch'] == 'position_move'
    assert result['source']['request_id'] == 2
    assert result['latest_source']['request_id'] == 3
    assert result['skipped_chunks'] == 1
    assert result['anchor_state'][0] == pytest.approx(0.)


def test_open_timeout_holds_and_must_be_explicitly_accepted():
    k = tracker()
    execute(k, 0., 0., .1, 1, .1)
    execute(k, .03, .001, .096, 1, .1, measured_width=.1)
    k.trigger(.04, state(.002, .04), published(.002, .096), gripper_scale=10.)
    command, event = k.step(3.1, state(.002, .04), .04)
    assert event == 'rollback_failed' and k.reason == 'open_timeout'
    assert command[0] == pytest.approx(.002) and k.phase == 'await_human'
    assert k.request_human()


def test_keyboard_config_uses_rtc_and_no_automatic_recovery():
    cfg, raw = check_config(ROOT / 'Config/flexiv_pi_cylinder_rtc_keyboard_hil.yaml')
    assert raw['inference']['rtc'] == {'enabled': True, 'prefix_length': 4}
    assert raw['hil']['output_dir'] == 'data'
    assert not raw['hil']['keyboard_hil']['enabled']
    assert not raw['hil']['stuck_detection']['enabled']
    assert not raw['hil']['stuck_recovery']['enabled']
    assert not raw['hil']['correction']['enabled']


def test_keyboard_config_accepts_optional_rtc_but_rejects_automatic_recovery(tmp_path):
    _, raw = check_config(ROOT / 'Config/flexiv_pi_cylinder_rtc_keyboard_hil.yaml')
    path = tmp_path / 'keyboard.yaml'
    raw['inference']['rtc']['enabled'] = False
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='requires inference.rtc.enabled'):
        check_config(path)
    raw['inference']['rtc']['enabled'] = True
    raw['hil']['keyboard_hil']['enabled'] = True
    raw['hil']['stuck_detection']['enabled'] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='keyboard_hil requires'):
        check_config(path)


def test_episode_indexes_transitions_and_marks_manifest(tmp_path):
    ep = Episode(tmp_path, {'hil_variant': 'keyboard_two_press_hil_v1', 'fps': 30, 'horizon': 50})
    ep.add('tick', tick=0, mode='policy', hil_id=None, hil_phase='none')
    ep.add('hil_transition', event='rollback_requested', tick=1, hil_id=1)
    ep.add('tick', tick=1, mode='recovery', hil_id=1, hil_phase='opening')
    ep.add('hil_transition', event='human_started', tick=2, hil_id=1)
    ep.add('tick', tick=2, mode='human', hil_id=1, hil_phase='human', pair_valid=False)
    ep.add('episode_end', last_tick=2)
    ep.seal()
    manifest = json.loads((ep.path / 'manifest.json').read_text())
    assert (manifest['hil_attempts'], manifest['hil_entries']) == (1, 1)
    indexed = list(compact_records(ep.path))
    assert [r['event'] for r in indexed if r['kind'] == 'hil_transition'] == ['rollback_requested', 'human_started']
    assert [r['hil_phase'] for r in indexed if r['kind'] == 'tick'] == ['none', 'opening', 'human']
    assert len(list(records(ep.path))) == len(indexed)


def test_kept_keyboard_pair_carries_hil_identity(tmp_path):
    ep = Episode(tmp_path, {'hil_variant': 'keyboard_two_press_hil_v1', 'fps': 30, 'horizon': 50})
    ep.add('hil_transition', event='rollback_requested', tick=0, hil_id=1)
    ep.add('hil_transition', event='human_started', tick=1, hil_id=1)
    ep.add('tick', tick=1, mode='human', segment=1, hil_id=1, hil_phase='human',
           state=state(), state_time=.03, publish_start=.03, publish_time=.031,
           human_action=published(), policy_action=published(), published_action=published(),
           policy_source={'request_id': 2, 'chunk_index': 0, 'observation_time': .02},
           policy_observation_age_s=.01, pair_valid=True)
    ep.add('episode_end', last_tick=1)
    ep.metadata['outcome'] = 'success'
    path = ep.decide(True)
    with (path / 'pairs.msgpack').open('rb') as stream:
        pair = next(msgpack.Unpacker(stream, raw=False))
    assert pair['hil_id'] == 1
    assert pair['hil_variant'] == 'keyboard_two_press_hil_v1'
    assert pair['preference_label'] == 'operator_kept_human_correction'


@pytest.mark.parametrize('closing', [False, True])
@pytest.mark.parametrize('resume_key', [' ', 'p'])
@pytest.mark.parametrize('rtc_enabled', [True])
def test_live_loop_requires_both_h_presses_and_records_intervening_ticks(monkeypatch, tmp_path, closing, resume_key, rtc_enabled):
    cfg, raw = check_config(ROOT / 'Config/flexiv_pi_cylinder_rtc_keyboard_hil.yaml')
    # Exercise the copied optional flow only with fake hardware, while deployed
    # PI configs leave all rollback paths disabled.
    raw['hil']['keyboard_hil']['enabled'] = True
    raw['hil']['output_dir'] = str(tmp_path)
    raw['inference']['rtc']['enabled'] = rtc_enabled
    raw['inference']['n_execute'] = 1
    raw['inference']['smooth'] = {'enabled': False}
    rows, requests, clock = [], [], [0.]
    first_h = [None]
    resumed = [False]

    def mono():
        clock[0] += .00001
        return clock[0]

    monkeypatch.setattr(runtime.time, 'monotonic', mono)
    monkeypatch.setattr(runtime.time, 'sleep', lambda duration: clock.__setitem__(0, clock[0] + duration))

    def ticks():
        return [row for row in rows if row['kind'] == 'tick']

    class Keyboard:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, *args): return '\n'
        def read(self):
            if len(ticks()) > 100:
                raise AssertionError('Keyboard HIL did not finish')
            if resumed[0]:
                after_resume = next((r['tick'] for r in rows if r.get('event') == 'policy_resumed'), None)
                if after_resume is not None and len([r for r in ticks() if r['mode'] == 'policy' and r['tick'] >= after_resume]) >= 5:
                    return 'q'
            elif len([r for r in ticks() if r['mode'] == 'human']) >= 2:
                resumed[0] = True
                return resume_key
            ready = (len(ticks()) >= 3 and
                     (not closing or ticks()[-1].get('commanded_gripper_m', .1) < .099))
            if first_h[0] is None and ready:
                first_h[0] = clock[0]
                return 'h'
            if first_h[0] is not None and clock[0] - first_h[0] > .7:
                if not any(r.get('event') == 'human_requested' for r in rows):
                    return 'h'
            return ''

    class Mouse:
        def __init__(self, *args): self.activity = threading.Event()
        def open(self): pass
        def close(self): pass
        def sample(self):
            if any(r.get('event') == 'human_requested' for r in rows):
                return np.zeros(6), 0
            self.activity.set()
            return np.array([.5, 0, 0, 0, 0, 0]), 0

    class Grip:
        def connect(self): pass
        def disconnect(self): pass
        def observe(self): return types.SimpleNamespace(width=.1, max_width=.1)
        def move(self, *args, **kwargs): pass

    class Robot:
        dof = 7
        @staticmethod
        def from_config(path): return robot
        def connect(self): pass
        def disconnect(self): pass
        def is_connected(self): return True
        def is_fault(self): return False
        def enable(self): pass
        def wait_until_operational(self, **kwargs): return True
        def create_gripper(self): return Grip()
        def stop(self): pass
        def move_joint_position(self, *args, **kwargs): return True
        def observe(self): return ArmState(timestamp=clock[0], eef_pose=[.4, 0, .3, 1, 0, 0, 0])
        def act(self, action, state=None): pass

    robot = Robot()

    class Sensors:
        @staticmethod
        def from_config(*args, **kwargs): return Sensors()
        def open_all(self): pass
        def close_all(self): pass

    class Camera:
        def __init__(self, *args): pass
        def start(self): pass
        def close(self): pass
        def snapshot(self, state, *args, **kwargs): return {'cmd': 'predict', 'state': state}, {}

    class Ring:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def clear(self): pass
        def wait_for_frames(self, *args): return True

    class Capture:
        def __init__(self, *args): self.metadata = {}; self.path = tmp_path; self.episode_id = 'fake'
        def add(self, kind, **kwargs): rows.append(dict(kind=kind, **kwargs))
        def seal(self): pass
        def decide(self, *args): return tmp_path

    class Worker:
        hil_session = None
        def __init__(self, *args): self.pending = None; self.request_id = 0
        @property
        def busy(self): return self.pending is not None
        def reset(self): pass
        def submit(self, payload, **kwargs):
            assert payload['cmd'] == ('predict_rtc' if rtc_enabled else 'predict')
            self.request_id += 1
            requests.append((len(ticks()), kwargs['generation'], deepcopy(payload)))
            self.pending = dict(kwargs, request_id=self.request_id, request_time=clock[0])
            return self.request_id
        def poll(self):
            if self.pending is None: return None
            context, self.pending = self.pending, None
            width = .095 if closing and context['request_id'] >= 3 else .1
            return dict(context, response_time=clock[0], response={'actions': [published(.4, width)] * 50})
        def drain(self): return self.poll()
        def close(self): pass

    for name, attrs in [('Robot', dict(BaseRobot=Robot)),
                        ('Sensor.manager', dict(SensorManager=Sensors)),
                        ('Sensor.tactile.ring_buffer', dict(TactileRingBuffer=Ring)),
                        ('HIL_PI.observations', dict(CameraWorker=Camera)),
                        ('HIL_PI.outcome', dict(choose_episode_decision=lambda *args: False))]:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    for name, cls in [('Keyboard', Keyboard), ('SpaceMouse', Mouse), ('Episode', Capture), ('PolicyWorker', Worker)]:
        monkeypatch.setattr(runtime, name, cls)

    runtime.run_live(ROOT / 'Config/flexiv_pi_cylinder_rtc_keyboard_hil.yaml', cfg, raw, 'test')
    assert all(len(req['state']) == 10 for _, _, req in requests)
    assert [r['event'] for r in rows if r['kind'] == 'hil_transition'] == [
        'rollback_requested', 'rollback_finished' if closing else 'rollback_failed',
        'human_requested', 'human_started', 'policy_resumed']
    assert next(r for r in rows if r.get('event') == 'policy_resumed')['key'] == resume_key
    assert next(r for r in rows if r.get('event') == 'key_received')['key'] == resume_key
    phases = [r['hil_phase'] for r in ticks()]
    assert phases[:3] == ['none'] * 3
    if closing:
        assert 'opening' in phases
        assert next(r for r in rows if r.get('event') == 'rollback_requested')['branch'] == 'gripper_close'
    assert 'await_human' in phases and 'human' in phases
    assert all(r['request_id'] is None and not r['pair_valid'] for r in ticks() if r['hil_phase'] == 'await_human')
    assert all(r['mode'] == 'policy' for r in ticks()[:3])
    assert ticks()[-1]['mode'] == 'policy' and ticks()[-1]['hil_phase'] == 'none'
    if rtc_enabled:
        assert any(req[2]['rtc']['prefix_length'] > 0 for req in requests if req[0] < 3)
        human_ticks = {row['tick'] for row in ticks() if row['mode'] == 'human'}
        assert all(req[2]['rtc']['prefix_length'] == 0 for req in requests if req[0] in human_ticks)
        resume_tick = next(row['tick'] for row in rows if row.get('event') == 'policy_resumed')
        resumed = [req for req in requests if req[0] >= resume_tick]
        assert resumed and resumed[0][2]['rtc']['prefix_length'] == 0
        assert resumed[0][1] > requests[0][1]
