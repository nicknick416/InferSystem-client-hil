"""CSV lifecycle, frame associations and fail-closed encoding, without robot I/O."""
import copy
import importlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from HIL import video_storage as vs


class FakeStream:
    def __init__(self, directory, shape, fps, options):
        self.directory, self.shape, self.frames = directory, shape, 0
        directory.mkdir(parents=True)
    def write(self, image, digest, tick, sensor_time):
        index = self.frames
        self.frames += 1
        return index
    def close(self): pass


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(vs, 'VideoStream', FakeStream)
    monkeypatch.setattr(vs, 'find_ffmpeg', lambda value=None: '/unused/ffmpeg')


@pytest.fixture
def images():
    a = np.random.default_rng(123).integers(0,256,(64,64,3),dtype=np.uint8)
    return [cv2.imencode('.png',a)[1].tobytes(), cv2.imencode('.jpg',a)[1].tobytes()]


def metadata():
    return dict(fps=30, horizon=50, outcome='success', config={'hil':{'recording':{'format':'collector_h265','qp':30}}})


def tick(i, image):
    state = [.4,.1,.3,1.,0.,0.,0.,1.,0.,.02]+[0.]*22
    action = state.copy(); action[9] = .5
    return dict(kind='tick', tick=i, mode='human', segment=1, state=state, state_time=10+i/30,
                publish_start=10+i/30, publish_time=10+i/30+.001,
                policy_action=action, human_action=action, published_action=action,
                pair_valid=True, policy_source={'request_id':1,'chunk_index':i},
                policy_observation_age_s=.1, published_pose=[.4,.1,.3,1.,0.,0.,0.], commanded_gripper_m=.04,
                robot_state={'joint_positions':[.1]*6,'eef_pose':[.4,.1,.3,1.,0.,0.,0.]},
                observation={'observation/left_tactile':image,'extra':{'tactile_history':{'observation/left_tactile':[image]},
                           'tactile_baseline':{'observation/left_tactile':image}}},
                sensor_times={'observation/left_tactile':9+i/30})


def test_png_and_jpeg_histories_and_baselines_resolve_to_explicit_frames(tmp_path, images, fake_encoder):
    writer = vs.CollectorWriter(tmp_path,30)
    a, b = tick(0,images[0]), tick(1,images[1])
    b['observation']['extra']['tactile_history']['observation/left_tactile']=[images[0],images[1]]
    b['observation']['extra']['tactile_baseline']['observation/left_tactile']=images[0]
    writer.add(a);writer.add(b);writer.close()
    rows=list(vs.iter_records(tmp_path))
    first = rows[0]['observation']['observation/left_tactile']
    current = rows[1]['observation']['observation/left_tactile']
    history = rows[1]['observation']['extra']['tactile_history']['observation/left_tactile']
    assert history == [first,current]
    assert first['frame']==0 and current['frame']==1
    assert sum(s.frames for s in writer.archive.streams.values())==3
    assert rows[0]['published_action']==a['published_action']
    assert rows[1]['sensor_times']==b['sensor_times']


def test_repeated_current_frame_keeps_one_frame_per_control_tick(tmp_path, images, fake_encoder):
    writer=vs.CollectorWriter(tmp_path,30)
    writer.add(tick(0,images[0]));writer.add(tick(1,images[0]));writer.close()
    rows=list(vs.iter_records(tmp_path))
    assert rows[1]['observation']['observation/left_tactile']['frame']==1
    assert rows[1]['observation']['extra']['tactile_history']['observation/left_tactile'][0]['frame']==1


def test_unseen_history_does_not_add_frames_to_current_video(tmp_path, images, fake_encoder):
    writer=vs.CollectorWriter(tmp_path,30)
    row=tick(0,images[0])
    row['observation']['extra']['tactile_history']['observation/left_tactile']=[images[1],images[0]]
    writer.add(row);writer.close()
    saved=list(vs.iter_records(tmp_path))[0]['observation']
    current=saved['observation/left_tactile']
    history=saved['extra']['tactile_history']['observation/left_tactile']
    assert history[0]['stream'].startswith('history.')
    assert history[1]==current
    assert writer.archive.streams[current['stream']].frames==1


@pytest.mark.parametrize('package',['HIL','HIL_PI'])
def test_server_commit_reads_new_metadata_and_tables(tmp_path, images, fake_encoder, package):
    rec=importlib.import_module(package+'.recording')
    server=importlib.import_module(package+'.server_recording')
    ep=rec.Episode(tmp_path,metadata())
    row=tick(0,images[0]);ep.add(row.pop('kind'),**row);ep.add('episode_end',last_tick=0)
    rec.atomic_json(ep.path/'server_transaction.json',dict(episode_id=ep.episode_id,token='test'))
    calls=[]
    class Worker:
        def control(self,payload):calls.append(payload);return {'status':'ok'}
    ep.decide(True,sync=lambda path:server.sync_decision(path,Worker()))
    assert [c['cmd'] for c in calls]==['hil_end','hil_upload','hil_decide']
    assert calls[1]['rows'][0]['published_action']==row['published_action']
    assert calls[-1]['keep'] is True


def test_optional_fields_and_event_order_round_trip_exactly(tmp_path, fake_encoder):
    rows=[{'kind':'tick','tick':0,'optional':17}, {'kind':'hil_transition','event':'human_started'},
          {'kind':'tick','tick':1,'new':False}, {'kind':'episode_end','last_tick':1}]
    writer=vs.CollectorWriter(tmp_path,30)
    for row in rows:writer.add(row)
    writer.close()
    assert list(vs.iter_records(tmp_path))==rows


@pytest.mark.parametrize('package',['HIL','HIL_PI'])
def test_new_episode_lifecycle_and_pairing_without_legacy_files(tmp_path, images, fake_encoder, package):
    rec=importlib.import_module(package+'.recording')
    ep=rec.Episode(tmp_path,metadata())
    for i in range(3):
        row=tick(i,images[i%2]);ep.add(row.pop('kind'),**row)
    ep.add('episode_end',last_tick=2)
    result=ep.decide(True)
    assert rec.read_metadata(result)['storage_format']==vs.FORMAT
    assert rec.read_metadata(result)['paired_steps']==3
    assert rec.read_metadata(result)['total_frames']==3
    assert len(list(rec.compact_records(result)))==4
    assert len((result/'hil/pairs.jsonl').read_text().splitlines())==3
    assert not list(result.glob('*.msgpack')) and not (result/'blobs').exists()
    assert not (result/'manifest.json').exists()
    assert (result/'actions.eef_pose/data.csv').exists()
    assert (result/'observation.state.joint_position/data.csv').exists()


@pytest.mark.parametrize('package',['HIL','HIL_PI'])
def test_new_storage_label_prompt_and_discard(tmp_path, fake_encoder, package):
    rec=importlib.import_module(package+'.recording')
    choose=importlib.import_module(package+'.outcome').choose_episode_decision
    ep=rec.Episode(tmp_path,metadata());ep.add('tick',tick=0);ep.add('episode_end',last_tick=0)
    class Keyboard:
        def wait(self,*args):return 'f'
    assert choose(ep,Keyboard()) is True
    assert rec.read_metadata(ep.path)['outcome']=='failure'
    ep.decide(False)
    assert not ep.path.exists()


def test_encoder_failure_cannot_publish_kept_episode(tmp_path, images, fake_encoder, monkeypatch):
    from HIL.recording import Episode
    def fail(self):raise RuntimeError('encoder failed')
    monkeypatch.setattr(FakeStream,'close',fail)
    ep=Episode(tmp_path,metadata())
    row=tick(0,images[0]);ep.add(row.pop('kind'),**row)
    with pytest.raises(RuntimeError,match='incomplete'):
        ep.decide(True)
    assert ep.path.name.startswith('.pending_')
    assert json.loads((ep.path/'metadata.json').read_text())['status']=='pending'


def test_missing_observation_row_is_rejected(tmp_path, fake_encoder):
    writer=vs.CollectorWriter(tmp_path,30);writer.add({'kind':'tick','tick':0});writer.close()
    (tmp_path/'hil/observations.jsonl').write_text('')
    with pytest.raises(ValueError,match='Incomplete'):list(vs.iter_records(tmp_path))


def test_compact_reader_and_replay_do_not_decode_videos(tmp_path, images, fake_encoder, monkeypatch):
    from HIL.recording import Episode
    from HIL.ur_replay import load_episode
    m=metadata();m['config']['robot']={'type':'ur','gripper':{'max_width':.085}}
    ep=Episode(tmp_path,m)
    for i in range(2):
        row=tick(i,images[0]);ep.add(row.pop('kind'),**row)
    path=ep.decide(True)
    monkeypatch.setattr(vs,'VideoDecoder',lambda *_:pytest.fail('Replay decoded video'))
    result=load_episode(path)
    assert len(result.times)==2 and result.widths[0]==.04
    assert load_episode(path/'hil/ticks.csv',source='measured').widths[0]==.02


def test_video_reference_cannot_escape_episode(tmp_path):
    decoder=vs.VideoDecoder(tmp_path)
    with pytest.raises(ValueError,match='Invalid video reference'):
        decoder.decode({vs.VIDEO_KEY:1,'stream':'../escape','frame':0})


@pytest.mark.skipif(os.environ.get('HIL_TEST_VAAPI') != '1', reason='Set HIL_TEST_VAAPI=1 for encoder-only hardware test')
def test_real_vaapi_encode_decode_rgb_and_tactile(tmp_path, images):
    # Only an encoder subprocess is opened. No cameras, robot, or network.
    vs.probe()
    # This machine's VAAPI encoder requires at least 256x128.
    images = [cv2.imencode(ext, cv2.resize(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR),
                                        (256,128)))[1].tobytes() for ext,data in zip(['.png','.jpg'],images)]
    writer=vs.CollectorWriter(tmp_path,30)
    for i in range(4):writer.add(tick(i,images[i%2]))
    writer.close()
    decoder=vs.VideoDecoder(tmp_path)
    try:
        rows=list(vs.iter_records(tmp_path))
        for row in rows:
            image=decoder.decode(row['observation']['observation/left_tactile'])
            decoded=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR)
            assert decoded.shape==(128,256,3)
        # Backwards access beyond the current position returns the requested frame.
        decoder.cache.clear()
        assert decoder.decode(rows[0]['observation']['observation/left_tactile'])
    finally:decoder.close()


def test_recording_omits_only_settled_holds_and_preserves_alignment(tmp_path, images, fake_encoder):
    from HIL.recording import Episode, read_metadata
    from HIL.outcome import choose_episode_decision
    m = metadata()
    m['config']['hil']['omit_settled_human_frames'] = True
    ep = Episode(tmp_path, m)
    for i in range(100):
        row = tick(i, images[i % 2]); row.pop('kind')
        row.update(time=row['publish_start'], spacemouse_axes=[0.] * 6, spacemouse_buttons=0)
        if i < 5 or i == 38 or i >= 54 and i < 62:
            row['spacemouse_axes'][0] = .5
        if 39 <= i < 54:
            row.update(mode='policy', human_action=None, pair_valid=False)
        if i >= 62:
            row['spacemouse_buttons'] = int(i == 62)
        ep.add('tick', **row)
        if i == 35:
            ep.add('prediction', tick=25, request_id=3)
    ep.add('episode_end', last_tick=99)
    class Keys:
        def wait(self, *args): return 's'
    assert choose_episode_decision(ep, Keys()) is True
    path = ep.decide(True)
    rows = list(vs.iter_records(path)); saved = [v for v in rows if v['kind'] == 'tick']
    meta = read_metadata(path)
    ticks = [v['tick'] for v in saved]
    assert set(range(5)) | set(range(37, 63)) | {99} <= set(ticks)
    assert not (set(range(19, 37)) | set(range(76, 99))) & set(ticks)
    assert meta['source_tick_count'] == 100 and meta['terminal_last_tick'] == 99
    assert meta['total_frames'] == len(ticks) == meta['human_ticks'] + 15
    assert meta['omitted_human_idle_frames'] == 100-len(ticks)
    assert {k for g in meta['omitted_human_idle_ranges'] for k in range(g['first_tick'],g['last_tick']+1)} == set(range(100))-set(ticks)
    assert [v['frame_index'] for v in saved] == list(range(len(ticks)))
    assert all(v['observation']['observation/left_tactile']['frame'] == j for j,v in enumerate(saved))
    pairs = [json.loads(l) for l in (path/'hil/pairs.jsonl').read_text().splitlines()]
    assert all(ticks[v['observation_ref']['frame_index']] == v['tick'] for v in pairs)
    assert next(v for v in rows if v['kind']=='prediction')['tick'] == 25
    import csv
    with (path/'actions.eef_pose/data.csv').open() as f:
        assert [int(v['tick']) for v in csv.DictReader(f)] == ticks


@pytest.mark.parametrize('motion', ['gripper','xyz','joints','rotation','command','input','missing','stale'])
def test_recording_keeps_motion_or_uncertain_feedback(motion, images):
    from HIL.human_idle_recording import SettledHumanFrames
    class Sink:
        tick_count = 0
        def add(self, row): self.tick_count += int(row['kind']=='tick')
    sink = Sink(); gate = SettledHumanFrames(sink, 30)
    for i in range(55):
        row = tick(i, images[0]); row.update(time=row['publish_start'], spacemouse_axes=[0.] * 6,
                                            spacemouse_buttons=int(i == 0))
        if motion == 'gripper': row['state'][9] += i*.0002
        if motion == 'xyz': row['state'][0] += i*.0001
        if motion == 'joints': row['robot_state']['joint_positions'][2] += i*.001
        if motion == 'rotation': row['state'][3:9] = [np.cos(i*.004), np.sin(i*.004),0,-np.sin(i*.004),np.cos(i*.004),0]
        if motion == 'command': row['published_action'][9] += i*.0001
        if motion == 'input': row['spacemouse_axes'][1] = .5
        if motion == 'missing': row['state'] = None
        if motion == 'stale': row['state_time'] = 10.
        gate.add(row)
    assert gate.finish()['omitted_human_idle_frames'] == 0
    assert sink.tick_count == 55
