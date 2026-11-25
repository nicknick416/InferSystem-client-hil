"""Storage lifecycle checks; no devices, networking or robot motion."""
import importlib
import json
import re

import pytest


@pytest.fixture(params=['HIL', 'HIL_PI'])
def package(request):
    return request.param


def test_sessions_number_all_attempts_and_delete_only_discarded_episode(tmp_path, package):
    rec = importlib.import_module(package + '.recording')
    a, b = rec.RecordingSession(tmp_path), rec.RecordingSession(tmp_path)
    assert a.path != b.path
    assert re.fullmatch(r'\d{8}_\d{6}_\d{6}', a.path.name)
    kept = []
    for index, outcome in enumerate(['success', None, 'failure'], start=1):
        ep = rec.Episode(a.path, dict(a.next_metadata(), outcome=outcome))
        pending = ep.path
        ep.add('event', payload=b'png-placeholder' * 200)
        (pending/'extra').mkdir()
        (pending/'extra'/'sidecar.txt').write_text('must also be deleted on discard')
        path = ep.decide(outcome is not None)
        assert not pending.exists()
        if outcome is None:
            assert not path.exists()
        else:
            assert re.fullmatch(rf'\d{{8}}_\d{{6}}_\d{{6}}_episode_{index:06d}', path.name)
            manifest = json.loads((path/'manifest.json').read_text())
            assert manifest['episode_index'] == index and manifest['outcome'] == outcome
            assert manifest['session_name'] == a.path.name and manifest['status'] == 'kept'
            kept.append(path)
    assert set(a.path.iterdir()) == set(kept)
    assert b.next_metadata()['episode_index'] == 1
    assert b.path.exists()


@pytest.mark.parametrize('mode', ['policy', 'human'])
@pytest.mark.parametrize('choice,outcome', [('s','success'), ('f','failure'), ('n',None), ('q',None)])
def test_all_control_modes_require_an_explicit_outcome(tmp_path, package, mode, choice, outcome):
    rec = importlib.import_module(package + '.recording')
    choose = importlib.import_module(package + '.outcome').choose_episode_decision
    ep = rec.Episode(tmp_path, {})
    ep.add('tick', tick=0, mode=mode)
    ep.add('episode_end', last_tick=0)
    class Keyboard:
        def wait(self, keys, message):
            assert keys == 'sfnq'
            assert 'q 删除本轮并退出' in message
            return choice
    result = choose(ep, Keyboard())
    assert result is {'s':True, 'f':True, 'n':False, 'q':None}[choice]
    assert ep.metadata.get('outcome') == outcome
    ep.decide(False)


def test_keep_refuses_unlabeled_data(tmp_path, package):
    rec = importlib.import_module(package + '.recording')
    ep = rec.Episode(tmp_path, {})
    with pytest.raises(ValueError, match='success or failure'):
        ep.decide(True)
    assert ep.metadata['status'] == 'pending'
    ep.decide(False)


@pytest.mark.parametrize('fail_sync', [False, True])
def test_discard_sync_happens_before_complete_local_deletion(tmp_path, package, fail_sync):
    rec = importlib.import_module(package + '.recording')
    sync_decision = importlib.import_module(package + '.server_recording').sync_decision
    ep = rec.Episode(tmp_path, {})
    rec.atomic_json(ep.path/'server_transaction.json', dict(episode_id=ep.episode_id,token='test'))
    ep.add('event', image=b'x'*4096)
    calls = []
    class Worker:
        def control(self, payload):
            assert ep.path.exists()
            calls.append(payload)
            if fail_sync:
                raise TimeoutError('mock unavailable server')
            return {'status':'ok'}
    run = lambda: ep.decide(False, sync=lambda path: sync_decision(path, Worker()))
    if fail_sync:
        with pytest.raises(TimeoutError):
            run()
    else:
        run()
        assert [c['cmd'] for c in calls] == ['hil_end', 'hil_decide']
        assert calls[-1]['keep'] is False
    assert not ep.path.exists() and list(tmp_path.iterdir()) == []


def test_discard_deletes_even_after_writer_failure(tmp_path, package):
    rec = importlib.import_module(package + '.recording')
    ep = rec.Episode(tmp_path, {})
    ep.add('event', image=b'x'*4096)
    ep.seal()
    ep.error = OSError('mock disk failure')
    with pytest.raises(RuntimeError, match='incomplete'):
        ep.decide(False)
    assert not ep.path.exists()
