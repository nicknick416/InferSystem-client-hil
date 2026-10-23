import pytest
from HIL.monitor_client import MonitorClient, validate_config


def response(client, critical=.99, problem=.9, generation=0):
    return dict(status='ok', model_id='head', episode_id=client.episode_id,
                generation=generation, request_id=1,
                critical_probability=critical, problem_probability=problem)


def client(**options):
    result = MonitorClient(dict(monitor=options))
    result.model_id = 'head'
    return result


@pytest.mark.parametrize('options', [dict(confirm_frames=0), dict(confirm_frames=1.5),
    dict(critical_threshold=1.1), dict(problem_threshold=float('nan')),
    dict(max_request_hz=False), dict(unknown=1)])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        validate_config(options)


def test_both_probabilities_required():
    c = client()
    c._consume(response(c, problem=.1), (0, 1, 10), 10.1)
    assert not c.last['stuck']
    c._consume(response(c), (0, 1, 10), 10.1)
    assert c.last['stuck']


def test_stale_and_generation_cannot_trigger():
    c = client()
    c._consume(response(c), (0, 1, 10), 10.3)
    assert not c.last['stuck']
    c.reset()
    c._consume(response(c), (0, 1, 10), 10.1)
    assert c.last is None


def test_confirmation_resets_on_negative():
    c = client(confirm_frames=2)
    c._consume(response(c), (0, 1, 10), 10.1)
    assert not c.last['stuck']
    c._consume(response(c, critical=0), (0, 1, 10), 10.1)
    assert c.count == 0


def test_invalid_identity_rejected():
    c = client()
    r = response(c)
    r['model_id'] = 'wrong'
    with pytest.raises(ValueError):
        c._consume(r, (0, 1, 10), 10.1)


def test_non_policy_never_triggers():
    c = client()
    c._consume(response(c), (0, 1, 10), 10.1)
    assert not c.update(None, None, 'human', 10.1)['stuck']


def test_invalid_frame_invalidates_inflight_generation():
    c = client()
    c.mode = 'policy'
    old = c.generation
    assert c.update(None, None, 'policy', 10)['reason'] == 'invalid_or_stale_frame'
    assert c.generation > old


def test_rule_alias_preserves_options():
    from HIL.tactile_stuck_detection import TactileStuckDetector
    old = TactileStuckDetector(dict(method='local_normal', enabled=True, normal_min_blocks=3))
    new = TactileStuckDetector(dict(method='rule_based', enabled=True, normal_min_blocks=3))
    assert old.config == new.config


def test_monitor_does_not_create_rule_worker(monkeypatch):
    from HIL.tactile_stuck_detection import CombinedStuckDetector
    calls = []
    monkeypatch.setattr(MonitorClient, 'start', lambda self: calls.append('monitor'))
    detector = CombinedStuckDetector({}, dict(enabled=True, method='monitor'))
    detector.start()
    assert isinstance(detector.worker, MonitorClient)
    assert calls == ['monitor']
    detector.close()


def test_rpc_timeout_releases_socket(monkeypatch):
    import zmq
    from HIL.monitor_client import RPCSession
    events = []
    class Socket:
        def setsockopt(self, *args): pass
        def connect(self, *args): pass
        def send(self, *args): pass
        def recv(self): raise zmq.Again()
        def close(self): events.append('closed')
    class Context:
        def socket(self, *args): return Socket()
        def term(self): events.append('terminated')
    monkeypatch.setattr(zmq, 'Context', Context)
    session = RPCSession(validate_config({}))
    with pytest.raises(zmq.Again):
        session(dict(cmd='health'))
    assert session.socket is None
    assert events == ['closed', 'terminated']


def test_inflight_update_never_waits_or_resubmits():
    c = client()
    c.mode = 'policy'
    class Future:
        def done(self): return False
        def result(self): raise AssertionError('must not wait')
    c.future = Future()
    c.stamps = dict(rgb_read_complete=10., **{'observation/left_tactile':10., 'observation/right_tactile':10.})
    assert c.update(None, None, 'policy', 10.01)['reason'] == 'waiting_worker'
    assert c.sequence == 0
