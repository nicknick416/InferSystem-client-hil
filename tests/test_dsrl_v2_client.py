from concurrent.futures import Future
import hashlib

import msgpack
import pytest

from HIL_PI.dsrl_v2_client import DSRLV2Worker


class Pool:
    def submit(self, fn, payload):
        self.payload = payload
        self.future = Future()
        return self.future


def worker():
    result = DSRLV2Worker("unused")
    result.pool.shutdown()
    result.pool = Pool()
    result.dsrl_episode_id = "a"*32
    payload = {"state": [0.]*10, "extra": {"dsrl_decision": {"time": 1.}}}
    result.submit(payload, tick=1, generation=1, observation_time=1.)
    assert "dsrl" not in payload["extra"] and "noise" not in result.pool.payload
    receipt = dict(result.pool.payload["extra"]["dsrl"], captured=True, client_paired=True,
                   schema="dsrl_inference_v2", record_sha256="b"*64,
                   request_sha256=hashlib.sha256(msgpack.packb(
                       result.pool.payload, use_bin_type=True)).hexdigest())
    return result, receipt


def test_exact_identity_and_preserved_request():
    w, receipt = worker()
    w.pool.future.set_result({"dsrl": receipt, "actions": [[0.]*32]})
    row = w.poll()
    assert row["request_id"] == 1 and row["dsrl_request_sha256"] == receipt["request_sha256"]
    assert w.poll() is None


@pytest.mark.parametrize("field,value", [("request_id", 9), ("client_paired", False),
                                        ("request_sha256", "bad"), ("schema", "dsrl_inference_v1")])
def test_bad_receipt_rejected(field, value):
    w, receipt = worker()
    receipt[field] = value
    w.pool.future.set_result({"dsrl": receipt})
    with pytest.raises(RuntimeError):
        w.poll()


def test_server_task_binding(monkeypatch):
    w = DSRLV2Worker("unused")
    w.pool.shutdown()
    monkeypatch.setattr(w, "control", lambda _: dict(dsrl=dict(enabled=True, mode="capture",
                         model=dict(config="cord", default_prompt="task"))))
    w.begin_dsrl_episode("c"*32, "task", "cord")
    with pytest.raises(RuntimeError):
        w.begin_dsrl_episode("c"*32, "wrong", "cord")
    with pytest.raises(ValueError):
        w.begin_dsrl_episode("../bad", "task", "cord")
