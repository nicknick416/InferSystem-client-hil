"""Client capture identity and fail-closed receipt checks; no hardware."""
from concurrent.futures import Future
import hashlib
import msgpack
import pytest
from HIL_PI.control import PolicyWorker


class Pool:
    def submit(self, fn, payload):
        self.payload = payload
        self.future = Future()
        return self.future


def worker():
    w = PolicyWorker("unused")
    w.pool.shutdown()
    w.pool = Pool()
    w.dsrl_episode_id = "a"*32
    w.submit({"state":[0.]*10},tick=3,generation=1)
    receipt = dict(w.pool.payload["dsrl"],captured=True,schema="dsrl_inference_v1",
                   record_sha256="b"*64,
                   request_sha256=hashlib.sha256(msgpack.packb(w.pool.payload,use_bin_type=True)).hexdigest())
    return w, receipt


def test_capture_request_and_receipt():
    w, receipt = worker()
    w.pool.future.set_result({"dsrl":receipt})
    row = w.poll()
    assert row["dsrl"]["episode_id"] == "a"*32
    assert row["request_id"] == 1 and row["tick"] == 3


@pytest.mark.parametrize("field,value", [("captured",False),("request_id",2),
                                        ("episode_id","c"*32),("request_sha256","bad")])
def test_bad_receipt_rejected(field,value):
    w, receipt = worker()
    receipt[field] = value
    w.pool.future.set_result({"dsrl":receipt})
    with pytest.raises(RuntimeError,match="durable matching capture"):
        w.poll()
