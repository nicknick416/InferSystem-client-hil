"""Run in the inference client checkout; no hardware or network is used."""
import hashlib
import msgpack
import pytest

from HIL_PI.control import PolicyWorker


class Actor:
    def prepare(self, payload):
        return dict(payload, noise=[[.3] * 32] * 50,
                    extra={"dsrl_noise": {"behavior": "sac_actor", "step": 100}})


class PI:
    def _request(self, payload):
        self.payload = payload
        return {"dsrl": dict(payload["dsrl"], captured=True, schema="dsrl_inference_v1",
                             record_sha256="b" * 64,
                             request_sha256=hashlib.sha256(
                                 msgpack.packb(payload, use_bin_type=True)).hexdigest())}


def test_actor_precedes_pi_and_hashes_actual_noise():
    worker = PolicyWorker("unused")
    worker.dsrl_episode_id = "a" * 32
    worker.dsrl_actor = Actor()
    worker.client = PI()
    original = {"cmd": "predict", "noise": [[0.] * 32] * 50}
    try:
        worker.submit(original, tick=1, generation=1)
        row = worker.drain()
        assert worker.client.payload["noise"][0][0] == .3
        assert original["noise"][0][0] == 0.
        assert row["dsrl_actor"]["step"] == 100
        assert row["dsrl_request_sha256"] == row["response"]["dsrl"]["request_sha256"]
    finally:
        worker.pool.shutdown()


def test_actor_failure_never_calls_pi():
    worker = PolicyWorker("unused")
    worker.dsrl_episode_id = "a" * 32
    class BrokenActor:
        def prepare(self, payload):
            raise TimeoutError("Actor timed out")
    worker.dsrl_actor = BrokenActor()
    worker.client = PI()
    try:
        worker.submit({"cmd": "predict"}, tick=1, generation=1)
        with pytest.raises(TimeoutError):
            worker.drain()
        assert not hasattr(worker.client, "payload")
    finally:
        worker.pool.shutdown()
