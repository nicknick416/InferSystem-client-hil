from types import SimpleNamespace
from collections import deque
import numpy as np
import pytest
from HIL_PI.dsrl_decisions import snapshot, prepare_request


def arb():
    return SimpleNamespace(mode="policy",actions=deque([{"action":[.2]*32}]),
                           last_action=None,smooth_steps=8,awaiting_first_policy=False,generation=1)


def test_snapshot_is_detached_and_preserves_queue():
    a=arb()
    row=snapshot(a,[0.]*32,[[.2]*32],1.)
    a.actions[0]["action"][0]=5.
    assert row["queue"][0][0]==.2
    assert len(row["state"])==10 and len(row["prefix"][0])==10


def test_short_noise_matches_wire_and_metadata():
    d=snapshot(arb(),[0.]*32,[],1.)
    wire=prepare_request({"extra":{"original":1}},d,np.random.default_rng(4))
    short=wire["extra"]["dsrl_noise"]["short_noise"]
    np.testing.assert_array_equal(wire["noise"],np.repeat(short,50,axis=0))
    assert wire["extra"]["original"]==1
    assert wire["extra"]["dsrl_decision"]==d
    d["state"][0]=9.
    assert wire["extra"]["dsrl_decision"]["state"][0]==0.


def test_only_autonomous_and_bounded():
    a=arb();a.mode="human"
    with pytest.raises(ValueError):
        snapshot(a,[0.]*32,[],1.)
    a.mode="policy";a.actions=deque([{"action":[0.]*32}]*51)
    with pytest.raises(ValueError):
        snapshot(a,[0.]*32,[],1.)
