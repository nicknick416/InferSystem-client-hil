"""Autonomous DSRL decision snapshots; no robot imports or device operations."""
from copy import deepcopy
import numpy as np


def snapshot(arbiter, state, prefix, when):
    if arbiter.mode != "policy":
        raise ValueError("DSRL autonomous collection requires policy mode")
    queue = [list(row["action"][:10]) for row in arbiter.actions]
    if len(queue) > 50 or len(prefix) > 8:
        raise ValueError("DSRL queue/prefix exceeds fixed state capacity")
    return dict(schema="dsrl_decision_v1", time=float(when),
                state=list(state[:10]), queue=queue,
                prefix=[list(row[:10]) for row in prefix],
                last_action=None if arbiter.last_action is None else list(arbiter.last_action[:10]),
                smooth_steps=arbiter.smooth_steps,
                awaiting_first_policy=arbiter.awaiting_first_policy,
                generation=arbiter.generation)


def prepare_request(payload, decision, rng, horizon=50, action_dim=32):
    """Initial exploration uses a single noise row, repeated over the horizon."""
    short = rng.standard_normal((1, action_dim)).astype(np.float32)
    result = dict(payload)
    result["noise"] = np.repeat(short, horizon, axis=0).tolist()
    result["extra"] = dict(payload.get("extra", {}), dsrl_decision=deepcopy(decision),
                          dsrl_noise=dict(short_noise=short.tolist(), expansion="repeat_last",
                                          behavior="gaussian_initial_exploration"))
    return result
