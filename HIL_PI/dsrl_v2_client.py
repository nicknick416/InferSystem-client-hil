"""Native PI capture identity; no Actor RPC, noise injection or scheduling gate."""
import hashlib
import re

import msgpack

from HIL_PI.control import PolicyWorker


class DSRLV2Worker(PolicyWorker):
    def __init__(self, address, timeout_ms=30000):
        super().__init__(address, timeout_ms)
        self.dsrl_episode_id = None
        self.dsrl_actor_version = None

    def begin_dsrl_episode(self, episode_id, prompt, model_config):
        if not re.fullmatch("[a-f0-9]{32}", episode_id):
            raise ValueError("Invalid DSRL episode ID")
        status = self.control({"cmd": "dsrl_status"}).get("dsrl", {})
        model = status.get("model", {})
        if (not status.get("enabled") or status.get("mode") not in ("capture", "steer")
                or model.get("config") != model_config
                or model.get("default_prompt") != prompt):
            raise RuntimeError("DSRL server mode/model/prompt does not match collector")
        if status["mode"] == "steer" and not re.fullmatch(
                "[a-f0-9]{64}", status.get("actor_version", "")):
            raise RuntimeError("DSRL steer mode requires a versioned Actor")
        self.dsrl_actor_version = status.get("actor_version") if status["mode"] == "steer" else None
        self.dsrl_episode_id = episode_id

    def submit(self, payload, **kwargs):
        if self.dsrl_episode_id is None:
            raise RuntimeError("DSRL episode has not started")
        if "noise" in payload or "dsrl_noise" in payload.get("extra", {}):
            raise ValueError("v2 capture must retain native PI noise")
        if "dsrl_decision" not in payload.get("extra", {}):
            raise ValueError("Missing DSRL decision snapshot")
        request = dict(payload, extra=dict(
            payload.get("extra", {}),
            dsrl=dict(episode_id=self.dsrl_episode_id, request_id=self.request_id+1),
        ))
        digest = hashlib.sha256(msgpack.packb(request, use_bin_type=True)).hexdigest()
        rid = super().submit(request, **kwargs)
        self.context["dsrl_request_sha256"] = digest
        return rid

    def poll(self):
        row = super().poll()
        if row is None:
            return None
        receipt = row["response"].get("dsrl", {})
        if self.dsrl_actor_version is not None and (
                receipt.get("mode") != "steer"
                or receipt.get("actor_version") != self.dsrl_actor_version):
            raise RuntimeError("Actor version changed during the episode")
        if (receipt.get("schema") != "dsrl_inference_v2"
                or not receipt.get("captured") or not receipt.get("client_paired")
                or receipt.get("episode_id") != self.dsrl_episode_id
                or receipt.get("request_id") != row["request_id"]
                or receipt.get("request_sha256") != row["dsrl_request_sha256"]
                or not re.fullmatch("[a-f0-9]{64}", receipt.get("record_sha256", ""))):
            raise RuntimeError("Server did not confirm durable matching v2 capture")
        return row
