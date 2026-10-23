"""Actor RPC used inside PolicyWorker's executor, never the control thread."""
import json
from pathlib import Path
import time
import urllib.request

import msgpack
import numpy as np


class ActorClient:
    def __init__(self, url, token_file, config_sha256, timeout_s=3):
        if not url.startswith(("http://", "https://")) or timeout_s <= 0:
            raise ValueError("Invalid Actor URL/timeout")
        self.url = url.rstrip("/") + "/sample"
        self.token = Path(token_file).expanduser().read_text().strip()
        if len(self.token) < 32:
            raise ValueError("Invalid Actor token")
        self.config_sha256, self.timeout_s = config_sha256, timeout_s
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self):
        with self.opener.open(self.url.rsplit("/", 1)[0] + "/health",
                              timeout=self.timeout_s) as response:
            status = json.loads(response.read(16384))
        if status.get("state") not in ("waiting_for_data", "training", "waiting_for_new_data"):
            raise RuntimeError("Online learner is not ready: " + str(status.get("state")))
        if status.get("config_sha256") != self.config_sha256:
            raise RuntimeError("Online Actor config differs from client")
        return status

    def prepare(self, payload):
        extra = dict(payload["extra"], dsrl_config_sha256=self.config_sha256)
        request_payload = dict(payload, extra=extra)
        started = time.monotonic()
        request = urllib.request.Request(
            self.url, data=msgpack.packb(request_payload, use_bin_type=True),
            headers={"Content-Type": "application/msgpack", "Authorization": "Bearer " + self.token})
        # No automatic retry or local-noise fallback after an Actor RPC failure.
        with self.opener.open(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read(16384))
        noise = np.asarray(result.get("short_noise"), dtype=np.float32)
        if (result.get("schema") != "dsrl_actor_v1" or result.get("expansion") != "repeat_last"
                or noise.shape != (1, 32) or not np.isfinite(noise).all()
                or result.get("behavior") not in ("gaussian_initial_exploration", "sac_actor")
                or type(result.get("step")) is not int or result["step"] < 0):
            raise RuntimeError("Invalid Actor response")
        if result["behavior"] == "sac_actor" and (
                result["step"] <= 0 or not isinstance(result.get("actor_sha256"), str)
                or len(result["actor_sha256"]) != 64):
            raise RuntimeError("Missing Actor version")
        extra["dsrl_noise"] = dict(result, rpc_seconds=time.monotonic() - started)
        return dict(request_payload, extra=extra, noise=np.repeat(noise, 50, axis=0).tolist())
