"""Control arbitration and single-owner inference networking; no robot imports."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np


class Arbiter:
    def __init__(self, smooth_steps=0, *, rtc_enabled=False):
        self.rtc_enabled = rtc_enabled
        self.smooth_steps = max(0, int(smooth_steps))
        self.last_action = None
        self.last_source = None
        self.mode = 'paused'
        self.generation = 0
        self.segment = 0
        self.actions = deque()
        self.policy_steps = 0
        self.awaiting_first_policy = False

    def switch(self, mode):
        if mode not in ('paused', 'human', 'policy', 'recovery', 'recovery_hold', 'correction'):
            raise ValueError(mode)
        previous = self.mode
        self.mode = mode
        # Ordinary takeover preserves the shadow timeline. RTC invalidates its
        # committed prefix on every mode change, including human takeover.
        if self.rtc_enabled or not (previous == 'policy' and mode == 'human'):
            self.generation += 1
            self.actions.clear()
            self.last_action = self.last_source = None
        self.awaiting_first_policy = self.rtc_enabled and mode == 'policy'
        if mode == 'human':
            self.segment += 1

    def accept(self, response, generation, n_execute, skip_steps=0, source=None):
        if self.rtc_enabled and (generation != self.generation or self.mode not in ('policy', 'human')):
            return False
        actions = np.asarray(response['actions'], dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 32 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError('Expected finite nonempty [H,32] policy actions')
        if self.mode in ('policy', 'human') and generation == self.generation:
            old = list(self.actions)
            had_queue = bool(old)
            indices = list(range(skip_steps, min(len(actions), skip_steps+n_execute)))
            if not indices:
                return False  # An expired response must not erase a usable queue.
            if not old and self.last_action is not None:
                old = [dict(action=self.last_action, **(self.last_source or {}))]
            overlap = (min(len(old), len(indices)) if self.rtc_enabled and self.smooth_steps and had_queue
                       else min(self.smooth_steps, len(indices)))
            if old and overlap:
                old = old[:overlap]
                old += [old[-1]] * (overlap-len(old))
            self.actions.clear()
            for j, index in enumerate(indices):
                raw = actions[index].tolist()
                row = dict(action=raw, raw_action=raw, chunk_index=index, **(source or {}))
                if old and j < overlap:
                    weight = 1.-j/overlap
                    row['action'] = self._blend(old[j]['action'], raw, weight)
                    row['blend_weight_old'] = weight
                    row['blend_source_old'] = {k: old[j][k] for k in
                        ('request_id','chunk_index','observation_time') if k in old[j]}
                self.actions.append(row)
            if self.mode == 'policy':
                self.awaiting_first_policy = False
            return True
        return False


    def rtc_prefix(self, length):
        """Capture queued targets with the observation, before this tick pops one.

        Human shadow actions are never commitments to robot execution.
        """
        if not self.rtc_enabled or self.mode != 'policy':
            return []
        return [list(row['action']) for row in list(self.actions)[:length]]

    def advance(self, candidate):
        # A fresh RTC generation has no executed prefix. Waiting in hold must
        # not consume the first prediction's actions before they even arrive.
        if self.awaiting_first_policy and candidate is None:
            return
        if candidate is not None or self.mode == 'human' or (self.rtc_enabled and self.mode == 'policy'):
            self.policy_steps += 1

    def track(self, action, source=None):
        # During HIL this tracks the shadow policy, never the human command.
        self.last_action = list(action)
        self.last_source = dict(source or {})

    @staticmethod
    def _blend(old, new, weight):
        from Inference.action_processing import canonicalize_action_values, blend_action_values
        from Inference.dispatch import quaternion_to_rotation6d
        def pose(a):
            return canonicalize_action_values(a, 'cartesian', policy_format='canonical',
                                               canonical_dim=32, effective_action_dim=10)
        value = blend_action_values(pose(old), pose(new), weight_old=weight, action_space='cartesian')
        result = list(new)
        result[:3] = value[:3]
        result[3:9] = quaternion_to_rotation6d(value[3:7])
        result[9] = value[7]
        return result


class PolicyWorker:
    """The socket is created, used and closed exclusively by the executor thread.

    Exactly one in-flight request; old responses are drained before reset. No
    automatic retries: duplicates would mutate the server's tactile history.
    """
    def __init__(self, address, timeout_ms=30000):
        self.address, self.timeout_ms = address, timeout_ms
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hil-policy')
        self.client = None
        self.future = None
        self.request_id = 0
        self.context = None
        self.hil_session = None

    def _call(self, payload):
        if self.client is None:
            from Inference.client import InferenceClient
            self.client = InferenceClient(self.address, recv_timeout_ms=self.timeout_ms, max_retries=1)
            self.client.connect()
        response = self.client._request(payload)
        if payload.get('cmd') == 'predict_rtc':
            from HIL.rtc import validate_response
            validate_response(response, payload)
        return response

    @property
    def busy(self):
        return self.future is not None

    def submit(self, payload, *, tick, generation, policy_steps=0, observation_time=None):
        if self.busy:
            raise RuntimeError('One request already in flight')
        self.request_id += 1
        self.context = dict(tick=tick, generation=generation, request_id=self.request_id,
                            request_time=time.monotonic(), policy_steps=policy_steps,
                            observation_time=time.monotonic() if observation_time is None else observation_time)
        if self.hil_session:
            payload = dict(payload, hil=dict(self.hil_session, request_id=self.request_id, tick=tick))
        self.future = self.pool.submit(self._call, payload)
        return self.request_id

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        response = future.result()
        if self.hil_session:
            receipt = response.get('hil', {})
            if (not receipt.get('captured') or receipt.get('episode_id') != self.hil_session['episode_id']
                    or receipt.get('request_id') != self.context['request_id']):
                raise RuntimeError('Server did not acknowledge matching HIL feature capture')
        return dict(self.context, response=response, response_time=time.monotonic())

    def drain(self):
        if self.future is not None:
            self.future.result()  # bounded by socket timeout
            return self.poll()
        return None

    def control(self, payload):
        if self.busy:
            raise RuntimeError('Drain prediction before HIL control request')
        result = self.pool.submit(self._call, payload).result()
        if result.get('status') != 'ok':
            raise RuntimeError(f'HIL control failed: {result}')
        return result

    def begin_episode(self, episode_id, metadata):
        result = self.control(dict(cmd='hil_begin', episode_id=episode_id, metadata=metadata))
        self.hil_session = dict(episode_id=episode_id, token=result['token'])
        current = result.get('serving_critic')
        previous = getattr(self, 'last_serving_critic', None)
        if current:
            if previous is None:
                print(f"[QGF] 本轮使用 Q（server 已确认）：{current}", flush=True)
            elif current != previous:
                print(f"\n[QGF] Q 已切换，本轮推理生效\n  旧 Q：{previous}\n  新 Q：{current}\n", flush=True)
            else:
                print(f"[QGF] Q 未变，本轮继续使用：{current}", flush=True)
            self.last_serving_critic = current
        else:
            print("[QGF] server 未返回 Q 版本，无法确认本轮是否切换", flush=True)
        return self.hil_session

    def end_episode(self):
        if self.hil_session:
            result = self.control(dict(cmd='hil_end', **self.hil_session))
            self.hil_session = None
            return result

    def reset(self):
        if self.busy:
            raise RuntimeError('Drain old request before reset')
        response = self.pool.submit(self._call, {'cmd': 'reset'}).result()
        if response.get('status') != 'ok':
            raise RuntimeError(f'Reset failed: {response}')

    def close(self):
        def close_client():
            if self.client:
                self.client.close()
        self.pool.submit(close_client).result()
        self.pool.shutdown()
