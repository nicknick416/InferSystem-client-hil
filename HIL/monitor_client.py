"""Single-flight Monitor RPC off the robot control thread."""
from concurrent.futures import ThreadPoolExecutor
import math
import time
import uuid


DEFAULTS = dict(server='tcp://127.0.0.1:5559', critical_threshold=0.9236635566,
                problem_threshold=0.5587342978, confirm_frames=1, max_request_hz=30.,
                request_timeout_s=1., max_result_age_s=.2)


def validate_config(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('Invalid tactile monitor options')
    config = dict(DEFAULTS, **raw)
    if not isinstance(config['server'], str) or not config['server'].startswith('tcp://'):
        raise ValueError('Monitor server must be tcp://host:port')
    for key, value in config.items():
        if key == 'server':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid monitor.' + key)
        if key.endswith('_threshold'):
            if not 0 <= value <= 1:
                raise ValueError('Invalid monitor.' + key)
        elif value <= 0:
            raise ValueError('Invalid monitor.' + key)
    if type(config['confirm_frames']) is not int:
        raise ValueError('monitor.confirm_frames must be integer')
    return config


class RPCSession:
    """Persistent socket owned exclusively by its calling worker thread."""
    def __init__(self, config):
        self.config = config
        self.context = self.socket = None

    def __call__(self, message):
        import msgpack
        import zmq
        if self.socket is None:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            timeout = max(1, int(self.config['request_timeout_s'] * 1000))
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout)
            self.socket.connect(self.config['server'])
        try:
            self.socket.send(msgpack.packb(message, use_bin_type=True))
            return msgpack.unpackb(self.socket.recv(), raw=False)
        except Exception:
            self.close()  # A timed-out REQ socket cannot accept another request.
            raise

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.context.term()
            self.socket = self.context = None


def rpc(config, message):
    session = RPCSession(config)
    try:
        return session(message)
    finally:
        session.close()


class MonitorClient:
    def __init__(self, config):
        self.config = validate_config(config.get('monitor', {}))
        self.pool = None
        self.future = None
        self.episode_id = uuid.uuid4().hex
        self.generation = 0
        self.mode = None
        self.sequence = 0
        self.count = 0
        self.last = None
        self.last_send = 0.
        self.last_stamp = None
        self.payload = self.stamps = None
        self.transport = RPCSession(self.config)

    def start(self):
        health = rpc(self.config, dict(cmd='health'))
        if health.get('status') != 'ok' or not health.get('ready'):
            raise RuntimeError('Monitor service unavailable')
        self.model_id = health['model_id']
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hil-monitor')

    def reset(self):
        self.generation += 1
        self.count = 0
        self.last = self.last_stamp = None

    def calibrate_hold(self, width):
        self.reset()  # Baselines remain in the episode observation payload.

    def set_observation(self, payload, stamps):
        self.payload, self.stamps = payload, stamps

    def _consume(self, response, context, now):
        generation, sequence, stamp = context
        if generation != self.generation:
            return
        if (response.get('status') != 'ok' or response.get('model_id') != self.model_id
                or response.get('generation') != generation or response.get('request_id') != sequence
                or response.get('episode_id') != self.episode_id):
            raise ValueError('Monitor response identity/error')
        values = [response[k] for k in ('critical_probability', 'problem_probability')]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid Monitor probabilities')
        fresh = 0 <= now - stamp <= self.config['max_result_age_s']
        positive = (values[0] >= self.config['critical_threshold'] and
                    values[1] >= self.config['problem_threshold'])
        self.count = self.count + 1 if fresh and positive else 0
        stuck = self.count >= self.config['confirm_frames']
        self.last = dict(response, stuck=stuck, ready=True, capture_time=stamp,
                         confirm_count=self.count, critical_threshold=self.config['critical_threshold'],
                         problem_threshold=self.config['problem_threshold'],
                         reason='tactile_monitor' if stuck else 'below_threshold' if fresh else 'stale_result',
                         trigger_reasons=['tactile_monitor'] if stuck else [])

    def update(self, pixels, stamp, mode, now):
        if mode != self.mode:
            self.reset()
            self.mode = mode
        if self.future is not None and self.future.done():
            try:
                self._consume(self.future.result(), self.context, now)
            except Exception as exc:
                self.count = 0
                self.last = dict(stuck=False, reason='monitor_error', error=type(exc).__name__)
            self.future = None
        if mode != 'policy':
            return dict(stuck=False, reason='non_policy')
        stamps = self.stamps or {}
        times = [stamps.get(k) for k in ('rgb_read_complete', 'observation/left_tactile', 'observation/right_tactile')]
        valid = all(isinstance(t, (int, float)) and math.isfinite(t) for t in times)
        capture = min(times) if valid else None
        signature = tuple(times)
        if not valid or not 0 <= now-capture <= self.config['max_result_age_s']:
            self.reset()
            return dict(stuck=False, reason='invalid_or_stale_frame')
        if (self.pool is not None and self.future is None and signature != self.last_stamp
                and now-self.last_send >= 1/self.config['max_request_hz']):
            self.sequence += 1
            self.context = (self.generation, self.sequence, capture)
            # Only required current images and fixed episode baselines cross the wire.
            payload = {k: v for k, v in self.payload.items()
                       if k in ('state', 'prompt') or k.startswith('observation/')}
            payload['extra'] = {'tactile_baseline': self.payload['extra']['tactile_baseline']}
            self.future = self.pool.submit(self.transport, dict(cmd='predict_monitor', payload=payload,
                request_id=self.sequence, generation=self.generation, episode_id=self.episode_id))
            self.last_send, self.last_stamp = now, signature
        if self.last is None:
            return dict(stuck=False, reason='waiting_worker')
        age = now-self.last.get('capture_time', now)
        if age > self.config['max_result_age_s']:
            self.count = 0
            return dict(self.last, stuck=False, reason='stale_result', result_age_ms=age*1000)
        return dict(self.last, result_age_ms=age*1000)

    def close(self):
        if self.pool is not None:
            self.pool.submit(self.transport.close).result()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
