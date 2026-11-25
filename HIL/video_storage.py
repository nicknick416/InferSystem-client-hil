"""Collector-style CSV/JSONL + H.265 archives with explicit observation references.

No camera/robot I/O. Encoding runs only on the episode's background writer.
Video time is nominal; the original observation and control timestamps remain
authoritative. Historical frames and baselines resolve by explicit frame index.
"""
from __future__ import annotations

import base64
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

FORMAT = 'collector_h265_v1'
VIDEO_KEY = '__hil_video_frame__'
BYTES_KEY = '__hil_bytes_base64__'
PNG = b'\x89PNG\r\n\x1a\n'


def is_image(value):
    return isinstance(value, bytes) and (value.startswith(PNG) or value.startswith(b'\xff\xd8\xff'))


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sync_file(file):
    file.flush()
    os.fsync(file.fileno())


def find_ffmpeg(value=None):
    candidates = [value, shutil.which('ffmpeg'), str(Path(sys.prefix)/'bin/ffmpeg'),
                  str(Path(sys.prefix).parent/'dc/bin/ffmpeg')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError('H.265 recording requires ffmpeg; configure hil.recording.ffmpeg')


def settings(config=None):
    config = dict(config or {})
    allowed = {'format', 'qp', 'ffmpeg', 'vaapi_device'}
    if set(config)-allowed:
        raise ValueError(f'Unknown recording options: {set(config)-allowed}')
    if config.get('format', 'collector_h265') != 'collector_h265':
        raise ValueError('Expected recording.format: collector_h265')
    qp = config.get('qp', 30)
    if type(qp) is not int or not 1 <= qp <= 51:
        raise ValueError('recording.qp must be an integer in [1, 51]')
    return dict(qp=qp, ffmpeg=find_ffmpeg(config.get('ffmpeg')),
                vaapi_device=config.get('vaapi_device', '/dev/dri/renderD128'))


def probe(config=None):
    """Real one-frame encoder probe; no robot, network, or capture devices."""
    opt = settings(config)
    result = subprocess.run([opt['ffmpeg'], '-hide_banner', '-loglevel', 'error',
        '-vaapi_device', opt['vaapi_device'], '-f', 'lavfi', '-i', 'color=black:s=640x480:r=30',
        '-frames:v', '1', '-vf', 'format=nv12,hwupload', '-c:v', 'hevc_vaapi',
        '-qp', str(opt['qp']), '-f', 'null', '-'], capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError('H.265 hardware encoder unavailable: '+result.stderr.decode(errors='replace')[-1500:])
    return opt


class VideoStream:
    def __init__(self, directory, shape, fps, options):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.shape = tuple(shape)
        h, w, channels = shape
        if channels != 3 or h % 2 or w % 2:
            raise ValueError('H.265 requires even-sized HWC BGR images')
        self.frames = 0
        self.closed = False
        self.stderr = (directory/'encoder.log').open('wb')
        self.timestamps = (directory/'timestamps.csv').open('w', newline='')
        self.csv = csv.writer(self.timestamps)
        self.csv.writerow(['frame_index', 'tick_first_reference', 'sensor_time', 'source_sha256'])
        self.process = subprocess.Popen([options['ffmpeg'], '-hide_banner', '-loglevel', 'error',
            '-vaapi_device', options['vaapi_device'], '-f', 'rawvideo', '-pixel_format', 'bgr24',
            '-video_size', f'{w}x{h}', '-framerate', str(fps), '-i', 'pipe:0',
            '-vf', 'format=nv12,hwupload', '-c:v', 'hevc_vaapi', '-qp', str(options['qp']),
            '-g', str(max(1, round(fps))), '-y', str(directory/'video.mp4')],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)

    def write(self, frame, digest, tick, sensor_time):
        if tuple(frame.shape) != self.shape:
            raise ValueError('Image resolution changed within episode')
        data = memoryview(np.ascontiguousarray(frame)).cast('B')
        deadline = time.monotonic()+10.
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            while data:
                if self.process.poll() is not None:
                    raise RuntimeError('H.265 encoder exited while writing')
                if time.monotonic() > deadline:
                    raise TimeoutError('H.265 encoder stalled for 10 seconds')
                if not selector.select(.1):
                    continue
                try:
                    count = os.write(self.process.stdin.fileno(), data)
                    data = data[count:]
                except BlockingIOError:
                    continue
        index = self.frames
        self.frames += 1
        self.csv.writerow([index, tick, sensor_time, digest])
        return index

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise TimeoutError('H.265 finalization exceeded 30 seconds')
            if code:
                raise RuntimeError(f'H.265 encoder failed ({code}); see {self.directory}/encoder.log')
            sync_file(self.timestamps)
            with (self.directory/'video.mp4').open('rb') as video:
                os.fsync(video.fileno())
            with (self.directory/'config.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['encoder', 'width', 'height', 'frame_count', 'timing'])
                writer.writerow(['hevc_vaapi', self.shape[1], self.shape[0], self.frames,
                                 'explicit_reference_and_sensor_timestamp'])
                sync_file(f)
        finally:
            self.stderr.close()
            self.timestamps.close()


class VideoArchive:
    def __init__(self, root, fps, config=None):
        self.root, self.fps = Path(root), fps
        self.options = settings(config)
        self.streams = {}
        self.saved = {}
        self.identities = collections.OrderedDict()

    def encode(self, value, *, path=(), tick=None, sensor_times=None):
        if isinstance(value, bytes):
            if not is_image(value):
                return {BYTES_KEY: base64.b64encode(value).decode('ascii')}
            # Images use the wire key as identity; baselines have a separate stream.
            key = next((p for p in reversed(path) if p.startswith('observation/')), '/'.join(path))
            baseline = 'tactile_baseline' in path
            stream_key = ('baseline/' if baseline else '')+key
            current = len(path) == 2 and path[0] == 'observation'
            cached = self.identities.get(id(value))
            if cached is not None and cached[0] is value:
                digest = cached[1]
            else:
                digest = hashlib.sha256(value).hexdigest()
                self.identities[id(value)] = (value, digest)
                if len(self.identities) > 128:
                    self.identities.popitem(last=False)
            identity = (stream_key, digest)
            # Current views get one encoded frame per tick, as in the collector.
            # Histories refer to frames already written; repeated baselines are stored once.
            if not current and identity in self.saved:
                return self.saved[identity]
            if not current and not baseline:
                # A history first seen before recording began must not add an
                # extra frame to the current-view video. Keep its explicit ref.
                stream_key = 'history/'+key
            frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid recorded image')
            name = ''.join(c if c.isalnum() or c in '._-' else '.' for c in stream_key)
            name = name[:100]+'.'+hashlib.sha256(stream_key.encode()).hexdigest()[:8]
            if name not in self.streams:
                self.streams[name] = VideoStream(self.root/name, frame.shape, self.fps, self.options)
            stamp = (sensor_times or {}).get(key, (sensor_times or {}).get('rgb_read_complete'))
            # Only the current sample has a known capture stamp; histories/baselines do not.
            if not current:
                stamp = None
            index = self.streams[name].write(frame, digest, tick, stamp)
            ref = {VIDEO_KEY: 1, 'stream': name, 'frame': index, 'source_sha256': digest,
                   'shape': list(frame.shape)}
            self.saved[identity] = ref
            return ref
        if isinstance(value, dict):
            return {k:self.encode(v, path=path+(k,), tick=tick, sensor_times=sensor_times)
                    for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(v, path=path, tick=tick, sensor_times=sensor_times) for v in value]
        return value

    def close(self):
        error = None
        for stream in self.streams.values():
            try:
                stream.close()
            except BaseException as exc:
                error = error or exc
        if error:
            raise error


class CollectorWriter:
    """Numeric modalities + HIL table/events; no legacy msgpack/blob files."""
    def __init__(self, root, fps, config=None):
        self.root = Path(root)
        (self.root/'hil').mkdir(parents=True, exist_ok=False)
        self.archive = VideoArchive(self.root, fps, config)
        self.events = (self.root/'hil/events.jsonl').open('w')
        self.observations = (self.root/'hil/observations.jsonl').open('w')
        self.tick_file = (self.root/'hil/ticks.csv').open('w', newline='')
        self.tick_writer = None
        self.fields = []
        self.tick_count = 0
        self.modalities = {}

    def _numeric(self, name, tick, timestamp, names, values):
        if values is None:
            return
        if len(values) != len(names):
            raise ValueError(f'{name}: invalid number of values')
        if name not in self.modalities:
            directory = self.root/name
            directory.mkdir()
            f = (directory/'data.csv').open('w', newline='')
            writer = csv.writer(f)
            writer.writerow(['tick', 'timestamp', *names])
            self.modalities[name] = (f, writer)
        self.modalities[name][1].writerow([tick, timestamp, *values])

    def add(self, item):
        encoded = self.archive.encode(item, tick=item.get('tick'), sensor_times=item.get('sensor_times'))
        if item['kind'] != 'tick':
            self.events.write(dumps(encoded)+'\n')
            return
        observation = encoded.pop('observation', None)
        if self.tick_writer is None:
            self.fields = list(encoded)
            self.tick_writer = csv.DictWriter(self.tick_file, fieldnames=self.fields+['_extra','_missing'])
            self.tick_writer.writeheader()
        row = {k:dumps(encoded.get(k)) for k in self.fields}
        row['_extra'] = dumps({k:v for k,v in encoded.items() if k not in self.fields})
        row['_missing'] = dumps([k for k in self.fields if k not in encoded])
        self.tick_writer.writerow(row)
        self.observations.write(dumps(observation)+'\n')
        self.events.write(dumps({'__hil_tick_row__':self.tick_count,
                                 'has_observation':'observation' in item})+'\n')
        self.tick_count += 1
        tick, timestamp = item['tick'], item.get('publish_start')
        state = item.get('robot_state', {})
        pose_names = ['x','y','z','qw','qx','qy','qz']
        self._numeric('actions.eef_pose', tick, timestamp, pose_names, item.get('published_pose'))
        width = item.get('commanded_gripper_m')
        self._numeric('actions.gripper', tick, timestamp, ['width_m'], None if width is None else [width])
        self._numeric('observation.state.eef_pose', tick, item.get('state_time'), pose_names, state.get('eef_pose'))
        joints = state.get('joint_positions')
        self._numeric('observation.state.joint_position', tick, item.get('state_time'),
                      [f'joint_{i}' for i in range(len(joints or []))], joints)
        canonical = item.get('state')
        self._numeric('observation.state.gripper', tick, item.get('state_time'), ['width_m'],
                      [canonical[9]] if canonical is not None and len(canonical)>9 else None)
        if self.tick_count % 30 == 0:
            for f in [self.events, self.observations, self.tick_file]:
                f.flush()

    def close(self):
        error = None
        try:
            self.archive.close()
        except BaseException as exc:
            error = exc
        for f in [self.events, self.observations, self.tick_file]+[v[0] for v in self.modalities.values()]:
            try:
                sync_file(f)
            except BaseException as exc:
                error = error or exc
            finally:
                f.close()
        if error:
            raise error


class VideoDecoder:
    """Sequential readers with exact backwards seek by replay; bounded PNG cache."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.readers = {}
        self.cache = collections.OrderedDict()

    def decode(self, value):
        if isinstance(value, dict):
            if VIDEO_KEY in value:
                stream, index = value['stream'], value['frame']
                if type(index) is not int or index < 0 or Path(stream).name != stream or stream in ('.','..'):
                    raise ValueError('Invalid video reference')
                path = (self.root/stream/'video.mp4').resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError('Video reference escapes episode')
                key = (stream,index)
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                entry = self.readers.get(stream)
                if entry is None or entry[1] > index:
                    if entry is not None:
                        entry[0].release()
                    entry = [cv2.VideoCapture(str(path)), 0]
                    self.readers[stream] = entry
                cap = entry[0]
                while entry[1] <= index:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Missing video frame {stream}/{index}')
                    entry[1] += 1
                if list(frame.shape) != value['shape']:
                    raise ValueError('Decoded video shape differs from recorded shape')
                ok, png = cv2.imencode('.png', frame)
                if not ok:
                    raise RuntimeError('Cannot encode reconstructed observation')
                result = png.tobytes()
                self.cache[key] = result
                if len(self.cache)>128:
                    self.cache.popitem(last=False)
                return result
            if BYTES_KEY in value:
                return base64.b64decode(value[BYTES_KEY], validate=True)
            return {k:self.decode(v) for k,v in value.items()}
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        return value

    def close(self):
        for cap, _ in self.readers.values():
            cap.release()
        self.readers.clear()
        self.cache.clear()


def iter_records(root, *, observations=True):
    """Restore event order without decoding any video. Numeric values stay exact."""
    root = Path(root)
    with (root/'hil/events.jsonl').open() as events, (root/'hil/ticks.csv').open() as ticks, \
         (root/'hil/observations.jsonl').open() as obs:
        reader = csv.DictReader(ticks)
        index = 0
        for line in events:
            event = json.loads(line)
            if '__hil_tick_row__' not in event:
                yield event
                continue
            if event['__hil_tick_row__'] != index:
                raise ValueError('Non-contiguous HIL row index')
            raw = next(reader, None)
            observation_line = obs.readline()
            if raw is None or not observation_line:
                raise ValueError('Incomplete HIL table/observation mapping')
            row = {k:json.loads(v) for k,v in raw.items()}
            for key in row.pop('_missing', []):
                row.pop(key)
            row.update(row.pop('_extra'))
            if observations and event.get('has_observation', True):
                row['observation'] = json.loads(observation_line)
            index += 1
            yield row
        if next(reader, None) is not None or obs.readline():
            raise ValueError('Unindexed HIL rows or observations')
