"""Timestamped capture sessions; keep labeled outcomes and delete discarded data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import msgpack
import numpy as np


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def metadata_path(path):
    path = Path(path)
    return path/'metadata.json' if (path/'metadata.json').exists() else path/'manifest.json'


def read_metadata(path):
    return json.loads(metadata_path(path).read_text())


def decode_observation(value, path):
    if (Path(path)/'hil/ticks.csv').exists():
        from HIL.video_storage import VideoDecoder
        decoder = VideoDecoder(path)
        try:
            return decoder.decode(value)
        finally:
            decoder.close()
    from HIL_PI.blobs import decode
    return decode(value, Path(path)/'blobs')


def capture_time():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


class RecordingSession:
    """One directory per live invocation; indices count all attempted episodes."""

    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        while True:
            started = capture_time()
            self.path = root / started.strftime('%Y%m%d_%H%M%S_%f')
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                continue
        self.started_at = started.isoformat()
        self.episode_index = 0

    def next_metadata(self):
        self.episode_index += 1
        return dict(session_name=self.path.name, session_started_at=self.started_at,
                    episode_index=self.episode_index)


class Episode:
    def __init__(self, root, metadata):
        started = capture_time()
        index = metadata.get('episode_index', 1)
        if type(index) is not int or index < 1:
            raise ValueError('episode_index must be a positive integer')
        self.final_name = started.strftime('%Y%m%d_%H%M%S_%f') + f'_episode_{index:06d}'
        self.path = Path(root) / ('.pending_' + self.final_name)
        self.path.mkdir(parents=True, exist_ok=False)
        self.episode_id = uuid.uuid4().hex
        self.metadata = dict(metadata, episode_index=index, episode_started_at=started.isoformat(),
                             episode_id=self.episode_id, status='pending', schema='flexiv_hil_execution_v3',
                             observation_storage='lossless_sha256_blobs',compact_records='control.msgpack')
        self.recording_config = metadata.get('config', {}).get('hil', {}).get('recording', {})
        self.video_format = self.recording_config.get('format') == 'collector_h265'
        self.queue_high_watermark = 0
        self.storage_stats = {}
        if self.video_format:
            from HIL.video_storage import FORMAT, settings
            settings(self.recording_config)
            self.metadata.update(schema=FORMAT, storage_format=FORMAT,
                observation_storage='h265_explicit_frame_refs', compact_records='hil/ticks.csv',
                image_storage_lossy=True, video_encoding=dict(codec='hevc_vaapi', qp=self.recording_config.get('qp',30), pixel_format='yuv420p'))
        self.metadata_filename = 'metadata.json' if self.video_format else 'manifest.json'
        atomic_json(self.path/self.metadata_filename, self.metadata)
        self.queue = queue.Queue(maxsize=128)
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def add(self, kind, **data):
        if self.error:
            raise RuntimeError('HIL disk writer failed') from self.error
        if self.closed:
            raise RuntimeError('Episode already sealed')
        # No silent data loss or blocking on disk in the control loop.
        self.queue.put_nowait(dict(kind=kind, **data))
        self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())

    def _write(self):
        if self.video_format:
            self._write_video()
            return
        try:
            from HIL_PI.blobs import BlobWriter
            with BlobWriter(self.path/'blobs') as blobs, (self.path/'records.msgpack').open('wb') as f, (self.path/'control.msgpack').open('wb') as compact:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    f.write(msgpack.packb(blobs.encode(item), use_bin_type=True))
                    if item['kind'] in ('tick','episode_end','hil_transition'):
                        compact.write(msgpack.packb({k:v for k,v in item.items() if k!='observation'},use_bin_type=True))
                        compact.flush()
                    f.flush()
                os.fsync(f.fileno())
                os.fsync(compact.fileno())
        except Exception as exc:
            self.error = exc

    def _write_video(self):
        from HIL.video_storage import CollectorWriter
        writer = None
        try:
            writer = CollectorWriter(self.path, self.metadata.get('fps',30), self.recording_config)
            while True:
                item = self.queue.get()
                if item is None:
                    break
                writer.add(item)
        except BaseException as exc:
            self.error = exc
        finally:
            if writer is not None:
                started = time.monotonic()
                try:
                    writer.close()
                    self.storage_stats = dict(total_frames=writer.tick_count,
                        video_frames={name:stream.frames for name,stream in writer.archive.streams.items()},
                        encoder_finalization_s=time.monotonic()-started)
                except BaseException as exc:
                    self.error = self.error or exc

    def seal(self):
        if not self.closed:
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()
            self.closed = True
        if self.error:
            raise RuntimeError('HIL capture incomplete; left pending') from self.error
        if self.video_format:
            self.metadata.update(self.storage_stats, writer_queue_high_watermark=self.queue_high_watermark)
            atomic_json(self.path/self.metadata_filename, self.metadata)
        if self.metadata.get('hil_variant') == 'keyboard_two_press_hil_v1':
            events = [row for row in compact_records(self.path) if row['kind'] == 'hil_transition']
            self.metadata.update(hil_attempts=sum(row['event'] == 'rollback_requested' for row in events),
                                 hil_entries=sum(row['event'] == 'human_started' for row in events))
            atomic_json(self.path/self.metadata_filename, self.metadata)

    def decide(self, keep, *, sync=None):
        if type(keep) is not bool:
            raise ValueError('keep must be an explicit boolean decision')
        if self.metadata['status'] != 'pending':
            raise RuntimeError('Episode already decided')
        if not keep:
            # Notify optional server capture while its transaction still exists.
            # A failed write/sync must not leave discarded images or metadata.
            try:
                self.seal()
                self.metadata['status'] = 'discarded'
                atomic_json(self.path/self.metadata_filename, self.metadata)
                if sync is not None:
                    sync(self.path)
            finally:
                self.metadata['status'] = 'discarded'
                shutil.rmtree(self.path)
            return self.path  # Deleted path, useful only for reporting.
        if self.metadata.get('outcome') not in ('success', 'failure'):
            raise ValueError('A kept episode must be labeled success or failure')
        self.seal()
        counts = build_pairs(self.path, int(self.metadata.get('horizon', 50)),
                             float(self.metadata.get('fps', 30)))
        self.metadata.update(status='kept', **counts)
        atomic_json(self.path/self.metadata_filename, self.metadata)
        dest = self.path.with_name(self.final_name)
        if dest.exists():
            raise FileExistsError(dest)
        self.path.rename(dest)
        self.path = dest
        if sync is not None:
            sync(dest)
        return dest


def records(path, *, resolve_blobs=True):
    if (Path(path)/'hil/ticks.csv').exists():
        from HIL.video_storage import VideoDecoder, iter_records
        decoder = VideoDecoder(path) if resolve_blobs else None
        try:
            for row in iter_records(path):
                yield decoder.decode(row) if decoder else row
        finally:
            if decoder:
                decoder.close()
        return
    from HIL_PI.blobs import decode
    with (Path(path)/'records.msgpack').open('rb') as f:
        for row in msgpack.Unpacker(f, raw=False, max_buffer_size=128*1024*1024):
            yield decode(row,Path(path)/'blobs') if resolve_blobs else row


def compact_records(path):
    """No image reads for end-of-episode processing; v2 remains readable."""
    if (Path(path)/'hil/ticks.csv').exists():
        from HIL.video_storage import iter_records
        for row in iter_records(path, observations=False):
            if row['kind'] in ('tick','episode_end','hil_transition'):
                yield row
        return
    compact=Path(path)/'control.msgpack'
    if compact.exists():
        with compact.open('rb') as f:
            yield from msgpack.Unpacker(f,raw=False,max_buffer_size=128*1024*1024)
    else:
        for row in records(path,resolve_blobs=False):
            if row['kind'] in ('tick','episode_end','hil_transition'):
                yield {k:v for k,v in row.items() if k!='observation'}


def build_pairs(path, horizon=50, fps=30.):
    """One pair per ACTUAL execution tick, not a synthetic 50-step request pair.

    execution_segments groups contiguous selected/shadow policy samples by origin
    chunk. Its variable length reflects replacement, starvation and intervention.
    Neither file claims that a mixed-origin sequence is a single-state Q chunk.
    """
    paired, unmatched, human_ticks, segment_count = 0, 0, 0, 0
    span = None
    with _RecordOutput(path, 'pairs') as pair_file, \
         _RecordOutput(path, 'execution_segments') as segment_file:
        def flush_span():
            nonlocal span, segment_count
            if span is not None:
                span['length'] = len(span['ticks'])
                segment_file.add(span)
                segment_count += 1
                span = None

        for row in compact_records(path):
            if row['kind'] != 'tick':
                continue
            if row['mode'] == 'human':
                human_ticks += 1
                if row.get('pair_valid'):
                    pair = {k: row[k] for k in ('tick', 'segment', 'state', 'state_time',
                        'publish_start', 'publish_time', 'human_action', 'policy_action',
                        'policy_source', 'policy_observation_age_s', 'published_action')}
                    pair.update(robot_send_start=row.get('robot_send_start'),
                                robot_send_end=row.get('robot_send_end'),
                                policy_raw_action=row.get('policy_raw_action'),
                                observation_ref={'kind':'tick', 'tick':row['tick']},
                                alignment='same_execution_tick_not_same_generation_state',
                                policy_executed=False, preference_label='operator_kept_human_correction',
                                task_reward=None)
                    if row.get('hil_id') is not None:
                        pair.update(hil_id=row['hil_id'], hil_variant='keyboard_two_press_hil_v1')
                    pair_file.add(pair)
                    paired += 1
                else:
                    unmatched += 1
            source = row.get('policy_source')
            if source is None:
                flush_span()
                continue
            key = (row['segment'], row['mode'], source['request_id'])
            if span is not None:
                old_key = (span['segment'], span['mode'], span['request_id'])
                contiguous = (row['tick'] == span['ticks'][-1]+1 and
                              source['chunk_index'] == span['chunk_indices'][-1]+1 and
                              0 < row['publish_start']-span['publish_times'][-1] <= 1.5/fps)
                if key != old_key or not contiguous:
                    flush_span()
            if span is None:
                span = dict(segment=row['segment'], mode=row['mode'], request_id=source['request_id'],
                            policy_executed=row['mode']=='policy', ticks=[], chunk_indices=[],
                            publish_times=[], policy_action=[], human_action=[], pair_valid=[])
            span['ticks'].append(row['tick'])
            span['chunk_indices'].append(source['chunk_index'])
            span['publish_times'].append(row['publish_start'])
            span['policy_action'].append(row.get('policy_action'))
            span['human_action'].append(row['human_action'])
            span['pair_valid'].append(row.get('pair_valid', False))
        flush_span()
    return dict(paired_steps=paired, unmatched_human_steps=unmatched,
                human_ticks=human_ticks, execution_segments=segment_count)


class _RecordOutput:
    def __init__(self, path, name):
        self.json = (Path(path)/'hil/ticks.csv').exists()
        self.path = Path(path)/('hil/'+name+'.jsonl' if self.json else name+'.msgpack')

    def __enter__(self):
        self.file = self.path.open('w' if self.json else 'wb')
        return self

    def add(self, value):
        self.file.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n' if self.json
                        else msgpack.packb(value, use_bin_type=True))

    def __exit__(self, *args):
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
        finally:
            self.file.close()
