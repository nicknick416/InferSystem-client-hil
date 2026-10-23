"""Append-only capture with explicit pending/kept/discarded episode decisions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid
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


class Episode:
    def __init__(self, root, metadata):
        self.path = Path(root) / ('pending_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        self.episode_id = uuid.uuid4().hex
        self.metadata = dict(metadata, episode_id=self.episode_id, status='pending', schema='flexiv_hil_execution_v3',
                             observation_storage='lossless_sha256_blobs',compact_records='control.msgpack')
        atomic_json(self.path/'manifest.json', self.metadata)
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

    def _write(self):
        try:
            from HIL.blobs import BlobWriter
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
        if self.metadata.get('hil_variant') == 'keyboard_two_press_hil_v1':
            events = [row for row in compact_records(self.path) if row['kind'] == 'hil_transition']
            self.metadata.update(hil_attempts=sum(row['event'] == 'rollback_requested' for row in events),
                                 hil_entries=sum(row['event'] == 'human_started' for row in events))
            atomic_json(self.path/'manifest.json', self.metadata)

    def decide(self, keep):
        self.seal()
        counts = build_pairs(self.path, int(self.metadata.get('horizon', 50)),
                             float(self.metadata.get('fps', 30))) if keep else {}
        self.metadata.update(status='kept' if keep else 'discarded', **counts)
        atomic_json(self.path/'manifest.json', self.metadata)
        dest = self.path.with_name(self.path.name.replace('pending_', 'kept_' if keep else 'discarded_', 1))
        self.path.rename(dest)
        self.path = dest
        if not keep:
            # Only this application's just-closed episode data, not arbitrary paths.
            (dest/'records.msgpack').unlink()
            (dest/'control.msgpack').unlink(missing_ok=True)
            if (dest/'blobs').exists():shutil.rmtree(dest/'blobs')
        return dest


def records(path, *, resolve_blobs=True):
    from HIL.blobs import decode
    with (Path(path)/'records.msgpack').open('rb') as f:
        for row in msgpack.Unpacker(f, raw=False, max_buffer_size=128*1024*1024):
            yield decode(row,Path(path)/'blobs') if resolve_blobs else row


def compact_records(path):
    """No image reads for end-of-episode processing; v2 remains readable."""
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
    with (Path(path)/'pairs.msgpack').open('wb') as pair_file, \
         (Path(path)/'execution_segments.msgpack').open('wb') as segment_file:
        def flush_span():
            nonlocal span, segment_count
            if span is not None:
                span['length'] = len(span['ticks'])
                segment_file.write(msgpack.packb(span, use_bin_type=True))
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
                    pair_file.write(msgpack.packb(pair, use_bin_type=True))
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
