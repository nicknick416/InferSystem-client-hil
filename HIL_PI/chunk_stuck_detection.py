"""Stall detection over actually dispatched chunks, without hardware I/O."""
import numpy as np


class ChunkStuckDetector:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.chunk = None
        self.last_time = None
        self.previous_target = None
        self.pending_end = False
        self.stuck = False
        self.consecutive = 0
        self.closed_id = None

    def update(self, timestamp, measured_xyz, published_xyz, *, mode='policy',
               fresh_command=True, source=None, generation=None, chunk_end=False):
        was_stuck = self.stuck
        def report(reason, summary=None):
            result = dict(stuck=self.stuck,
                event='entered' if self.stuck and not was_stuck else 'cleared' if was_stuck and not self.stuck else None,
                reason=reason, tracking_error_m=None, position_excursion_m=None,
                window_span_s=0., samples=0, suspect_duration_s=0.,
                detector='translation_stall_chunk_v1', monitor_only=True,
                evaluated_chunk=None, consecutive_stuck_chunks=self.consecutive)
            if summary: result.update(summary)
            return result
        try:
            t = float(timestamp)
            measured, target = np.asarray(measured_xyz, float), np.asarray(published_xyz, float)
            valid = np.isfinite(t) and measured.shape == (3,) and target.shape == (3,)
            valid = valid and np.isfinite(measured).all() and np.isfinite(target).all()
        except (TypeError, ValueError):
            valid = False
        if not valid or mode != 'policy':
            self.reset()
            return report('invalid_sample' if not valid else 'non_policy')
        identity = ((generation, source['request_id']) if fresh_command and source is not None
                    and source.get('request_id') is not None else None)
        if self.last_time is not None and not 0 < t-self.last_time <= self.config['max_sample_gap_s']:
            # Incomplete chunk evidence must not survive an observation gap.
            self.reset()
            self.closed_id = identity
            self.last_time = t
            self.previous_target = target.copy()
            return report('sample_gap')
        self.last_time = t
        summary = None
        reason = 'collecting_chunk' if self.chunk else 'waiting_chunk'
        if self.chunk is not None:
            c = self.chunk
            c['positions'].append(measured.copy())
            delta = self.previous_target-measured
            c['errors'].append(float(np.linalg.norm(delta)))
            c['deltas'].append(delta)
            boundary = identity is not None and identity != c['identity']
            if self.pending_end or boundary:
                reason, summary = self._evaluate(t)
                self.closed_id = c['identity']
                self.chunk = None
            elif t-c['last_fresh'] > self.config['command_timeout_s']:
                self.closed_id = c['identity']
                self.chunk = None
                self.stuck = False
                self.consecutive = 0
                reason = 'stale_command'
        if identity is not None and identity != self.closed_id:
            if self.chunk is None:
                self.chunk = dict(identity=identity, start=t, last_fresh=t,
                    positions=[measured.copy()], errors=[], deltas=[],
                    source=dict(generation=generation, request_id=source['request_id'],
                                first_executed_chunk_index=source.get('chunk_index')))
            self.chunk['last_fresh'] = t
        self.pending_end = bool(chunk_end and self.chunk is not None and identity == self.chunk['identity'])
        self.previous_target = target.copy()
        return report(reason, summary)

    def _evaluate(self, timestamp):
        c = self.chunk
        errors = np.asarray(c['errors'])
        excursion = float(np.linalg.norm(np.ptp(np.asarray(c['positions']), axis=0)))
        threshold = self.config['release_tracking_error_m' if self.stuck else 'min_tracking_error_m']
        qualifying = errors >= threshold
        fraction = float(np.mean(qualifying)) if len(errors) else 0.
        coherent = False
        if np.any(qualifying):
            directions = np.asarray(c['deltas'])[qualifying]/errors[qualifying,None]
            coherent = bool(np.all(directions @ directions[0] >= self.config['min_direction_cosine']))
        if len(errors) < self.config['min_samples']:
            reason = 'short_chunk'
        elif excursion > self.config['max_position_excursion_m']:
            reason = 'moving'
        elif fraction < self.config['min_error_fraction']:
            reason = 'below_error_threshold'
        elif not coherent:
            reason = 'direction_changed'
        else:
            reason = 'suspected'
        if reason == 'suspected':
            self.consecutive += 1
            self.stuck = self.consecutive >= self.config['confirm_chunks']
            if self.stuck: reason = 'stuck'
        else:
            self.consecutive = 0
            self.stuck = False
        return reason, dict(tracking_error_m=float(np.median(errors)) if len(errors) else None,
            position_excursion_m=excursion, window_span_s=timestamp-c['start'],
            samples=len(errors), error_fraction=fraction, evaluated_chunk=c['source'],
            consecutive_stuck_chunks=self.consecutive)
