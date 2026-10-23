"""Online stall detection from dispatched targets and subsequent measured feedback."""
from collections import deque
import numpy as np


class FeedbackStuckDetector:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.config['feedback_samples'])
        self.last_time = self.last_fresh = None
        self.previous_target = self.previous_measured = None
        self.source = self.identity = None
        self.stuck = False

    def update(self, timestamp, measured_xyz, published_xyz, *, mode='policy',
               fresh_command=True, source=None, generation=None, chunk_end=False):
        was_stuck = self.stuck
        def report(reason, error=None, excursion=None):
            return dict(stuck=self.stuck,
                event='entered' if self.stuck and not was_stuck else 'cleared' if was_stuck and not self.stuck else None,
                reason=reason, tracking_error_m=error, position_excursion_m=excursion,
                window_span_s=0. if not self.samples else t-self.samples[0][0],
                samples=len(self.samples), suspect_duration_s=0., detector='translation_stall_feedback_v1',
                monitor_only=True, evaluated_chunk=None if self.source is None else dict(self.source))
        try:
            t=float(timestamp); measured=np.asarray(measured_xyz,float); target=np.asarray(published_xyz,float)
            valid=np.isfinite(t) and measured.shape==(3,) and target.shape==(3,)
            valid=valid and np.isfinite(measured).all() and np.isfinite(target).all()
        except (TypeError,ValueError):
            valid=False
        if not valid or mode!='policy':
            self.reset();t=0.
            return report('invalid_sample' if not valid else 'non_policy')
        identity=((generation,source['request_id']) if fresh_command and source is not None
                  and source.get('request_id') is not None else None)
        gap=self.last_time is not None and not 0<t-self.last_time<=self.config['max_sample_gap_s']
        if gap:
            self.reset()
        reason='sample_gap' if gap else 'waiting_chunk'
        error=excursion=None
        # Today's observation is feedback to the PREVIOUS published command.
        if self.previous_target is not None and self.source is not None and not gap:
            delta=self.previous_target-measured;error=float(np.linalg.norm(delta))
            threshold=self.config['release_tracking_error_m' if self.stuck else 'min_tracking_error_m']
            if self.last_fresh is None or t-self.last_fresh>self.config['command_timeout_s']:
                reason='stale_command';self.samples.clear();self.stuck=False
            elif error<threshold:
                reason='below_error_threshold';self.samples.clear();self.stuck=False
            else:
                direction=delta/error
                # Require persistent intention, not alternating high-error targets.
                if self.samples and any(np.dot(direction,s[3])<self.config['min_direction_cosine'] for s in self.samples):
                    self.samples.clear();self.stuck=False
                    reason='direction_changed'
                else:
                    reason='collecting_feedback'
                self.samples.append((self.last_time,self.previous_measured.copy(),measured.copy(),direction))
                positions=[self.samples[0][1]]+[s[2] for s in self.samples]
                excursion=float(np.linalg.norm(np.ptp(np.asarray(positions),axis=0)))
                if excursion>self.config['feedback_max_excursion_m']:
                    self.stuck=False;reason='moving'
                elif len(self.samples)>=self.config['feedback_samples']:
                    self.stuck=True;reason='stuck'
        result=report(reason,error,excursion)
        if identity is not None:
            if identity!=self.identity:
                # Boundary feedback above belongs to the old chunk. Do not mix chunks.
                self.samples.clear()
                self.identity=identity
                self.source=dict(generation=generation,request_id=source['request_id'],
                                 first_executed_chunk_index=source.get('chunk_index'))
            self.last_fresh=t
        self.previous_target=target.copy();self.previous_measured=measured.copy();self.last_time=t
        return result
