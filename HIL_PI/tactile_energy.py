"""Two-point tactile deformation proxy, not physical force or mechanical energy."""
from collections import deque
import cv2
import numpy as np


class TactileEnergy:
    def __init__(self, config):
        self.config = config
        self.zero = None
        self.hold = self.threshold = None
        self.hold_epoch = 0
        self.recent = deque(maxlen=5)
        self.last_stamp = None
        self.reset()

    def reset(self):
        # Preserve two-point calibration across recovery; never preserve an alarm.
        self.count = 0
        self.stuck = False

    def set_zero(self, encoded):
        self.zero = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.hold = self.threshold = None
        self.last_stamp = None
        self.recent.clear()
        self.reset()

    def update(self, pixels, stamp, hold_epoch, active, opening):
        report = dict(stuck=False, reason='waiting_zero', score=None, energy=None,
                      hold_score=self.hold, threshold=self.threshold, increase=None)
        if self.zero is None or pixels.shape != self.zero.shape:
            return report
        if self.last_stamp is not None and not 0 < stamp-self.last_stamp <= self.config['max_sample_gap_s']:
            self.recent.clear()
            self.count = 0
        if stamp == self.last_stamp:
            return dict(report, reason='duplicate_frame')
        self.last_stamp = stamp
        diff = cv2.absdiff(pixels, self.zero)
        roi = cv2.resize(diff,(80,45),interpolation=cv2.INTER_AREA)[5:40,12:68]
        score = float(np.mean(roi))
        self.recent.append(score)
        if opening:
            self.hold = self.threshold = None
            self.count = 0
            self.stuck = False
            self.hold_epoch = hold_epoch
            return dict(report, reason='gripper_opening', score=score, hold_score=None, threshold=None)
        if hold_epoch != self.hold_epoch:
            self.hold_epoch = hold_epoch
            hold = float(np.median(self.recent))
            mad = float(np.median(np.abs(np.asarray(self.recent)-hold)))
            self.hold = hold if hold >= self.config['energy_min_hold_score'] else None
            self.threshold = (max(.5+self.config['energy_margin'], .5+6*1.4826*mad*.5/hold)
                              if self.hold is not None else None)
            self.count = 0
            self.stuck = False
        if self.hold is None:
            return dict(report, reason='waiting_hold_p', score=score, hold_score=None, threshold=None)
        energy = .5*score/self.hold
        if not active:
            self.count = 0
            self.stuck = False
        elif energy > self.threshold:
            self.count += 1
            self.stuck |= self.count >= self.config['energy_confirm_frames']
        else:
            self.count = 0
        return dict(report, stuck=self.stuck, reason='energy_high' if self.stuck else
                    'non_policy' if not active else 'energy_normal', score=score, energy=energy,
                    hold_score=self.hold, threshold=self.threshold, increase=energy-.5,
                    confirm_frames=self.count)
