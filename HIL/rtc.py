"""Opt-in HIL RTC protocol, compatible with InferSystem2.0 train_time_prefix."""
from copy import deepcopy
import numpy as np


def rtc_config(raw):
    value = raw.get('inference', {}).get('rtc', {})
    if not isinstance(value, dict):
        raise ValueError('inference.rtc must be a mapping')
    unknown = set(value) - {'enabled', 'prefix_length'}
    if unknown:
        raise ValueError(f'Unknown HIL RTC settings: {sorted(unknown)}')
    enabled = value.get('enabled', False)
    length = value.get('prefix_length', 4)
    if type(enabled) is not bool or type(length) is not int or not 0 <= length <= 8:
        raise ValueError('RTC requires boolean enabled and integer prefix_length in [0, 8]')
    if enabled:
        inf = raw.get('inference', {})
        if inf.get('policy_format') != 'canonical' or inf.get('canonical_dim', 32) != 32:
            raise ValueError('HIL RTC requires VTLA canonical 32D')
        if raw.get('hil', {}).get('policy_motion_limits', {}).get('enabled', False):
            raise ValueError('RTC requires policy_motion_limits disabled: clipping changes the committed prefix')
        for key in ('max_eef_xyz_step', 'max_eef_rot_step_deg', 'max_eef_gripper_step'):
            if inf.get(key, 0) > 0:
                raise ValueError(f'RTC requires {key}=0; retain rejecting safety checks')
    return enabled, length


def make_request(payload, prefix):
    state = np.asarray(payload.get('state'), dtype=np.float64)
    if state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError('RTC state must be finite canonical 32D')
    rows = np.asarray(prefix, dtype=np.float64)
    d = len(prefix)
    if d == 0:
        rows = np.empty((0, 32))
    if not 0 <= d <= 8 or rows.shape != (d, 32) or not np.isfinite(rows).all():
        raise ValueError('RTC prefix must be finite [d,32], d <= 8')
    request = dict(payload, cmd='predict_rtc', rtc=dict(prefix_length=d, prefix_actions=rows.tolist()))
    return deepcopy(request)


def validate_response(response, request):
    rtc = request['rtc']
    d = rtc['prefix_length']
    meta = response.get('rtc')
    if not isinstance(meta, dict) or meta.get('applied') is not True or meta.get('mode') != 'train_time_prefix':
        raise RuntimeError('RTC server did not confirm train_time_prefix; no fallback')
    if type(meta.get('prefix_length')) is not int or meta['prefix_length'] != d or meta.get('max_prefix_length') != 8:
        raise RuntimeError('RTC server prefix length/checkpoint mismatch')
    actions = np.asarray(response.get('actions'), dtype=np.float64)
    if actions.shape != (50, 32) or not np.isfinite(actions).all():
        raise RuntimeError('RTC server must return finite [50,32] actions')
    if d and not np.array_equal(actions[:d], np.asarray(rtc['prefix_actions'])):
        raise RuntimeError('RTC server modified submitted prefix')
