# -*- coding: utf-8 -*-
"""Input validity guard, not an out-of-distribution or fault severity detector."""
import numpy as np

POLICY = {'minimum_ac_rms_codes': 8.0, 'maximum_rail_fraction': 0.02,
          'rail_code':32760, 'minimum_winning_votes':3}


def assess_signal_quality(group):
    x = np.asarray(group)
    result = {'usable':False, 'reasons':[], 'channels':[], 'policy':dict(POLICY)}
    if x.ndim != 3 or x.shape[1:] != (3, 1024) or len(x) < 1:
        result['reasons'] = ['invalid_shape']
        return result
    if x.dtype != np.int16:
        result['reasons'] = ['expected_int16_codes']
        return result
    for channel in range(3):
        values = x[:, channel].astype(np.float64)
        centered = values - values.mean(axis=1, keepdims=True)
        ac_rms = float(np.sqrt(np.mean(centered ** 2)))
        clip_fraction = float(np.mean(np.abs(values) >= POLICY['rail_code']))
        reasons = []
        if ac_rms < POLICY['minimum_ac_rms_codes']:
            reasons.append('no_or_very_low_ac_signal')
        if clip_fraction >= POLICY['maximum_rail_fraction']:
            reasons.append('excessive_clipping')
        result['channels'].append({'channel':channel, 'ac_rms_codes':ac_rms,
                                   'rail_fraction':clip_fraction, 'reasons':reasons})
        result['reasons'].extend(['ch%d:%s' % (channel, r) for r in reasons])
    result['usable'] = not result['reasons']
    return result
