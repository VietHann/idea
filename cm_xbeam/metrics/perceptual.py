"""Complementary intelligibility and legacy WB-PESQ with explicit missing values."""
import math
import warnings
import numpy as np
from scipy.signal import resample_poly


def perceptual_scores(estimate, target, sample_rate):
    try:
        from pystoi import stoi
        from pesq import pesq
    except ImportError as exc:
        raise ImportError('Perceptual evaluation requires pystoi and pesq; use --no-perceptual to opt out explicitly') from exc
    result, errors = {'stoi': None, 'wb_pesq': None}, {}
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            value = float(stoi(target, estimate, sample_rate, extended=False))
        if notices:
            errors['stoi'] = '; '.join(str(x.message) for x in notices)
        elif np.isfinite(value):
            result['stoi'] = value
        else:
            errors['stoi'] = 'nonfinite_metric'
    except (ValueError, RuntimeError) as exc:
        errors['stoi'] = str(exc)
    try:
        common = math.gcd(int(sample_rate), 16000)
        if sample_rate != 16000:
            target = resample_poly(target, 16000 // common, sample_rate // common)
            estimate = resample_poly(estimate, 16000 // common, sample_rate // common)
        value = float(pesq(16000, target, estimate, 'wb'))
        if np.isfinite(value):
            result['wb_pesq'] = value
        else:
            errors['wb_pesq'] = 'nonfinite_metric'
    except Exception as exc:
        # PESQ's extension uses several distinct exception classes for short,
        # silent, or otherwise invalid utterances. Preserve the actual reason.
        errors['wb_pesq'] = f'{type(exc).__name__}: {exc}'
    if errors:
        result['metric_errors'] = errors
    return result
