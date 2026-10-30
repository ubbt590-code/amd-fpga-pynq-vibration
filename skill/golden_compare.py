"""Compare same-shape numeric .npy arrays, including nonfinite and empty checks."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, actual, atol=0.0, rtol=0.0):
    a, b = np.asarray(reference), np.asarray(actual)
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError('Tolerances must be finite and nonnegative')
    result = {'reference_shape':list(a.shape), 'actual_shape':list(b.shape),
              'atol':float(atol), 'rtol':float(rtol), 'passed':False}
    if a.shape != b.shape:
        result['reason'] = 'shape_mismatch'
        return result
    if not a.size:
        result['reason'] = 'empty_input'
        return result
    if a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
        result['reason'] = 'nonnumeric_input'
        return result
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        result['reason'] = 'nonfinite_input'
        return result
    dtype = np.complex128 if a.dtype.kind == 'c' or b.dtype.kind == 'c' else np.float64
    # Converting integer arrays beyond 2**53 could conceal a one-count error.
    if any(v.dtype.kind in 'iu' and np.any(np.abs(v.astype(np.float64)) > 2**53)
           for v in (a, b)):
        result['reason'] = 'integer_precision_exceeds_float64'
        return result
    delta = np.abs(b.astype(dtype) - a.astype(dtype))
    tolerance = atol + rtol * np.abs(a.astype(dtype))
    result.update(max_abs_error=float(delta.max()),
                  rmse=float(np.sqrt(np.mean(delta**2))),
                  failed_values=int(np.sum(delta > tolerance)),
                  compared_values=int(a.size), passed=bool(np.all(delta <= tolerance)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('actual')
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = compare(np.load(args.reference, allow_pickle=False),
                     np.load(args.actual, allow_pickle=False), args.atol, args.rtol)
    payload = json.dumps(result, indent=2, allow_nan=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload, encoding='utf8')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
