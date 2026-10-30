"""Portable comparator self-test; no FPGA hardware claim is made by this test."""
import json
from pathlib import Path
import numpy as np
from golden_compare import compare


def main():
    reference = np.array([0.0, -0.25, 0.5, 1.0])
    tests = {
        'exact_array': compare(reference, reference.copy())['passed'],
        'within_tolerance': compare(reference, reference + 1e-5, atol=2e-5)['passed'],
        'outside_tolerance': not compare(reference, reference + 0.1, atol=1e-4)['passed'],
        'shape_mismatch': compare(reference, reference[:2])['reason'] == 'shape_mismatch',
        'nonfinite': compare(reference, np.array([0., np.nan, 0., 0.]))['reason'] == 'nonfinite_input',
        'empty': compare([], [])['reason'] == 'empty_input',
        'large_integer': not compare(np.array([2**60]), np.array([2**60 + 1]))['passed'],
        'complex': compare(np.array([1+2j]), np.array([1+2j]))['passed'],
    }
    result = {'tested_on':'2026-10-23', 'tests':tests,
              'passed':all(tests.values()), 'fpga_validation':False,
              'cross_board_validation':False}
    Path(__file__).with_name('validation.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    assert result['passed']


if __name__ == '__main__':
    main()
