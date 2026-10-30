# -*- coding: utf-8 -*-
"""Run with the HDMI service stopped; save real board results and page buffers."""
import json
from pathlib import Path
import numpy as np
from pynq.lib.video import VideoMode
from pynq.lib.video.hierarchies import VideoOut
from vib_driver import VibAccel
from engine import Engine
from analysis_core import software_frame
from pages import render
from app import save_json


def main():
    root = Path(__file__).resolve().parent
    import os
    os.chdir(str(root))
    out = root / 'evidence'
    out.mkdir(exist_ok=True)
    acc = VibAccel('vib_accel.bit', 'fir_coef_q15.npy')
    e = Engine(acc, 'demo_set.npz', 'svm_model_multich.npz', 'fir_coef_q15.npy')
    checks = []
    try:
        for condition in range(len(e.classes)):
            sample = e.frame_job(condition, 0)
            for c, feature in enumerate(sample['features']):
                expected = np.mean((sample['frames'][c].astype(float) / 32768) ** 2)
                assert abs(feature['rms2'] - expected) < 1e-12
                assert abs(sum(feature['band_ratio']) - 1) < 1e-10
                assert len(feature['spectrum']) == 512 and np.any(feature['spectrum'])
                assert int(feature['peak_hz'] * 1024 / 12000) == int(np.argmax(feature['spectrum'][1:]) + 1)
            result = e.diagnosis_job(condition, 0)
            assert result['prediction'] == e.classes[condition], result
            assert 0 <= result['confidence'] <= 1
            checks.append({'class': e.classes[condition], 'diagnosis': result})
            print('PASS class', e.classes[condition], result['prediction_name'], flush=True)
        zero = acc.run_frame(np.zeros(1024, dtype=np.int16))
        assert zero['rms2'] == 0 and not np.any(zero['spectrum']) and zero['peak_abs'] == 0
        filt = e.filter_job(0)
        assert filt['attenuation_db'] > 25, filt['attenuation_db']
        taps = []
        for i in range(32):
            word = acc.pre.read(0x80 + 4 * i)
            taps.extend([word & 65535, (word >> 16) & 65535])
        assert np.array_equal(np.asarray(taps, dtype=np.uint16).view(np.int16), e.coef)
        print('PASS FIR suppression %.2f dB and coefficient restoration' % filt['attenuation_db'], flush=True)
        accuracy = e.accuracy_job(progress=lambda d,t: print('Evaluation %d/%d' % (d,t),flush=True) if d%5==0 else None)
        assert accuracy['groups'] == len(e.groups)
        assert accuracy['pl_ps_match'] == accuracy['groups']
        print('Measured accuracy %d/%d; all PL/PS predictions agree' %
              (accuracy['correct'],accuracy['groups']), flush=True)
        performance = e.performance_job()
        state = {'frame_count':len(e.data), 'source':e.source, 'sample':e.frame_job(1,0),
                 'filter':filt, 'diagnosis':checks[1]['diagnosis'], 'accuracy':accuracy,
                 'performance':performance,
                 'resources':json.loads(Path('resources.json').read_text(encoding='utf8'))}
        for page in range(8):
            render(page,state).save(str(out/('page_%02d.png' % (page+1))))
        for name, value in [('filter',filt),('accuracy',accuracy),('performance',performance),
                            ('diagnosis',checks[1]['diagnosis'])]:
            save_json(out/(name+'.json'),value)
        summary = {'passed':True, 'conditions':checks, 'zero_input':True,
                   'fir_attenuation_db':filt['attenuation_db'], 'coefficients_restored':True,
                   'accuracy':accuracy, 'performance':performance, 'frames_completed':acc.frames_completed}
        save_json(out/'board_validation.json',summary)
        print(json.dumps(summary,ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)),flush=True)
    finally:
        acc.close()


if __name__ == '__main__':
    main()
