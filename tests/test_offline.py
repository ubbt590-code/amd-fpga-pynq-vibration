"""Regression tests for source-bound grouping, invalid input and portable model."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_core import load_recording,_groups
from signal_quality import assess_signal_quality
from diagnosis import Diagnoser

class OfflineTests(unittest.TestCase):
    def test_plot_does_not_hide_narrow_spectral_peak(self):
        from pages import plot_indices
        power=np.zeros(512);power[401]=100.
        indices=plot_indices(power)
        self.assertIn(401,indices)
        self.assertLessEqual(len(indices),320)

    def test_recording_and_groups(self):
        x,y,m=load_recording(str(ROOT/'dataset_test_3ch.npz'))
        groups=_groups(y,len(x),m['source_ids'])
        self.assertEqual(len(groups),262)
        self.assertEqual(m['dataset_id'],'cwru_resampled_20261023')
        for ids,truth in groups:
            self.assertEqual(len(set(m['source_ids'][ids])),1)
            self.assertTrue(np.all(y[ids]==truth))
            self.assertTrue(assess_signal_quality(x[ids])['usable'])
        # Two short sources must never be glued into a fictitious 8-frame group.
        self.assertEqual(_groups(np.zeros(8,dtype=int),8,np.array([0]*4+[1]*4)),[])

    def test_bad_signals(self):
        x,_,_=load_recording(str(ROOT/'dataset_test_3ch.npz'))
        base=x[:8].copy()
        for value in (0,12000,32767):
            self.assertFalse(assess_signal_quality(np.full_like(base,value))['usable'])
        base[:,1]=0
        self.assertFalse(assess_signal_quality(base)['usable'])
        self.assertFalse(assess_signal_quality(base.astype(float))['usable'])

    def test_wrong_sampling_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wrong.npz'
            np.savez(str(path),X=np.zeros((8,3,1024),dtype=np.int16),fs_hz=48000)
            with self.assertRaises(ValueError):load_recording(str(path))

    def test_held_out_model(self):
        features=np.load(str(ROOT/'training/group_features.npz'))
        self.assertFalse(set(features['train_source']) & set(features['test_source']))
        dg=Diagnoser(str(ROOT/'svm_model_multich.npz'),str(ROOT/'fir_coef_q15.npy'))
        pred,votes=dg.predict(features['test'])
        self.assertEqual(int(np.sum(pred==features['y_test'])),261)
        self.assertTrue(np.all((votes>=0)&(votes<=1)))

if __name__=='__main__':unittest.main()
