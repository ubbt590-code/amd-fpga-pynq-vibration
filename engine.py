# -*- coding: utf-8 -*-
"""Board analysis jobs. Hardware access is serialized by the console worker."""
import time
import numpy as np
from analysis_core import (FS, FRAME, GROUP_SIZE, load_recording, software_frame,
                           diagnose_group, _groups, run_performance)
from diagnosis import CLASS_NAMES, Diagnoser
from signal_quality import assess_signal_quality, POLICY

CHANNELS = ['DE 驱动端', 'FE 风扇端', 'BA 基座']


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled()


def fixed_fir(frame, coefficients):
    """Integer reference for HLS ap_fixed AP_TRN/AP_WRAP FIR (zero initial state)."""
    acc = np.convolve(frame.astype(np.int64), coefficients.astype(np.int64))[:FRAME]
    return (acc >> 15).astype(np.int16)


class Engine:
    def __init__(self, accelerator, demo, model, coef):
        self.acc = accelerator
        self.data, self.labels, self.metadata = load_recording(demo)
        self.dg = Diagnoser(model, coef)
        self.coef = np.load(coef).astype(np.int16)
        self.source = demo
        self.classes = (sorted(set(int(v) for v in self.labels))
                        if self.labels is not None else [None])
        if any(c is not None and (c < 0 or c >= 5) for c in self.classes):
            raise ValueError('数据标签必须在 0..4 之间')
        self.class_ids = {c: (np.where(self.labels == c)[0] if c is not None
                              else np.arange(len(self.data))) for c in self.classes}
        self.source_ids = self.metadata['source_ids']
        self.groups = _groups(self.labels, len(self.data), self.source_ids)

    def provenance_note(self, cls):
        if self.metadata['dataset_id'] not in ('cwru_legacy_20261019','cwru_resampled_20261023'):
            return '文件回放；源测点元数据未提供'
        if cls == 0:
            return '正常：DE/FE 实测；第三路由 DE 移位派生'
        if cls == 4:
            return '不平衡：三路均为合成信号'
        return 'CWRU 人工缺陷；DE/FE/BA 文件数据回放'

    def sample(self, condition, offset):
        cls = self.classes[condition % len(self.classes)]
        ids = self.class_ids[cls]
        index = int(ids[offset % len(ids)])
        return {'index': index, 'class': cls,
                'class_name': CLASS_NAMES[cls] if cls is not None else '未标注输入',
                'provenance': self.provenance_note(cls),
                'waveforms': self.data[index].astype(np.float64) / 32768.0,
                'frames': self.data[index]}

    def frame_job(self, condition, offset, cancel=None):
        sample = self.sample(condition, offset)
        feats = []
        started = time.perf_counter()
        for frame in sample['frames']:
            check_cancel(cancel)
            result=self.acc.run_frame(frame)
            # This display annotation stays outside the timed classifier/DMA path.
            top=np.partition(result['spectrum'][1:],-2)[-2:].astype(float)
            margin=10*np.log10(top.max()/max(top.min(),1.0)) if top.max()>0 else 0.
            result.update(peak_margin_db=float(margin),peak_ambiguous=bool(margin<1.))
            feats.append(result)
        sample['elapsed_ms'] = (time.perf_counter() - started) * 1000
        sample['features'] = feats
        return sample

    def _frame_features(self, frames, cancel=None):
        feats = []
        for frame in np.asarray(frames):
            check_cancel(cancel)
            result = self.acc.run_frame(frame)
            top = np.partition(result['spectrum'][1:], -2)[-2:].astype(float)
            margin = 10 * np.log10(top.max() / max(top.min(), 1.0)) if top.max() > 0 else 0.
            result.update(peak_margin_db=float(margin), peak_ambiguous=bool(margin < 1.))
            feats.append(result)
        return feats

    def live_frame_job(self, frames, site_id, sequence, cancel=None):
        """One gateway packet -> the same three PL feature paths as file replay."""
        values = np.asarray(frames)
        if values.shape != (3, FRAME) or values.dtype != np.int16:
            raise ValueError('gateway data must be (3,1024) int16')
        started = time.perf_counter()
        sample = {'index': int(sequence), 'class': None, 'class_name': '网关实时输入',
                  'provenance': 'PC 多站点网关 · %s · 当前为回放/仿真输入' % site_id,
                  'waveforms': values.astype(np.float64) / 32768.0,
                  'frames': values, 'features': self._frame_features(values, cancel)}
        sample['elapsed_ms'] = (time.perf_counter() - started) * 1000
        sample['site_id'] = str(site_id)
        sample['sequence'] = int(sequence)
        return sample

    def live_diagnosis_job(self, group, site_id, sequence, cancel=None):
        values = np.asarray(group)
        if values.ndim != 3 or values.shape[1:] != (3, FRAME) or len(values) < GROUP_SIZE:
            raise ValueError('gateway diagnosis requires eight packets of (3,1024) int16')
        result = self.diagnose_data(values[-GROUP_SIZE:], use_pl=True, cancel=cancel)
        result.update({'class': None, 'class_name': '网关实时输入',
                       'provenance': 'PC 多站点网关 · %s · 八帧实时组' % site_id,
                       'first_index': int(sequence) - GROUP_SIZE + 1,
                       'last_index': int(sequence), 'site_id': str(site_id),
                       'sequence': int(sequence)})
        return result

    def filter_job(self, mode, cancel=None, sweep_index=0):
        """Two PL runs prove filtering using common FFT scaling, then restore taps."""
        t = np.arange(FRAME) / FS
        # Integer FFT-bin tones isolate attenuation from spectral leakage.
        low_hz, high_hz = 52 * FS / FRAME, 469 * FS / FRAME
        phase = 2 * np.pi * (int(sweep_index) % 16) / 16.0
        x = np.round((0.24 * np.sin(2 * np.pi * low_hz * t + phase) +
                      0.24 * np.sin(2 * np.pi * high_hz * t + phase)) * 32768).astype(np.int16)
        if mode % 2:
            x = self.sample(1 % len(self.classes), 0)['frames'][0]
            source = 'CWRU 存储样本 · DE 通道'
        else:
            source = '双频验证信号：%.1f Hz + %.1f Hz（PS 生成）' % (low_hz, high_hz)
        bypass = np.zeros(64, dtype=np.int16)
        bypass[0] = 32767
        check_cancel(cancel)
        try:
            self.acc.load_coefs(bypass)
            before = self.acc.run_frame(x)
        finally:
            self.acc.load_coefs(self.coef)
        check_cancel(cancel)
        after = self.acc.run_frame(x)
        band = slice(450, 500)
        p_in = float(np.sum(before['spectrum'][band], dtype=np.float64))
        p_out = float(np.sum(after['spectrum'][band], dtype=np.float64))
        attenuation = 10 * np.log10(max(p_in, 1.0) / max(p_out, 1.0))
        return {'source': source, 'raw': x.astype(np.float64) / 32768.0,
                'filtered_ref': fixed_fir(x, self.coef).astype(np.float64) / 32768.0,
                'before': before, 'after': after, 'attenuation_db': float(attenuation),
                'low_hz': low_hz, 'high_hz': high_hz,
                'coefficient_writes': 32, 'taps_restored': True,
                'sweep_index': int(sweep_index)}

    def diagnosis_job(self, condition, offset, cancel=None):
        cls = self.classes[condition % len(self.classes)]
        choices = [ids for ids, truth in self.groups if truth == cls]
        if not choices:
            raise ValueError('当前工况不足 8 帧，无法进行三通道融合诊断')
        chosen = choices[offset % len(choices)]
        result = self.diagnose_data(self.data[chosen], use_pl=True, cancel=cancel)
        result.update({'class':cls,
                       'class_name': CLASS_NAMES[cls] if cls is not None else '未标注输入',
                       'first_index':int(chosen[0]), 'last_index':int(chosen[-1]),
                       'provenance':self.provenance_note(cls)})
        return result

    def diagnose_data(self, frames, use_pl=True, cancel=None):
        check_cancel(cancel)
        t0 = time.perf_counter()
        quality = assess_signal_quality(frames)
        if not quality['usable']:
            return {'accepted':False, 'prediction':None, 'prediction_name':'信号质量不足',
                    'confidence':0., 'votes':0, 'votes_possible':4, 'channels':[],
                    'quality':quality, 'rejection_reason':'; '.join(quality['reasons']),
                    'elapsed_ms':(time.perf_counter()-t0)*1000}
        prediction, vote_share, rows = diagnose_group(
            self.acc, self.dg, frames, np.arange(len(frames)), use_pl)
        pair_count = len(self.dg.classes) * (len(self.dg.classes) - 1) // 2
        votes = int(round(vote_share * pair_count))
        accepted = votes >= POLICY['minimum_winning_votes']
        return {'accepted':accepted, 'quality':quality,
                'prediction': prediction if accepted else None,
                'candidate_prediction':prediction,
                'prediction_name': CLASS_NAMES[prediction] if accepted else '分类证据不足',
                'rejection_reason':'' if accepted else 'OVO winning votes below policy',
                'votes': votes, 'votes_possible': len(self.dg.classes) - 1,
                'confidence': votes / float(len(self.dg.classes) - 1),
                'vote_share_all_pairs': vote_share, 'channels': rows,
                'elapsed_ms': (time.perf_counter() - t0) * 1000}

    def accuracy_job(self, cancel=None, progress=None):
        if self.labels is None:
            raise ValueError('评估需要带真实类别标签的数据')
        groups = self.groups
        if not groups:
            raise ValueError('数据中没有完整的 8 帧评估组')
        matrix = np.zeros((5, 5), dtype=np.int64)
        software_matrix = np.zeros_like(matrix)
        matched = 0
        rejected = 0
        software_rejected = 0
        records = []
        t0 = time.perf_counter()
        for i, (ids, truth) in enumerate(groups):
            check_cancel(cancel)
            pl_result = self.diagnose_data(self.data[ids], True, cancel)
            ps_result = self.diagnose_data(self.data[ids], False, cancel)
            predicted, software = pl_result['prediction'], ps_result['prediction']
            if predicted is None:
                rejected += 1
            else:
                matrix[int(truth), predicted] += 1
            if software is None:
                software_rejected += 1
            else:
                software_matrix[int(truth), software] += 1
            matched += int(predicted == software)
            records.append({'group':i, 'truth':int(truth), 'prediction':predicted,
                            'software_prediction':software,
                            'source':str(self.source_ids[ids[0]]) if self.source_ids is not None else None,
                            'first_index':int(ids[0]), 'last_index':int(ids[-1]),
                            'pl_ms':pl_result['elapsed_ms'], 'ps_ms':ps_result['elapsed_ms'],
                            'accepted':pl_result['accepted']})
            if progress:
                progress(i + 1, len(groups))
        return {'confusion': matrix, 'software_confusion': software_matrix,
                'rejected':rejected, 'software_rejected':software_rejected,
                'coverage':float(len(groups)-rejected)/len(groups),
                'records':records, 'file_bound_grouping':self.source_ids is not None,
                'correct': int(np.trace(matrix)), 'groups': len(groups),
                'accuracy': float(np.trace(matrix)) / len(groups),
                'software_accuracy': float(np.trace(software_matrix)) / len(groups),
                'pl_ps_match': matched, 'group_size': GROUP_SIZE,
                'elapsed_s': time.perf_counter() - t0,
                'evaluated_frames': len(groups) * GROUP_SIZE,
                'dropped_frames': len(self.data) - len(groups) * GROUP_SIZE}

    def performance_job(self, cancel=None, progress=None):
        # Measure with rendering quiet, using the same baseline features as the model.
        check_cancel(cancel)
        if progress:
            progress(0, 1)
        result = run_performance(self.acc, self.dg, self.data, self.labels, self.source_ids)
        result['frame_speedup'] = result['frame_ps_us'] / result['frame_pl_us']
        result['group_speedup'] = result['group_ps_ms'] / result['group_pl_ms']
        result['pl_samples_s'] = FRAME * 1e6 / result['frame_pl_us']
        result['ps_samples_s'] = FRAME * 1e6 / result['frame_ps_us']
        return result
