# -*- coding: utf-8 -*-
"""
diagnosis.py — 故障诊断核心 (板上/PC 通用, 仅依赖 numpy)
- 特征向量: 13 维 = PL 提取的 10 维 + PS 包络谱 3 维
- 分类器: SVC-rbf 的纯 numpy 移植 (svm_model.npz)
- 多帧聚合: 默认 8 帧一组
"""
import numpy as np

FS = 12000.0
FRAME = 1024
ROTF = 29.0
BPFO, BPFI, BSF = 3.5848 * ROTF, 5.4152 * ROTF, 4.7135 * ROTF
CLASS_NAMES = ["正常", "内圈故障", "滚动体故障", "外圈故障", "转子不平衡"]


def _hilbert_env_bandpassed(x, f_lo=2500.0, f_hi=4500.0):
    """numpy 实现: FFT 域带通 + 解析信号包络 (等效 scipy filtfilt+hilbert 的包络提取)"""
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, 1 / FS)
    X[(np.abs(freqs) < f_lo) | (np.abs(freqs) > f_hi)] = 0.0   # 理想带通
    h = np.zeros(N); h[0] = 1.0
    h[1:N // 2] = 2.0                                          # 单边谱 -> 解析信号
    if N % 2 == 0:
        h[N // 2] = 1.0
    return np.abs(np.fft.ifft(X * h))


class Diagnoser:
    def __init__(self, model_path="svm_model.npz", coef_path="fir_coef_q15.npy",
                 q115_scale=3.6304):
        m = np.load(model_path)
        self.mean, self.scale_ = m["mean"], m["scale"]
        self.sv, self.dual = m["sv"], m["dual_coef"]
        self.intercept = m["intercept"]
        self.nsv = m["n_support"]
        self.classes = m["classes"]
        self.gamma = float(m["gamma"])
        self._sv_start = np.concatenate([[0], np.cumsum(self.nsv)])
        self.coef = np.load(coef_path).astype(np.float64) / 32768.0
        self.sc2 = q115_scale ** 2
        self._freqs512 = np.fft.rfftfreq(FRAME, 1 / FS)[:512]
        fft_freq = np.fft.fftfreq(FRAME, 1 / FS)
        self._env_keep = (np.abs(fft_freq) >= 2500.) & (np.abs(fft_freq) <= 4500.)
        self._analytic = np.zeros(FRAME)
        self._analytic[0] = self._analytic[FRAME // 2] = 1.
        self._analytic[1:FRAME // 2] = 2.
        self._hann = np.hanning(FRAME)
        self._score_masks = [(self._freqs512 >= f - 9) & (self._freqs512 <= f + 9)
                             for f in (BPFI, BPFO, BSF)]

    # ---------- PS 侧: 包络谱三分数 (numpy only) ----------
    def env_scores(self, x_raw_f32):
        """x_raw_f32: (1024,) 原始浮点帧(单位 g)"""
        yf = np.convolve(x_raw_f32, self.coef, mode="full")[:FRAME]  # 与 PL 同系数 FIR
        spectrum = np.fft.fft(yf)
        spectrum[~self._env_keep] = 0.
        e = np.abs(np.fft.ifft(spectrum * self._analytic))
        e = e - e.mean()
        E = np.abs(np.fft.rfft(e * self._hann))[:512]
        base = E[1:].mean() + 1e-12
        return tuple(float(E[mask].max() / base) for mask in self._score_masks)

    # ---------- 组帧 -> 13 维特征 ----------
    def feature_vector(self, pl_feats, raws_f32):
        """pl_feats: list of dict (vib_driver.run_frame 的返回); raws_f32: 对应原始帧
           返回 (13,) 特征向量"""
        n = len(pl_feats)
        rb = np.mean([f["band_ratio"] for f in pl_feats], axis=0)          # 6
        rms2 = np.mean([f["rms2"] for f in pl_feats])                       # 已归一化
        kurt = np.mean([f["kurtosis"] for f in pl_feats])
        pkb = np.mean([f["peak_hz"] / FS * FRAME for f in pl_feats]) / 512.0
        cent = np.mean([f["centroid_hz"] / FS * FRAME for f in pl_feats]) / 512.0
        env = np.mean([self.env_scores(r) for r in raws_f32], axis=0)       # 3
        return np.concatenate([rb, [np.log10(rms2 + 1e-9), kurt, pkb, cent], env])

    # ---------- 纯 numpy SVC-rbf 预测 ----------
    def predict(self, X):
        """X: (N,D) 原始特征 -> (pred[N], one-vs-one vote share[N])."""
        Xn = (X - self.mean) / self.scale_
        nc = len(self.classes)
        votes = np.zeros((len(Xn), nc), dtype=int)
        dec_sum = np.zeros((len(Xn), nc))
        pair = 0
        # Compute each support-vector kernel once, instead of redoing it for
        # each of the ten OVO pairs. Class votes and tie behavior are unchanged.
        kernel = np.exp(-self.gamma * ((Xn[:, None, :] - self.sv[None, :, :]) ** 2).sum(-1))
        for i in range(nc):
            for j in range(i + 1, nc):
                svi = self.sv[self._sv_start[i]:self._sv_start[i + 1]]
                svj = self.sv[self._sv_start[j]:self._sv_start[j + 1]]
                ci = self.dual[j - 1][self._sv_start[i]:self._sv_start[i + 1]]
                cj = self.dual[i][self._sv_start[j]:self._sv_start[j + 1]]
                K = np.concatenate([kernel[:, self._sv_start[i]:self._sv_start[i + 1]],
                                    kernel[:, self._sv_start[j]:self._sv_start[j + 1]]], axis=1)
                dec = K @ np.concatenate([ci, cj]) + self.intercept[pair]
                votes[dec > 0, i] += 1;  dec_sum[:, i] += np.abs(dec) * (dec > 0)
                votes[dec <= 0, j] += 1; dec_sum[:, j] += np.abs(dec) * (dec <= 0)
                pair += 1
        pred = self.classes[np.argmax(votes, axis=1)]
        # This is a vote share, not a calibrated probability.
        conf = votes.max(axis=1) / (nc * (nc - 1) / 2)
        return pred, conf

    # ---------- 顶层: 一组帧 -> 诊断结果 ----------
    def diagnose(self, pl_feats, raws_f32):
        fv = self.feature_vector(pl_feats, raws_f32)
        pred, conf = self.predict(fv[None, :])
        return int(pred[0]), float(conf[0]), fv

    def feature_vector_multichannel(self, pl_feats_by_channel, raws_by_channel):
        """Concatenate 13 features per channel in the supplied channel order."""
        if len(pl_feats_by_channel) != len(raws_by_channel) or not pl_feats_by_channel:
            raise ValueError("provide matching, non-empty feature and raw channel lists")
        return np.concatenate([
            self.feature_vector(pl_feats, raws)
            for pl_feats, raws in zip(pl_feats_by_channel, raws_by_channel)
        ])

    def diagnose_multichannel(self, pl_feats_by_channel, raws_by_channel):
        fv = self.feature_vector_multichannel(pl_feats_by_channel, raws_by_channel)
        pred, vote_share = self.predict(fv[None, :])
        return int(pred[0]), float(vote_share[0]), fv
