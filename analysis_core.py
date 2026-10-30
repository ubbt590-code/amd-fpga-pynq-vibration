# -*- coding: utf-8 -*-
"""Local AX7020 HDMI demonstration console for PYNQ 2.4."""

from __future__ import print_function

import argparse
import time

import numpy as np

from diagnosis import CLASS_NAMES, Diagnoser



FS = 12000.0
FRAME = 1024
GROUP_SIZE = 8
BAND_LO = [1, 43, 86, 171, 257, 342]
BAND_HI = [42, 85, 170, 256, 341, 511]


def load_recording(path):
    archive = np.load(path, allow_pickle=False)
    full_scale = 3.6304
    metadata = {'fs_hz': FS, 'frame_samples': FRAME, 'hop_samples': 512,
                'sampling_rate_declared': False, 'source_ids': None, 'dataset_id':'unspecified'}
    try:
        if isinstance(archive, np.ndarray):
            x = archive
            labels = None
        else:
            keys = list(archive.files)
            key = next((name for name in ("X", "X_test_i16", "X_test_f32", "X_test")
                        if name in keys), None)
            if key is None:
                raise ValueError("NPZ 中没有找到三通道数据数组")
            x = archive[key]
            label_key = next((name for name in ("y", "y_test", "labels") if name in keys), None)
            labels = np.asarray(archive[label_key]).reshape(-1) if label_key else None
            scale_key = next((name for name in ("q115_scale", "full_scale_g", "scale")
                              if name in keys), None)
            full_scale = float(np.asarray(archive[scale_key]).reshape(-1)[0]) if scale_key else 3.6304
            if 'group_id' in keys:
                metadata['source_ids'] = np.asarray(archive['group_id']).copy()
            if 'dataset_id' in keys:
                metadata['dataset_id'] = str(np.asarray(archive['dataset_id']).item())
            if 'fs_hz' in keys:
                metadata['fs_hz'] = float(np.asarray(archive['fs_hz']).item())
                metadata['sampling_rate_declared'] = True
            if 'hop_samples' in keys:
                metadata['hop_samples'] = int(np.asarray(archive['hop_samples']).item())
    finally:
        if hasattr(archive, "close"):
            archive.close()

    x = np.asarray(x)
    if x.ndim == 2 and x.shape == (3, FRAME):
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1:] != (3, FRAME) or len(x) < 1:
        raise ValueError("输入形状必须是 (N, 3, 1024)，当前为 %s" % (x.shape,))
    if np.issubdtype(x.dtype, np.integer):
        if x.min() < -32768 or x.max() > 32767:
            raise ValueError("整数样本必须在 int16 / Q1.15 范围内")
        x = x.astype(np.int16, copy=False)
    else:
        if not np.isfinite(x).all() or not np.isfinite(full_scale) or full_scale <= 0:
            raise ValueError("浮点样本或满量程无效")
        x = (np.clip(x.astype(np.float64) / full_scale, -1.0, 0.9999694824) * 32768.0).astype(np.int16)
    if labels is not None:
        if len(labels) != len(x):
            raise ValueError("标签数量必须与输入帧数一致")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("类别标签必须使用整数 0..4")
        if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
            raise ValueError("类别标签必须在 0..4 之间")
    origins = metadata['source_ids']
    if origins is not None:
        if origins.shape != (len(x),) or origins.dtype.kind not in 'iuUS':
            raise ValueError('group_id 必须逐帧提供整数或字符串源文件标识')
    if metadata['fs_hz'] != FS:
        raise ValueError('当前系数/模型只支持 12 kHz；不能仅修改采样率标签')
    if not 1 <= metadata['hop_samples'] <= FRAME:
        raise ValueError('hop_samples 必须在 1..1024 之间')
    metadata['full_scale'] = full_scale
    return x, labels, metadata


def load_demo(path):
    data, labels, _ = load_recording(path)
    return data, labels


def software_frame(frame, dg):
    """NumPy FIR/FFT feature baseline, matching the board notebook metric."""
    xq = frame.astype(np.float64) / 32768.0
    yf = np.convolve(xq, dg.coef, "full")[:FRAME]
    power = np.abs(np.fft.rfft(yf))[:512] ** 2
    total = float(power[1:].sum())
    if total <= 1e-18:
        total = 1.0
    bands = [float(power[lo:hi + 1].sum() / total)
             for lo, hi in zip(BAND_LO, BAND_HI)]
    bins = np.arange(1, 512, dtype=np.float64)
    centroid = float(np.dot(bins, power[1:]) / total * FS / FRAME)
    peak_bin = int(np.argmax(power[1:]) + 1)
    rms2 = float(np.mean(xq ** 2))
    kurtosis = float(np.mean(xq ** 4) / (rms2 * rms2)) if rms2 > 1e-12 else 0.0
    return {
        "rms2": rms2,
        "kurtosis": kurtosis,
        "peak_abs": float(np.max(np.abs(xq))),
        "band_ratio": bands,
        "peak_hz": peak_bin * FS / FRAME,
        "centroid_hz": centroid,
        "spectrum": power,
    }


def _groups(labels, count, source_ids=None):
    """Never join nonadjacent samples, source recordings or labels into one group."""
    result = []
    if labels is not None and len(labels) != count:
        raise ValueError('Label count mismatch')
    if source_ids is not None and len(source_ids) != count:
        raise ValueError('Source count mismatch')
    start = 0
    while start < count:
        end = start + 1
        while end < count and (labels is None or labels[end] == labels[start]) and (
                source_ids is None or source_ids[end] == source_ids[start]):
            end += 1
        for first in range(start, end - GROUP_SIZE + 1, GROUP_SIZE):
            result.append((np.arange(first, first + GROUP_SIZE),
                           int(labels[start]) if labels is not None else None))
        start = end
    return result


def _aggregate(features):
    return {
        "rms2": float(np.mean([f["rms2"] for f in features])),
        "kurtosis": float(np.mean([f["kurtosis"] for f in features])),
        "peak_hz": float(np.mean([f["peak_hz"] for f in features])),
        "centroid_hz": float(np.mean([f["centroid_hz"] for f in features])),
        "band_ratio": np.mean([f["band_ratio"] for f in features], axis=0).tolist(),
    }


def diagnose_group(acc, dg, data, ids, use_pl=True):
    features_by_channel = []
    raws_by_channel = []
    rows = []
    for channel in range(3):
        frames = [data[int(index), channel] for index in ids]
        if use_pl:
            features = [acc.run_frame(frame) for frame in frames]
        else:
            features = [software_frame(frame, dg) for frame in frames]
        features_by_channel.append(features)
        raws_by_channel.append([frame.astype(np.float64) / 32768.0 for frame in frames])
        rows.append(_aggregate(features))
    prediction, vote_share, _ = dg.diagnose_multichannel(features_by_channel, raws_by_channel)
    return int(prediction), float(vote_share), rows


def run_accuracy(acc, dg, data, labels, source_ids=None):
    groups = _groups(labels, len(data), source_ids)
    if labels is None:
        raise ValueError("输入数据没有类别标签，无法生成混淆矩阵")
    if not groups:
        raise ValueError("数据中没有完整的 8 帧评估组")
    matrix = np.zeros((5, 5), dtype=np.int64)
    for ids, truth in groups:
        predicted, _, _ = diagnose_group(acc, dg, data, ids, True)
        matrix[int(truth), predicted] += 1
    correct = int(np.trace(matrix))
    return {"confusion": matrix, "correct": correct, "groups": len(groups),
            "accuracy": float(correct) / len(groups), "group_size": GROUP_SIZE}


def run_performance(acc, dg, data, labels, source_ids=None):
    frame_count = min(40, len(data))
    frame_ids = np.linspace(0, len(data) - 1, frame_count, dtype=np.int64)
    first = data[int(frame_ids[0]), 0]
    acc.run_frame(first)
    software_frame(first, dg)
    pl_times, ps_times = [], []
    rounds = 3
    for _ in range(rounds):
        t0 = time.perf_counter()
        for index in frame_ids:
            acc.run_frame(data[int(index), 0])
        pl_times.append((time.perf_counter() - t0) / frame_count)
        t0 = time.perf_counter()
        for index in frame_ids:
            software_frame(data[int(index), 0], dg)
        ps_times.append((time.perf_counter() - t0) / frame_count)

    groups = _groups(labels, len(data), source_ids)
    if not groups:
        groups = _groups(None, len(data), source_ids)
    group_count = min(5, len(groups))
    if not group_count:
        raise ValueError('性能测试至少需要完整的 8 帧组')
    group_ids = np.linspace(0, len(groups) - 1, group_count, dtype=np.int64)
    chosen = [groups[int(index)][0] for index in group_ids]
    diagnose_group(acc, dg, data, chosen[0], True)
    diagnose_group(acc, dg, data, chosen[0], False)
    pl_group_times, ps_group_times = [], []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for ids in chosen:
            diagnose_group(acc, dg, data, ids, True)
        pl_group_times.append((time.perf_counter() - t0) / group_count)
        t0 = time.perf_counter()
        for ids in chosen:
            diagnose_group(acc, dg, data, ids, False)
        ps_group_times.append((time.perf_counter() - t0) / group_count)
    return {
        "frame_pl_us": float(np.median(pl_times)) * 1e6,
        "frame_ps_us": float(np.median(ps_times)) * 1e6,
        "group_pl_ms": float(np.median(pl_group_times)) * 1e3,
        "group_ps_ms": float(np.median(ps_group_times)) * 1e3,
        "frame_samples": frame_count,
        "group_samples": group_count,
        "group_size": GROUP_SIZE,
        "rounds": rounds,
    }


