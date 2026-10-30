# -*- coding: utf-8 -*-
"""Eight independent HDMI pages, rendered by PS Python (Pillow 5 compatible)."""
import os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from engine import CHANNELS, CLASS_NAMES

WIDTH, HEIGHT = 1280, 720
PAGES = ['数据输入', 'FIR 滤波', 'FFT 频谱', '特征提取', '故障识别', '识别评估', '性能对比', '系统配置']
BG, PANEL, EDGE = '#101d25', '#192a35', '#344854'
FG, DIM = '#eaf3f5', '#a1b5c0'
GREEN, BLUE, GOLD, RED = '#47d8b0', '#67b7ed', '#f0bd64', '#f08b85'
COLORS = [BLUE, GREEN, GOLD]
FONTS = {}
BANDS = ['12–492 Hz', '504–996 Hz', '1.01–1.99 kHz', '2.00–3.00 kHz', '3.01–4.00 kHz', '4.01–5.99 kHz']


def font(size, latin=False):
    key = (size, latin)
    if key not in FONTS:
        paths = (['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'] if latin
                 else ['/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 'C:/Windows/Fonts/msyh.ttc'])
        for path in paths:
            if os.path.exists(path):
                FONTS[key] = ImageFont.truetype(path, size)
                break
        else:
            FONTS[key] = ImageFont.load_default()
    return FONTS[key]


def font_runs(value):
    value = str(value)
    if not value:
        return []
    runs = []
    current, latin = '', ord(value[0]) < 0x2e80
    for ch in value:
        is_latin = ord(ch) < 0x2e80
        if is_latin != latin:
            runs.append((current, latin))
            current, latin = '', is_latin
        current += ch
    runs.append((current, latin))
    return runs


@lru_cache(maxsize=4096)
def string_width(value, size, latin):
    f = font(size, latin)
    if hasattr(f, 'getsize'):
        return f.getsize(value)[0]
    box = f.getbbox(value)
    return box[2] - box[0]


@lru_cache(maxsize=1024)
def text_bitmap(value, size, color):
    runs = font_runs(value)
    width = sum(string_width(part, size, latin) for part, latin in runs)
    bitmap = Image.new('RGBA', (max(1, width + 2), size * 2 + 4))
    draw = ImageDraw.Draw(bitmap)
    x = 0
    for part, latin in runs:
        draw.text((x, 0), part, font=font(size, latin), fill=color)
        x += string_width(part, size, latin)
    return bitmap


class Canvas:
    """Draw directly in the HDMI safe area instead of resampling a full frame."""
    def __init__(self, image, margin):
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.dx, self.dy = int(WIDTH * margin), int(HEIGHT * margin)
        self.sx = (WIDTH - 2 * self.dx) / float(WIDTH)
        self.sy = (HEIGHT - 2 * self.dy) / float(HEIGHT)
        self.scale = min(self.sx, self.sy)

    def point(self, x, y):
        return int(self.dx + x * self.sx), int(self.dy + y * self.sy)

    def rectangle(self, box, **kwargs):
        self.draw.rectangle(self.point(box[0], box[1]) + self.point(box[2], box[3]), **kwargs)

    def line(self, points, fill=None, width=1):
        if isinstance(points[0], (tuple, list)):
            transformed = [self.point(x, y) for x, y in points]
        else:
            transformed = [self.point(points[i], points[i+1]) for i in range(0, len(points), 2)]
        self.draw.line(transformed, fill=fill, width=max(1, int(round(width * self.scale))))

    def polygon(self, points, **kwargs):
        self.draw.polygon([self.point(x, y) for x, y in points], **kwargs)


def text(d, x, y, value, size=18, color=FG):
    bitmap = text_bitmap(str(value), max(1, int(round(size * d.scale))), color)
    d.image.paste(bitmap, d.point(x, y), bitmap)


def measure(d, value, size):
    pixels = max(1, int(round(size * d.scale)))
    return sum(string_width(part, pixels, latin) for part, latin in font_runs(value)) / d.scale


def fit(d, x, y, value, width, size=17, color=FG):
    value = str(value)
    while size > 11 and measure(d, value, size) > width:
        size -= 1
    while measure(d, value, size) > width and len(value) > 2:
        value = value[:-2] + '…'
    text(d, x, y, value, size, color)


def panel(d, box, edge=EDGE, fill=PANEL):
    d.rectangle(box, outline=edge, fill=fill)


def metric(d, x, y, label, value, color=GREEN, width=270, note=''):
    panel(d, (x, y, x + width, y + 95))
    text(d, x + 16, y + 10, label, 15, DIM)
    fit(d, x + 16, y + 32, value, width - 30, 27, color)
    if note:
        fit(d, x + 16, y + 72, note, width - 30, 12, DIM)


def plot_indices(values, limit=320):
    """Preserve local extrema and their x positions when plotting dense data."""
    values=np.asarray(values)
    if len(values)<=limit:return np.arange(len(values))
    edges=np.linspace(0,len(values),limit//2+1,dtype=int)
    result=[]
    for first,last in zip(edges[:-1],edges[1:]):
        part=values[first:last]
        result.extend(sorted({first+int(np.argmin(part)),first+int(np.argmax(part))}))
    return np.asarray(result,dtype=int)


def chart(d, box, series, title, xmax, xlabel='ms', bounds=None, marks=None):
    """Shared y scale and actual axis ticks; series: (values, color)."""
    x0, y0, x1, y1 = box
    panel(d, box)
    text(d, x0 + 12, y0 + 8, title, 16)
    left, right, top, bottom = x0 + 68, x1 - 18, y0 + 37, y1 - 30
    arrays = [(np.asarray(v, dtype=float).reshape(-1), c) for v, c in series]
    if bounds is None:
        all_values = np.concatenate([a for a, c in arrays])
        lo, hi = float(all_values.min()), float(all_values.max())
        if hi - lo < 1e-12:
            lo, hi = lo - 1, hi + 1
        pad = (hi - lo) * .08
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = bounds
    for i in range(3):
        yy = bottom - (bottom - top) * i / 2.0
        d.line((left, yy, right, yy), fill=EDGE)
        tick = lo + (hi - lo) * i / 2.0
        text(d, x0 + 7, yy - 6, ('%.3f' if abs(hi) < 2 else '%.0f') % tick, 11, DIM)
    for i in range(5):
        xx = left + (right - left) * i / 4.0
        d.line((xx, top, xx, bottom), fill=EDGE)
        tick = xmax * i / 4.0
        text(d, xx - 10, bottom + 7, '%.0f' % tick, 11, DIM)
    text(d, right - 4, y0 + 10, xlabel, 11, DIM)
    for arr, color in arrays:
        if len(arr) < 2:
            continue
        indices=plot_indices(arr)
        yy = bottom - (np.clip(arr[indices], lo, hi) - lo) / (hi - lo) * (bottom - top)
        xx = left+indices/float(len(arr)-1)*(right-left)
        d.line([(int(x), int(y)) for x, y in zip(xx, yy)], fill=color, width=2)
    for frequency, label in (marks or []):
        xx = left + frequency / xmax * (right - left)
        d.line((xx, top, xx, bottom), fill=GOLD)
        fit(d, min(xx + 4, right - 75), top + 4, label, 75, 11, GOLD)


def relative_db(power, reference=None):
    p = np.asarray(power, dtype=float)
    ref = max(float(np.max(p)) if reference is None else float(reference), 1.0)
    return np.maximum(-80, 10 * np.log10(np.maximum(p, 1.0) / ref))


def waiting(d, s, prompt):
    panel(d, (40, 224, 1240, 610))
    text(d, 72, 266, prompt, 27, GREEN)
    if s.get('busy'):
        done, total = s.get('progress', (0, 0))
        if total:
            suffix = ' 组' if '评估' in prompt else ''
            message = '正在计算  %d / %d%s' % (done, total, suffix)
        else:
            message = '正在执行板端计算…'
        text(d, 74, 322, message, 22, GOLD)
        if total:
            d.rectangle((74, 372, 1180, 394), fill=EDGE)
            d.rectangle((74, 372, 74 + 1106 * done / max(1, total), 394), fill=GREEN)
    else:
        text(d, 74, 325, '按 PL KEY4 开始，完成后停留在当前页面。', 21, DIM)


def input_page(d, s):
    sample = s.get('sample')
    if sample is None:
        return waiting(d, s, '三通道振动数据输入')
    waves = sample['waveforms']
    limit = max(.02, float(np.max(np.abs(waves))) * 1.08)
    for c in range(3):
        y = 216 + c * 137
        chart(d, (40, y, 934, y + 126), [(waves[c], COLORS[c])],
              CHANNELS[c] + ' · 原始时域波形', 1023 / 12.0, bounds=(-limit, limit))
    metric(d, 958, 216, '输入工况', sample['class_name'], BLUE, 282)
    if s.get('source_mode') == 'gateway':
        frame_value = '%d / %d' % (s.get('gateway_frame_offset', 0) + 1, s['frame_count'])
        frame_label = '缓存帧 / 总缓存'
    else:
        frame_value = '%03d / %d' % (sample['index'] + 1, s['frame_count'])
        frame_label = '帧编号 / 总帧数'
    metric(d, 958, 324, frame_label, frame_value, GREEN, 282)
    metric(d, 958, 432, '每通道采样率 / 帧长', '12 kHz / 1024', GOLD, 282)
    if s.get('source_mode') == 'gateway':
        text(d, 974, 550, 'PC 多站点网关', 15, GREEN)
        text(d, 974, 578, '站点 ' + str(sample.get('site_id', s.get('gateway_site', '?'))), 15, DIM)
    else:
        text(d, 974, 550, 'CWRU 数据 + 合成不平衡', 15, DIM)
        text(d, 974, 578, 'DDR 文件回放 · Q1.15', 15, DIM)
    footer = sample.get('provenance','文件回放')
    if s.get('source_mode') == 'gateway':
        footer += '；KEY3 冻结并查看上一缓存帧；KEY4 恢复最新帧'
    fit(d, 42, 635, footer, 1180, 14, DIM)


def filter_page(d, s):
    r = s.get('filter')
    if not r:
        return waiting(d, s, 'FIR 定点滤波与频谱抑制验证')
    source = r['source'] + (' · 连续验证第 %d 帧' % r['sweep_index'] if 'sweep_index' in r else '')
    fit(d, 42, 198, source, 1180, 15, DIM)
    chart(d, (40, 236, 626, 450), [(r['raw'][:180], BLUE), (r['filtered_ref'][:180], GREEN)],
          '输入 / 滤波波形（PS 定点参考）', 179 / 12.0)
    reference = max(float(np.max(r['before']['spectrum'])), 1.0)
    chart(d, (646, 236, 1240, 450),
          [(relative_db(r['before']['spectrum'], reference), BLUE),
           (relative_db(r['after']['spectrum'], reference), GREEN)],
          'PL 实测：旁路 / FIR 后功率谱', 511 * 12000 / 1024, 'Hz', (-80, 3))
    text(d, 58, 462, '蓝：输入 / 旁路     绿：FIR 输出', 15, DIM)
    metric(d, 40, 504, '定点结构', '64 taps · Q1.15', BLUE, 382, '转置 FIR · 零状态逐帧处理')
    metric(d, 448, 504, '5.27–5.85 kHz 能量抑制', '%.1f dB' % r['attenuation_db'], GREEN, 382, '同一输入 · 同一 FFT 缩放 · 两次 PL 实测')
    metric(d, 856, 504, '寄存器配置', '32 次系数写入', GOLD, 384, 'AXI-Lite · 验证后恢复诊断系数')
    text(d, 42, 635, '波形使用 PS 定点参考；两条频谱均来自真实 PL + DMA 回传，采用同一纵轴基准。', 14, DIM)


def fft_page(d, s):
    sample = s.get('sample')
    if not sample or 'features' not in sample:
        return waiting(d, s, '三通道硬件 FFT 频谱分析')
    for c, feat in enumerate(sample['features']):
        y = 216 + c * 137
        chart(d, (40, y, 936, y + 126), [(relative_db(feat['spectrum']), COLORS[c])],
              CHANNELS[c] + ' · FIR 后 PL 功率谱 (dB, 本通道峰值=0)',
              511 * 12000 / 1024, 'Hz', (-70, 3), [(feat['peak_hz'], '主峰')])
        metric(d, 958, y, CHANNELS[c] + ' · 主峰', '%.1f Hz' % feat['peak_hz'], COLORS[c], 282,
               ('近峰多解；质心 ' if feat.get('peak_ambiguous') else '谱质心 ') + '%.1f Hz' % feat['centroid_hz'])
    text(d, 42, 635, '1024 点定点 FFT · Δf = 11.71875 Hz · DMA 回传 512 bin · 当前工况：' + sample['class_name'], 14, DIM)


def feature_page(d, s):
    sample = s.get('sample')
    if not sample or 'features' not in sample:
        return waiting(d, s, '时域统计与频域特征提取')
    panel(d, (40, 216, 1240, 421))
    headers = [(58, '通道'), (286, 'RMS'), (484, '峭度'), (668, '峰值幅值'), (866, '主峰 / Hz'), (1064, '谱质心 / Hz')]
    for x, name in headers:
        text(d, x, 232, name, 16, DIM)
    for c, f in enumerate(sample['features']):
        y = 278 + c * 47
        text(d, 58, y, CHANNELS[c], 18, COLORS[c])
        for x, value in [(286, '%.5f' % np.sqrt(f['rms2'])), (484, '%.2f' % f['kurtosis']),
                         (668, '%.5f' % f['peak_abs']), (866, '%.1f' % f['peak_hz']),
                         (1064, '%.1f' % f['centroid_hz'])]:
            text(d, x, y, value, 21)
    for c, f in enumerate(sample['features']):
        x = 40 + c * 408
        panel(d, (x, 441, x + 384, 610))
        text(d, x + 14, 450, CHANNELS[c] + ' · 频带能量占比', 16, COLORS[c])
        for b, ratio in enumerate(f['band_ratio']):
            y = 481 + b * 19
            text(d, x + 12, y, BANDS[b], 12, DIM)
            d.rectangle((x + 142, y + 3, x + 300, y + 13), fill=EDGE)
            if ratio > 0:
                d.rectangle((x + 142, y + 3, x + 142 + 158 * min(1, ratio), y + 13), fill=COLORS[c])
            text(d, x + 310, y - 1, '%.1f%%' % (100 * ratio), 12)
    text(d, 42, 635, 'PL 归约并写入 AXI-Lite 特征寄存器；PS 完成归一化 / 开方。振幅单位为 Q1.15 归一化值。', 14, DIM)


def diagnosis_page(d, s):
    r = s.get('diagnosis')
    if not r:
        return waiting(d, s, '三通道融合故障识别')
    accepted = r.get('accepted', True)
    result_color = GREEN if accepted else GOLD
    panel(d, (40, 216, 456, 610), result_color)
    text(d, 64, 241, '诊断结果', 19, DIM)
    fit(d, 64, 285, r['prediction_name'], 366, 40, result_color)
    text(d, 64, 359, '投票证据（OVO）', 18, DIM)
    text(d, 64, 392, ('%d / %d' % (r['votes'], r['votes_possible'])) if accepted else '拒绝判类', 38, GOLD)
    text(d, 64, 457, '一对一分类胜出场数；不是概率', 16, DIM)
    fit(d, 64, 511, '输入标签：' + r['class_name'], 365, 19)
    text(d, 64, 551, '端到端 %.1f ms / 组' % r['elapsed_ms'], 21, BLUE)
    panel(d, (480, 216, 1240, 486))
    text(d, 502, 236, '三通道各 8 帧 → 39 维特征 → RBF SVM', 21)
    for x, v in [(502, '通道'), (718, 'RMS'), (876, '峭度'), (1010, '主峰 / Hz')]:
        text(d, x, 285, v, 16, DIM)
    for c, row in enumerate(r['channels']):
        y = 329 + c * 48
        text(d, 502, y, CHANNELS[c], 19, COLORS[c])
        text(d, 718, y, '%.5f' % np.sqrt(row['rms2']), 20)
        text(d, 876, y, '%.2f' % row['kurtosis'], 20)
        text(d, 1010, y, '%.1f' % row['peak_hz'], 20)
    if not accepted:
        fit(d, 502, 338, r.get('rejection_reason', '请检查输入信号'), 700, 19, GOLD)
    panel(d, (480, 506, 1240, 610))
    text(d, 502, 521, '支持工况', 16, DIM)
    text(d, 502, 554, '正常 / 转子不平衡 / 内圈 / 滚动体 / 外圈故障', 20)
    extra = ('；KEY3 冻结并查看上一缓存组；KEY4 恢复最新帧'
             if s.get('source_mode') == 'gateway' else '')
    fit(d, 42, 635, r.get('provenance','文件回放') + extra + '；坏输入检查不是未知故障检测器。', 1180, 14, DIM)


def accuracy_page(d, s):
    r = s.get('accuracy')
    if not r:
        source = os.path.basename(str(s.get('source', 'demo_set.npz')))
        return waiting(d, s, '带标签评估：%s · PL / PS 识别' % source)
    cm = r['confusion']
    short = ['正常', '内圈', '滚动体', '外圈', '不平衡']
    start_x, start_y, step = 206, 256, 66
    source = os.path.basename(str(s.get('source', 'dataset')))
    fit(d, 42, 198, '评估数据：' + source + ' · 每组 8 帧；下面的误分数是最终结果，不是进度', 1180, 15, DIM)
    text(d, 42, 218, '真实类别 ↓    预测类别 →', 16, DIM)
    for j, name in enumerate(short):
        text(d, start_x + j * step + 5, start_y - 28, name, 14, DIM)
    for i, name in enumerate(short):
        text(d, 58, start_y + i * step + 18, name, 19)
        for j in range(5):
            x, y = start_x + j * step, start_y + i * step
            panel(d, (x, y, x + 62, y + 62), fill=('#23654f' if i == j and cm[i, j] else PANEL))
            text(d, x + 21, y + 16, int(cm[i, j]), 24)
    metric(d, 608, 221, 'PL 组级结果', '%d 组误分 / %d 组' % (r['groups'] - r['correct'], r['groups']), GREEN, 632,
           '封闭数据集统计，不代表现场泛化')
    metric(d, 608, 330, '同一批数据 · PS 软件基线', '%d 组误分 / %d 组' %
           (r['groups'] - int(round(r['software_accuracy'] * r['groups'])), r['groups']), BLUE, 632)
    metric(d, 608, 439, 'PL / PS 预测一致性', '%d / %d 组' % (r['pl_ps_match'], r['groups']), GOLD, 632)
    text(d, 626, 564, '耗时 %.1f s · %d 组 × 3 通道 × 8 帧' % (r['elapsed_s'], r['groups']), 19, DIM)
    fit(d, 42, 635, '源文件内分组；尾帧丢弃 %d；质量拒判 %d；拒判计为未正确识别；不代表独立泛化精度。' %
        (r['dropped_frames'], r.get('rejected',0)),1180,14,DIM)


def performance_page(d, s):
    r = s.get('performance')
    if not r:
        return waiting(d, s, '软硬件端到端性能实测')
    for i, (label, pl, ps, unit) in enumerate([
            ('单通道单帧特征', r['frame_pl_us'], r['frame_ps_us'], 'μs / 帧'),
            ('三通道 8 帧组诊断', r['group_pl_ms'], r['group_ps_ms'], 'ms / 组')]):
        y = 218 + i * 166
        panel(d, (40, y, 1240, y + 148))
        text(d, 58, y + 14, label, 21)
        maximum = max(pl, ps, 1e-9)
        for j, (name, value, color) in enumerate([('PL + DMA', pl, GREEN), ('PS 软件', ps, GOLD)]):
            yy = y + 61 + 42 * j
            text(d, 58, yy - 2, name, 17, color)
            d.rectangle((250, yy, 250 + 570 * value / maximum, yy + 22), fill=color)
            text(d, 840, yy - 2, '%.1f %s' % (value, unit), 17)
        text(d, 1090, y + 52, '加速比', 15, DIM)
        text(d, 1082, y + 85, '%.2f×' % (ps / pl), 29, GREEN)
    text(d, 58, 565, '单帧吞吐：PL %.0f kSample/s     PS %.0f kSample/s' %
         (r['pl_samples_s'] / 1000, r['ps_samples_s'] / 1000), 21, BLUE)
    text(d, 58, 601, '口径：预热后 %d 轮中位数；%d 单帧 + %d 组。加速比 = PS 耗时 / PL 耗时。' %
         (r['rounds'], r['frame_samples'], r['group_samples']), 17, DIM)
    text(d, 42, 635, 'PL 时间含 DMA / 驱动；组诊断含 PS 包络特征与 SVM。绘制 HDMI 页面不计入计算耗时。', 14, DIM)


def system_page(d, s):
    r = s.get('resources', {})
    panel(d, (40, 216, 654, 610))
    text(d, 60, 231, 'FPGA 资源 · 含 HDMI 视频链路', 21)
    for i, (name, used, total, color) in enumerate([
            ('LUT', r.get('lut_used', 0), 53200, GREEN), ('FF', r.get('ff_used', 0), 106400, BLUE),
            ('DSP', r.get('dsp_used', 0), 220, GOLD), ('BRAM', r.get('bram_used', 0), 140, RED)]):
        y = 283 + i * 57
        text(d, 60, y, name, 18, color)
        d.rectangle((148, y + 4, 382, y + 24), fill=EDGE)
        d.rectangle((148, y + 4, 148 + 234 * used / total, y + 24), fill=color)
        text(d, 398, y, '%s / %d  %.1f%%' % (used, total, used / total * 100), 16)
    text(d, 60, 527, '100 MHz · WNS %+.3f ns' % r.get('wns_ns', 0), 22, GREEN)
    text(d, 60, 568, '资源 / 时序来自该 Overlay 的 Vivado 报告', 16, DIM)
    panel(d, (678, 216, 1240, 610))
    text(d, 698, 231, '任务配置与数据管理', 21)
    source_label = ('PC 网关 · ' + str(s.get('gateway_site') or '等待站点')) if s.get('source_mode') == 'gateway' else os.path.basename(s.get('source', 'demo_set.npz'))
    lines = [('数据源', source_label),
             ('输入配置', '%d 帧 · 3 通道 · int16 / Q1.15' % s['frame_count']),
             ('处理参数', '12 kHz · 1024 点 · 64 taps'),
             ('故障识别', '39 维 RBF SVM · 8 帧聚合'),
             ('显示配置', '板卡 HDMI · 1280 × 720 · 手动翻页'),
             ('通道调度', 'DE / FE / BA 时分复用同一 PL 链路'),
             ('版本标识', 'bit %s / BGRX32' % s.get('bitstream_short','unknown'))]
    for i, (label, value) in enumerate(lines):
        y = 281 + 43 * i
        text(d, 698, y, label, 14, DIM)
        fit(d, 804, y - 1, value, 416, 17)
    if s.get('source_mode') == 'gateway':
        footer = 'KEY3 选择站点；KEY4 运行网关数据；网口 %s；单击右下导出结果。%s' % (s.get('gateway_port'), s.get('notice',''))
    else:
        footer = 'KEY3 选择数据：%s；KEY4 应用；单击右侧下方保存运行结果。%s' % (s.get('selected_dataset','demo_set.npz'),s.get('notice',''))
    fit(d, 42, 635, footer,1180,14,DIM)
    panel(d,(698,565,1218,604),edge=GREEN)
    text(d,718,573,'点击导出当前诊断和测试结果',17,GREEN)


SUBTITLES = [
    '三路原始波形 · 数据文件回放 · 单独展示输入功能',
    'AXI-Lite 重载系数 · 64 taps 定点 FIR · 独立验证滤波效果',
    'FIR 后信号经 PL FFT 分析 · 512 bin 功率谱由 DMA 回传',
    '原始时域统计 + FIR 后频域特征 · 三通道分别显示',
    'PL 提取特征 · PS 执行包络谱与 SVM · 输出类别和置信指标',
    '同一批带标签数据分别执行 PL 与 PS 路径 · 统计混淆矩阵',
    '相同输入与特征口径 · 板端实际测量延迟和吞吐率',
    '任务参数 · 板载数据 · Overlay 资源与时序',
]
RENDERERS = [input_page, filter_page, fft_page, feature_page, diagnosis_page, accuracy_page, performance_page, system_page]


def render(page, state, margin=0):
    page %= len(PAGES)
    img = Image.new('RGB', (WIDTH, HEIGHT), '#0a141c')
    d = Canvas(img, margin)
    d.rectangle((0, 0, WIDTH, HEIGHT), fill=BG)
    d.rectangle((0, 0, WIDTH, 68), fill='#0a141c')
    d.rectangle((0, 0, 7, 68), fill=GREEN)
    text(d, 30, 9, '旋转设备多通道振动分析系统', 26)
    text(d, 32, 43, 'AMD Zynq-7020  /  PS Python + PL Accelerator  /  PYNQ', 13, DIM)
    text(d, 1040, 14, 'HDMI 本地演示', 19, GREEN)
    text(d, 1040, 43, 'USB 鼠标已连接' if state.get('mouse_names') else '板载按键操作', 13, DIM)
    presentation = bool(state.get('presentation_mode'))
    d.rectangle((0, 69, WIDTH, 113), fill=('#0a141c' if presentation else PANEL))
    if presentation:
        text(d, 40, 83, '功能点展示  ·  ' + PAGES[page], 19, GREEN)
    else:
        for i, name in enumerate(PAGES):
            x = i * 160
            text(d, x + 12, 82, '%02d %s' % (i + 1, name), 17, FG if i == page else DIM)
            if i == page:
                d.rectangle((x + 8, 109, x + 151, 113), fill=GREEN)
    text(d, 40, 132, '%02d   %s' % (page + 1, PAGES[page]), 30)
    if state.get('gateway_hold'):
        activity = '已冻结查看'
    elif page == 4 and state.get('source_mode') == 'gateway' and state.get('playing'):
        activity = '实时诊断中'
    else:
        activity = ('计算中…' if state.get('busy') else ('回放中' if state.get('playing') else '已暂停'))
    if page in (0, 1, 2, 3, 4):
        text(d, 1090, 140, activity, 20, GREEN if state.get('playing') else GOLD)
    subtitle = SUBTITLES[page]
    if page == 0 and state.get('source_mode') == 'gateway':
        subtitle = 'PC 多站点振动网关 · Ethernet 输入 · FPGA HDMI 本地显示'
    fit(d, 42, 177, subtitle, 1180, 17, DIM)
    if state.get('calibrate'):
        d.rectangle((32, 128, 1248, 650), fill=BG)
        text(d, 44, 142, 'HDMI 颜色校准 · 观察下方三块实际颜色', 28)
        for x, color, name in [(60, '#ff0000', '1 红 R'), (470, '#00ff00', '2 绿 G'), (880, '#0000ff', '3 蓝 B')]:
            d.rectangle((x, 230, x + 340, 493), fill=color)
            text(d, x + 85, 520, name, 32)
        text(d, 44, 595, '正确顺序：红 / 绿 / 蓝；外框与四角应完整可见。', 22, FG)
        d.rectangle((2, 2, WIDTH - 3, HEIGHT - 3), outline='white')
    else:
        RENDERERS[page](d, state)
    if state.get('error'):
        d.rectangle((32, 620, 1246, 660), fill='#693f3c')
        fit(d, 42, 629, state['error'], 1180, 17)
    if presentation:
        return img
    d.rectangle((0, 666, WIDTH, HEIGHT), fill='#0a141c')
    d.line((0, 666, WIDTH, 666), fill=EDGE)
    if state.get('source_mode') == 'gateway':
        third = '选择网关站点' if page == 7 else ('清除结果' if page in (5, 6) else '切换工况')
        fourth = ('暂停 / 继续' if page in (0, 1, 2, 3, 4) else ('重新运行' if page in (5, 6) else '刷新网关'))
    else:
        third = '切换验证信号' if page == 1 else ('选择数据集' if page == 7 else ('清除结果' if page in (5, 6) else '切换工况'))
        fourth = ('暂停 / 继续' if page in (0, 1, 2, 3, 4) else ('重新运行' if page in (5, 6) else '应用数据集'))
    for x, value in [(24, 'PL KEY1  上一页'), (238, 'PL KEY2  下一页'),
                     (454, 'PL KEY3  ' + third), (752, 'PL KEY4  ' + fourth)]:
        right = {24:226, 238:440, 454:736, 752:1108}[x]
        panel(d, (x - 8, 676, right, 714), fill=PANEL)
        text(d, x, 684, value, 15)
    text(d, 1142, 684, '%d / 8' % (page + 1), 15, DIM)
    return img
