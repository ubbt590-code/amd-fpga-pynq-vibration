# -*- coding: utf-8 -*-
"""
vib_driver.py — PYNQ 端振动诊断加速器驱动 (兼容 PYNQ 2.4 ~ 3.x)

用法:
    from vib_driver import VibAccel
    acc = VibAccel("vib_accel.bit", "fir_coef_q15.npy")
    feat = acc.run_frame(frame_i16)     # frame_i16: (1024,) int16 单通道
    # feat: dict(rms2, kurt, band_ratio[6], peak_hz, centroid_hz, spectrum[512])
"""
import numpy as np
import time
from pynq import Overlay

try:                      # PYNQ >= 2.5
    from pynq import allocate
except ImportError:       # PYNQ 2.4 (ALINX 镜像)
    from pynq import Xlnk
    _xlnk = Xlnk()
    def allocate(n, dtype):
        return _xlnk.cma_array(shape=(n,), dtype=dtype)

FS = 12000.0
FRAME = 1024
NBINS = 512
BAND_LO = [1, 43, 86, 171, 257, 342]
BAND_HI = [42, 85, 170, 256, 341, 511]

# vib_pre 寄存器偏移 (来自 xvib_pre_hw.h)
PRE_AP      = 0x00
PRE_SUM_SQ  = 0x10   # 64bit
PRE_SUM_4TH = 0x28   # 64bit
PRE_PEAK    = 0x40   # 32bit
PRE_COEF    = 0x80   # 64*16bit, 两个系数打包一个32bit字

# vib_post 寄存器偏移 (来自 xvib_post_hw.h)
POST_AP       = 0x00
POST_TOTAL    = 0x10
POST_BAND     = [0x28, 0x40, 0x58, 0x70, 0x88, 0xa0]
POST_CENTROID = 0xb8
POST_PEAKPOW  = 0xd0
POST_PEAKBIN  = 0xe8


def _rd64(ip, off):
    lo = ip.read(off) & 0xFFFFFFFF
    hi = ip.read(off + 4) & 0xFFFFFFFF
    return (hi << 32) | lo


class VibAccel:
    def __init__(self, bitstream="vib_accel.bit", coef_path="fir_coef_q15.npy", timeout=2.0):
        self.ol = Overlay(bitstream)
        self.dma = self.ol.axi_dma_0
        self.pre = self.ol.vib_pre_0
        self.post = self.ol.vib_post_0
        self.timeout = float(timeout)
        self.frames_completed = 0
        self.in_buf = allocate(FRAME, np.int16)
        self.out_buf = allocate(NBINS, np.uint32)
        self.load_coefs(np.load(coef_path))

    def load_coefs(self, coef_q15):
        """64 个 int16 系数, 打包写入 0x80 起始的 32 个寄存器字"""
        coef_q15 = np.asarray(coef_q15)
        if coef_q15.shape != (64,) or coef_q15.dtype != np.int16:
            raise ValueError("FIR coefficients must be a (64,) int16 array")
        for n in range(32):
            lo = int(coef_q15[2 * n]) & 0xFFFF
            hi = int(coef_q15[2 * n + 1]) & 0xFFFF
            self.pre.write(PRE_COEF + 4 * n, lo | (hi << 16))

    def run_frame(self, frame_i16):
        """跑一帧 (1024, int16), 返回特征 dict 和 512bin 功率谱"""
        frame_i16 = np.asarray(frame_i16)
        if frame_i16.shape != (FRAME,) or frame_i16.dtype != np.int16:
            raise ValueError("DMA frame must be a (1024,) int16 array")
        np.copyto(self.in_buf, frame_i16)

        # 启动两个 IP (ap_start)
        self.pre.write(PRE_AP, 1)
        self.post.write(POST_AP, 1)
        # 先挂接收, 再发送, 等两边完成
        self.dma.recvchannel.transfer(self.out_buf)
        self.dma.sendchannel.transfer(self.in_buf)
        deadline = time.monotonic() + self.timeout
        for offset, name in [(0x04, 'MM2S'), (0x34, 'S2MM')]:
            while True:
                status = self.dma.read(offset)
                if status & 0x70:
                    raise RuntimeError("AXI DMA %s error: 0x%08x" % (name, status))
                if status & 2:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("AXI DMA %s timed out: 0x%08x" % (name, status))
                time.sleep(0.00002)
        # Preserve PYNQ's buffer invalidation after bounded completion checks.
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        for ip, name in [(self.pre, 'FIR'), (self.post, 'features')]:
            while not (ip.read(0) & 6):
                if time.monotonic() > deadline:
                    raise TimeoutError("PL %s register results timed out" % name)
        self.frames_completed += 1

        # ---- 读特征寄存器 ----
        sum_sq  = _rd64(self.pre, PRE_SUM_SQ)
        sum_4th = _rd64(self.pre, PRE_SUM_4TH)
        peak    = self.pre.read(PRE_PEAK)
        total   = _rd64(self.post, POST_TOTAL)
        bands   = [_rd64(self.post, o) for o in POST_BAND]
        cent    = _rd64(self.post, POST_CENTROID)
        pkbin   = self.post.read(POST_PEAKBIN)

        # ---- 换算为物理量 (与 golden 单位一致: Q1.15 归一化) ----
        rms2 = sum_sq / (2.0 ** 30) / FRAME
        x4m  = sum_4th / (2.0 ** 44) / FRAME
        kurt = (x4m / (rms2 * rms2)) if rms2 > 1e-12 else 0.0
        tot  = float(total) if total > 0 else 1.0
        return dict(
            rms2=rms2,
            kurtosis=kurt,
            peak_abs=float(peak) / 32768.0,
            band_ratio=[b / tot for b in bands],
            peak_hz=pkbin * FS / FRAME,
            centroid_hz=(cent / tot) * FS / FRAME,
            spectrum=self.out_buf.copy(),       # uint32 功率, Q2.30 原始码
        )

    def close(self):
        for name in ('in_buf', 'out_buf'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.freebuffer()
                setattr(self, name, None)


if __name__ == "__main__":
    # 板上自检: 用数据集中的一帧测试
    d = np.load("dataset.npz", allow_pickle=True)
    acc = VibAccel()
    x = d["X_test_i16"][0, 0]
    f = acc.run_frame(x)
    print("rms2=%.6f kurt=%.2f peak=%.1fHz" % (f["rms2"], f["kurtosis"], f["peak_hz"]))
    print("band_ratio:", np.round(f["band_ratio"], 4))
