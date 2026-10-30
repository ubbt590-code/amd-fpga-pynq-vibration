#ifndef VIB_ACCEL_H
#define VIB_ACCEL_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <complex>

// ---------------- 系统参数 ----------------
#define FS_HZ       12000          // 采样率 (CWRU 12kHz 数据集)
#define FRAME_LEN   1024           // 每帧样本数 = FFT 点数
#define FFT_NLOG2   10
#define FFT_LEN     1024
#define FIR_TAPS    64             // FIR 低通滤波器阶数
#define NBINS       (FFT_LEN / 2)  // 单边谱 bin 数 = 512
#define NBANDS      6              // 频带数量

// ---------------- 定点类型 ----------------
typedef ap_axis<16, 0, 0, 0>  axis16_t;   // AXI-Stream 输入样本 (int16)
typedef ap_axis<32, 0, 0, 0>  axis32_t;   // AXI-Stream 输出频谱 (uint32/bin)
typedef ap_fixed<16, 1>       coef_t;     // FIR 系数 Q1.15
typedef ap_fixed<16, 1>       fxp_t;      // 信号 Q1.15
typedef std::complex<fxp_t>   cfxp_t;     // FFT 复数
typedef ap_fixed<40, 8>       acc_t;      // FIR 累加器
typedef ap_fixed<32, 2>       pow_t;      // 单 bin 功率 |X[k]|^2, 值域 [0,2)

// ---------------- 频带划分 ----------------
// bin = f * FFT_LEN / FS_HZ, 频率分辨率 11.71875 Hz
// 带0: bin 1~42  (~500Hz  低频/转频区, 不平衡特征)
// 带1: 43~85     (~1kHz)
// 带2: 86~170    (~2kHz)
// 带3: 171~256   (~3kHz   轴承共振带下缘)
// 带4: 257~341   (~4kHz   轴承共振带)
// 带5: 342~511   (~6kHz   高频)
const int BAND_LO[NBANDS] = {  1,  43,  86, 171, 257, 342};
const int BAND_HI[NBANDS] = { 42,  85, 170, 256, 341, 511};

// ---------------- 加速器顶层 ----------------
// 输入: AXI-Stream, 每帧 FRAME_LEN 个 int16 样本
// 输出: out_spec 流回传 512 个 bin 功率值 (DMA 搬回 PS, 供分类/可视化)
//       标量特征经 AXI-Lite 寄存器读取 (PS 端负责开方/除法等收尾计算)
void vib_accel(hls::stream<axis16_t>& in_stream,
               hls::stream<axis32_t>& out_spec,
               const coef_t  fir_coef[FIR_TAPS],
               ap_uint<64>*  td_sum_sq,        // 时域: sum(x^2), 原始位宽 Q2.30 累加
               ap_uint<64>*  td_sum_4th,       // 时域: sum(x^4), 右移16位后累加
               ap_uint<32>*  td_peak_abs,      // 时域: max|x| (Q1.15 原始码值)
               ap_uint<64>*  fd_total_pow,     // 频域: 总功率 (不含直流 bin0)
               ap_uint<64>*  fd_band0,         // 频域: 6 个频带功率
               ap_uint<64>*  fd_band1,
               ap_uint<64>*  fd_band2,
               ap_uint<64>*  fd_band3,
               ap_uint<64>*  fd_band4,
               ap_uint<64>*  fd_band5,
               ap_uint<64>*  fd_centroid_num,  // 频域: 质心分子 sum(k*P[k])
               ap_uint<64>*  fd_peak_pow,      // 频域: 峰值功率
               ap_uint<16>*  fd_peak_bin);     // 频域: 峰值 bin 索引

#endif
