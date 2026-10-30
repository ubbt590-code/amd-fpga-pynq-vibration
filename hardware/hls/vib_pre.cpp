// vib_pre: 预处理加速器 IP
//   AXI-Stream 输入 1024×int16 -> 64阶FIR低通 -> AXI-Stream 输出 (FFT IP 格式)
//   同时在原始输入上提取时域特征, 经 AXI-Lite 寄存器读出
#include "vib_accel.h"

// ---------------- 64 阶 FIR 低通滤波器 (转置脉动结构) ----------------
// 每 tap 一个"乘-加-寄存"单元并行工作 (每单元映射一个 DSP48)
// 所有单元读的都是上一拍的寄存器旧值, 无组合级联, II=1 每样本 1 拍
static void fir64(const coef_t c[FIR_TAPS],
                  const fxp_t  xin[FRAME_LEN],
                  fxp_t        yout[FRAME_LEN])
{
    acc_t acc[FIR_TAPS];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

FIR_INIT:
    for (int i = 0; i < FIR_TAPS; i++) {
#pragma HLS UNROLL
        acc[i] = 0;
    }

FIR_FRAME:
    for (int n = 0; n < FRAME_LEN; n++) {
#pragma HLS PIPELINE II=1
        fxp_t x = xin[n];
        // 乘法保持 16x16 (单 DSP), 乘积再扩位进入 40bit 累加链
        for (int t = 0; t < FIR_TAPS - 1; t++) {
#pragma HLS UNROLL
            acc[t] = acc[t + 1] + c[t] * x;
        }
        acc[FIR_TAPS - 1] = c[FIR_TAPS - 1] * x;
        yout[n] = (fxp_t)acc[0];
    }
}

void vib_pre(hls::stream<axis16_t>& in_stream,
             hls::stream<axis32_t>& out_stream,   // 到 FFT IP: tdata[15:0]=RE
             const coef_t  fir_coef[FIR_TAPS],
             ap_uint<64>*  td_sum_sq,
             ap_uint<64>*  td_sum_4th,
             ap_uint<32>*  td_peak_abs)
{
#pragma HLS INTERFACE s_axilite port=return       bundle=CTRL
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream
#pragma HLS INTERFACE s_axilite port=fir_coef     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=td_sum_sq    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=td_sum_4th   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=td_peak_abs  bundle=CTRL

    coef_t c_local[FIR_TAPS];
#pragma HLS ARRAY_PARTITION variable=c_local complete dim=1
COEF_COPY:
    for (int i = 0; i < FIR_TAPS; i++) {
#pragma HLS UNROLL
        c_local[i] = fir_coef[i];
    }

    // ---------- 1. 读入数据流 + 时域特征 ----------
    fxp_t x_raw[FRAME_LEN];
    fxp_t x_filt[FRAME_LEN];
    ap_uint<64> sum_sq = 0;
    ap_uint<64> sum_4  = 0;
    ap_uint<15> peak_abs = 0;
    ap_uint<1> last_seen = 0;

READ_LOOP:
    for (int n = 0; n < FRAME_LEN; n++) {
#pragma HLS PIPELINE II=1
        axis16_t s = in_stream.read();
        fxp_t x;
        x.range(15, 0) = (ap_uint<16>)s.data;
        x_raw[n] = x;
        if (n == FRAME_LEN - 1) last_seen = s.last;

        ap_fixed<32, 2> x2 = x * x;
        ap_fixed<64, 4> x4 = x2 * x2;
        sum_sq += (ap_uint<64>)(ap_uint<32>)x2.range(31, 0);
        sum_4  += (ap_uint<64>)(ap_uint<48>)x4.range(63, 16);
        int v = s.data;
        int a = (v < 0) ? -v : v;
        if (a > 32767) a = 32767;
        if ((ap_uint<15>)a > peak_abs) peak_abs = (ap_uint<15>)a;
    }

    // ---------- 2. FIR 低通滤波 ----------
    fir64(c_local, x_raw, x_filt);

    // ---------- 3. 打包输出给 FFT IP ----------
OUT_LOOP:
    for (int n = 0; n < FRAME_LEN; n++) {
#pragma HLS PIPELINE II=1
        axis32_t o;
        ap_uint<32> w = 0;
        w.range(15, 0) = (ap_uint<16>)x_filt[n].range(15, 0);  // XN_RE
        // [31:16] = XN_IM = 0
        o.data = w;
        o.keep = -1; o.strb = -1;
        o.last = (n == FRAME_LEN - 1) ? last_seen : (ap_uint<1>)0;
        out_stream.write(o);
    }

    *td_sum_sq   = sum_sq;
    *td_sum_4th  = sum_4;
    *td_peak_abs = (ap_uint<32>)peak_abs;
}
