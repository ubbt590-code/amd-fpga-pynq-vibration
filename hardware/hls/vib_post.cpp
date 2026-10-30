// vib_post: 频域特征提取加速器 IP
//   AXI-Stream 输入 1024 个 FFT bin (Vivado FFT IP 输出, tdata[15:0]=RE, [31:16]=IM)
//   前半帧(k=0..511)计算功率谱特征, 功率谱经 AXI-Stream 回传 PS
//   后半帧(共轭镜像)丢弃
//   特征经 AXI-Lite 寄存器读出
#include "vib_accel.h"

void vib_post(hls::stream<axis32_t>& in_stream,
              hls::stream<axis32_t>& out_spec,
              ap_uint<64>*  fd_total_pow,
              ap_uint<64>*  fd_band0,
              ap_uint<64>*  fd_band1,
              ap_uint<64>*  fd_band2,
              ap_uint<64>*  fd_band3,
              ap_uint<64>*  fd_band4,
              ap_uint<64>*  fd_band5,
              ap_uint<64>*  fd_centroid_num,
              ap_uint<64>*  fd_peak_pow,
              ap_uint<16>*  fd_peak_bin)
{
#pragma HLS INTERFACE s_axilite port=return          bundle=CTRL
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_spec
#pragma HLS INTERFACE s_axilite port=fd_total_pow    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band0        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band1        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band2        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band3        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band4        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_band5        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_centroid_num bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_peak_pow     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fd_peak_bin     bundle=CTRL

    ap_uint<64> band[NBANDS];
#pragma HLS ARRAY_PARTITION variable=band complete dim=1
    for (int b = 0; b < NBANDS; b++) {
#pragma HLS UNROLL
        band[b] = 0;
    }
    ap_uint<64> total = 0, centroid = 0, pkpow = 0;
    ap_uint<16> pkbin = 0;

FEAT_LOOP:
    for (int k = 0; k < FFT_LEN; k++) {
#pragma HLS PIPELINE II=1
        axis32_t w = in_stream.read();
        if (k < NBINS) {
            ap_int<16> re = (ap_int<16>)w.data.range(15, 0);
            ap_int<16> im = (ap_int<16>)w.data.range(31, 16);
            // int16 的 re^2+im^2 与原 Q1.15 方案位级等价 (raw 同为 Q2.30)
            ap_uint<32> praw = (ap_uint<32>)((ap_int<32>)re * re)
                             + (ap_uint<32>)((ap_int<32>)im * im);

            axis32_t o;
            o.data = praw;
            o.keep = -1; o.strb = -1;
            o.last = (k == NBINS - 1) ? 1 : 0;
            out_spec.write(o);

            if (k > 0) {                            // 剔除直流
                total    += praw;
                centroid += (ap_uint<64>)praw * (ap_uint<64>)k;
            }
            for (int b = 0; b < NBANDS; b++) {
#pragma HLS UNROLL
                if (k >= BAND_LO[b] && k <= BAND_HI[b]) band[b] += praw;
            }
            if (k > 0 && praw > pkpow) {
                pkpow = praw;
                pkbin = (ap_uint<16>)k;
            }
        }
        // k >= 512: 共轭镜像半帧, 直接丢弃
    }

    *fd_total_pow    = total;
    *fd_band0        = band[0];
    *fd_band1        = band[1];
    *fd_band2        = band[2];
    *fd_band3        = band[3];
    *fd_band4        = band[4];
    *fd_band5        = band[5];
    *fd_centroid_num = centroid;
    *fd_peak_pow     = pkpow;
    *fd_peak_bin     = pkbin;
}
