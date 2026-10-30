// fft_cfg_drv: FFT IP 配置字驱动器
// 上电后向 xfft 的 s_axis_config 发送一次配置字 0x557:
//   bit[0]   = 1      正变换 (FWD)
//   bit[10:1]= 0x2AB  逐级缩放表 [3,2,2,2,2]
// 无控制寄存器, 自由运行; 发完即停
#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axis<16, 0, 0, 0> axis16_t;
#define FFT_CFG_WORD 0x557

void fft_cfg_drv(hls::stream<axis16_t>& cfg_out)
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis       port=cfg_out
    static ap_uint<1> sent = 0;
    if (!sent) {
        axis16_t w;
        w.data = FFT_CFG_WORD;
        w.keep = -1; w.strb = -1;
        w.last = 0;
        cfg_out.write(w);
        sent = 1;
    }
}
