#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cassert>
#include <cstdio>
typedef ap_axis<16,0,0,0> axis16_t;
void fft_cfg_drv(hls::stream<axis16_t>&);
int main() {
    hls::stream<axis16_t> out;
    fft_cfg_drv(out);
    assert(!out.empty());
    axis16_t word=out.read();
    assert(word.data==0x557 && word.last==0);
    for(int i=0;i<4;++i) fft_cfg_drv(out);
    assert(out.empty());
    std::puts("PASS: single FFT configuration word 0x557");
}
