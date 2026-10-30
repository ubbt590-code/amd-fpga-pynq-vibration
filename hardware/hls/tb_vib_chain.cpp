// 链路 Testbench: vib_pre -> [FFT IP 仿真模型] -> vib_post
// FFT IP 在 csim 中由 hls::fft_core 位精确 C 模型扮演(综合时由 Vivado xfft IP 替代)
// 与 numpy golden 对拍, 判据同前
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include "vib_accel.h"
#include "hls_fft.h"

void vib_pre(hls::stream<axis16_t>&, hls::stream<axis32_t>&, const coef_t*,
             ap_uint<64>*, ap_uint<64>*, ap_uint<32>*);
void vib_post(hls::stream<axis32_t>&, hls::stream<axis32_t>&,
              ap_uint<64>*, ap_uint<64>*, ap_uint<64>*, ap_uint<64>*,
              ap_uint<64>*, ap_uint<64>*, ap_uint<64>*,
              ap_uint<64>*, ap_uint<64>*, ap_uint<16>*);

// FFT IP 仿真模型参数(与 Vivado 工程中 xfft IP 的配置一致):
// 1024点, pipelined streaming, natural order, scaled, 缩放表 [3,2,2,2,2]=0x2AB
struct FftCfg : hls::ip_fft::params_t {
    static const unsigned max_nfft     = FFT_NLOG2;
    static const unsigned ordering_opt = hls::ip_fft::natural_order;
    static const bool     ovflo        = false;
};

static void fft_ip_model(hls::stream<axis32_t>& in, hls::stream<axis32_t>& out)
{
    cfxp_t xn[FFT_LEN], xk[FFT_LEN];
    for (int n = 0; n < FFT_LEN; n++) {
        axis32_t w = in.read();
        fxp_t re;
        re.range(15, 0) = w.data.range(15, 0);
        xn[n] = cfxp_t(re, fxp_t(0));
    }
    hls::ip_fft::config_t<FftCfg> cfg;
    hls::ip_fft::status_t<FftCfg> st;
    cfg.setDir(true);
    cfg.setSch(0x2AB);
    hls::fft_core<FftCfg, 16, 16, fxp_t, fxp_t, FFT_LEN, 1, hls::ip_fft::fixed_point>(
        xn, xk, &st, &cfg);
    for (int n = 0; n < FFT_LEN; n++) {
        axis32_t w;
        ap_uint<32> d = 0;
        d.range(15, 0)  = (ap_uint<16>)xk[n].real().range(15, 0);
        d.range(31, 16) = (ap_uint<16>)xk[n].imag().range(15, 0);
        w.data = d;
        w.keep = -1; w.strb = -1;
        w.last = (n == FFT_LEN - 1) ? 1 : 0;
        out.write(w);
    }
}

static const char* CLASS_NAMES[5] = {"normal", "inner", "ball", "outer", "unbalance"};

static std::string find_file(const char* rel) {
    const char* prefixes[] = {"", "tb_data/", "../tb_data/", "../../tb_data/",
                              "../../../tb_data/", "../../../../tb_data/"};
    for (int i = 0; i < 6; i++) {
        std::string p = std::string(prefixes[i]) + rel;
        std::ifstream f(p.c_str());
        if (f.good()) return p;
    }
    return std::string(rel);
}

static bool load_samples(const char* name, ap_int<16> buf[FRAME_LEN]) {
    std::ifstream f(find_file(name).c_str());
    if (!f.good()) { printf("ERROR: cannot open %s\n", name); return false; }
    for (int n = 0; n < FRAME_LEN; n++) {
        long v; if (!(f >> v)) return false;
        buf[n] = (ap_int<16>)v;
    }
    return true;
}

static bool load_coefs(const char* name, coef_t c[FIR_TAPS]) {
    std::ifstream f(find_file(name).c_str());
    if (!f.good()) { printf("ERROR: cannot open %s\n", name); return false; }
    for (int i = 0; i < FIR_TAPS; i++) {
        long v; if (!(f >> v)) return false;
        c[i].range(15, 0) = (ap_uint<16>)(ap_int<16>)v;
    }
    return true;
}

static bool load_golden(const char* name, int* g_pkbin, int* g_has_peak,
                        double* g_rms2, double g_ratio[NBANDS]) {
    std::ifstream f(find_file(name).c_str());
    if (!f.good()) { printf("ERROR: cannot open %s\n", name); return false; }
    std::string key;
    f >> key >> *g_pkbin >> key >> *g_has_peak >> key >> *g_rms2 >> key;
    for (int b = 0; b < NBANDS; b++) f >> g_ratio[b];
    return (bool)f;
}

int main() {
    coef_t coefs[FIR_TAPS];
    if (!load_coefs("coefs.txt", coefs)) return 2;

    int fails = 0;
    for (int c = 0; c < 5; c++) {
        char fin[64], fgold[64];
        snprintf(fin,   sizeof(fin),   "in_%d.txt",     c);
        snprintf(fgold, sizeof(fgold), "golden_%d.txt", c);

        ap_int<16> buf[FRAME_LEN];
        if (!load_samples(fin, buf)) return 2;
        int g_pkbin, g_has_peak;
        double g_rms2, g_ratio[NBANDS];
        if (!load_golden(fgold, &g_pkbin, &g_has_peak, &g_rms2, g_ratio)) return 2;

        // ---- 驱动 vib_pre ----
        hls::stream<axis16_t> s_in;
        for (int n = 0; n < FRAME_LEN; n++) {
            axis16_t t;
            t.data = buf[n];
            t.keep = -1; t.strb = -1;
            t.last = (n == FRAME_LEN - 1) ? 1 : 0;
            s_in.write(t);
        }
        hls::stream<axis32_t> s_pre2fft, s_fft2post, s_spec;
        ap_uint<64> sum_sq, sum_4;
        ap_uint<32> pabs;
        vib_pre(s_in, s_pre2fft, coefs, &sum_sq, &sum_4, &pabs);

        // ---- FFT IP 仿真模型 ----
        fft_ip_model(s_pre2fft, s_fft2post);

        // ---- 驱动 vib_post ----
        ap_uint<64> band[NBANDS], total, centroid, pkpow;
        ap_uint<16> pkbin;
        vib_post(s_fft2post, s_spec, &total,
                 &band[0], &band[1], &band[2], &band[3], &band[4], &band[5],
                 &centroid, &pkpow, &pkbin);

        // ---- 频谱流回读 + 结构自检 ----
        ap_uint<64> spec_sum = 0;
        ap_uint<32> spec_pk = 0; int spec_pkbin = 0;
        for (int k = 0; k < NBINS; k++) {
            axis32_t o = s_spec.read();
            if (k == NBINS - 1 && o.last != 1) { printf("  FAIL tlast missing\n"); fails++; }
            if (k > 0) spec_sum += o.data;
            if (k > 0 && o.data > spec_pk) { spec_pk = o.data; spec_pkbin = k; }
        }
        if (spec_sum != total) { printf("  FAIL spec_sum mismatch\n"); fails++; }
        if (spec_pkbin != (int)pkbin) { printf("  FAIL peak reg/stream mismatch\n"); fails++; }

        // ---- 换算与对拍 ----
        double rms2_hw = (double)sum_sq / pow(2.0, 30) / FRAME_LEN;
        double x4_sum  = (double)sum_4  / pow(2.0, 44);
        double kurt_hw = (rms2_hw > 1e-12) ? (x4_sum / FRAME_LEN) / (rms2_hw * rms2_hw) : 0.0;
        double tot     = (double)total > 0 ? (double)total : 1.0;

        printf("=== class %d (%s) ===\n", c, CLASS_NAMES[c]);
        printf("  rms2      hw=%.6f  golden=%.6f\n", rms2_hw, g_rms2);
        printf("  kurtosis  hw=%.3f\n", kurt_hw);
        printf("  peak_bin  hw=%d  golden=%d\n", (int)pkbin, g_pkbin);
        for (int b = 0; b < NBANDS; b++)
            printf("    band%d  %.4f | %.4f\n", b, (double)band[b] / tot, g_ratio[b]);

        double rerr = fabs(rms2_hw - g_rms2) / (g_rms2 + 1e-12);
        if (rerr > 0.05) { printf("  FAIL rms2 relerr=%.3f\n", rerr); fails++; }
        if (g_has_peak && abs((int)pkbin - g_pkbin) > 2) {
            printf("  FAIL peak_bin hw=%d golden=%d\n", (int)pkbin, g_pkbin); fails++;
        }
        for (int b = 0; b < NBANDS; b++) {
            if (g_ratio[b] > 0.05) {
                double e = fabs((double)band[b] / tot - g_ratio[b]) / g_ratio[b];
                if (e > 0.15) { printf("  FAIL band%d relerr=%.3f\n", b, e); fails++; }
            }
        }
        printf("\n");
    }

    if (fails == 0) printf("TEST PASSED (all 5 classes, pre->fft->post chain)\n");
    else            printf("TEST FAILED (%d checks)\n", fails);
    return fails ? 1 : 0;
}
