# Hardware source and external build assets

The original vibration HLS sources and testbenches are in hls/:
vib_pre.cpp, vib_post.cpp, fft_cfg_drv.cpp, vib_accel.h and their tests.
The target device is xc7z020clg400-2, with a 10 ns HLS clock target.
The validated build used Vivado and Vitis HLS 2024.2.

The complete competition RAR includes the checked Vivado scripts, board preset,
video adaptation, pre-exported IP, bit/hwh/tcl and exact build reports.
This public tree intentionally excludes third-party/generated IP archives,
board presets and Overlay binaries whose public redistribution is not established.
Obtain compatible vendor assets under their licenses for a new hardware build;
do not assume this source-only tree is a standalone bitstream distribution.

HLS C simulation exercises the custom arithmetic but does not substitute for
AMD FFT RTL simulation or the board-level HDMI/DMA qualification.

The original recording-derived tb_data fixtures are also excluded; the supplied
C++ testbenches need the fixtures from the judging package or independently
regenerated, appropriately licensed fixtures.
