# Third-party attribution and license boundaries

The MIT license covers the project-authored application, drivers, protocol,
vibration HLS code and reproduction scripts in this public source tree only.

- AMD/Xilinx: Zynq platform, HLS headers, FFT, AXI DMA/VDMA and tool-generated
  design products are governed by AMD's corresponding software/IP licenses.
  Headers and vendor/generated IP archives are not bundled in this public tree.
- PYNQ: the framework and video APIs are credited to Xilinx/PYNQ under its
  BSD-3-Clause project license. Install the framework from its official project;
  this repository does not include its full source or installed dependencies.
- ALINX: AX7020 board image, board presets, documentation and schematic resources
  belong to the board vendor. The board preset and vendor video resources used in
  the judging build are excluded from this public tree.
- Digilent: applicable video IP retains Digilent's license. No vendor video IP
  archive is included here.
- CWRU Bearing Data Center: recordings remain attributed to the original source:
  https://engineering.case.edu/bearingdatacenter/download-data-file
  Raw and derived recordings are excluded from this repository; the provided
  download/reproduction scripts record original filenames and verification hashes.
- NumPy, SciPy, scikit-learn and Pillow retain their own licenses. Dependencies
  are installed separately, not copied into this repository.

Sampling-reference paper: Smith W.A. and Randall R.B., 2015,
DOI 10.1016/j.ymssp.2015.04.021. The paper itself is not redistributed.
