# Implementation and validation overview

The AX7020 runs the primary display locally through HDMI. PC software configures
and sends stored three-channel vibration frames over TCP. PS Python manages the
per-site cache, hardware scheduling, envelope features, SVM and eight functional
pages. PL performs a 64-tap Q1.15 FIR, a 1024-point FFT and feature reductions.
The channels time-share one accelerator chain.

The corrected test contains 262 source-bounded groups of eight. One group is
misclassified and all PL/PS predictions agree. The 600-second result is a paced
DDR recording experiment, rather than an ADC-to-screen hard-real-time guarantee.
Timing, resource and per-group numeric evidence are included under evidence/.

The complete judging RAR also contains build scripts, generated IP, the validated
Overlay, recording fixtures, the technical report, optimization tables and
recoverable AI collaboration records. Those additional files are not all part of
this public distribution. The README maps the recommended AMD directories to
both distributions; skill/ states the workflows' use and failure conditions.

There is no validated physical ADC producer. Normal BA is derived, imbalance is
synthetic, and votes are not calibrated probabilities. Cross-board skill reuse
and naturally developing early faults have not been experimentally established.
