# -*- coding: utf-8 -*-
"""Deterministic HDMI setup and pixel encoding for the AX7020 Overlay."""
import time

import numpy as np


IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]


def encode_rgb(image, order='BGR'):
    """One 32-bit BGRX word per pixel, with an explicit opaque padding byte."""
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('HDMI source must be an RGB image')
    encoded = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    for channel, source in enumerate(order):
        encoded[:, :, channel] = rgb[:, :, 'RGB'.index(source)]
    encoded[:, :, 3] = 255
    return encoded


def read_video_config(display):
    return {
        'bits_per_pixel': display.mode.bits_per_pixel,
        'pixel_unpack_mode': int(display.pixel_unpack.read(0x10)),
        'color_registers': [int(display.color_convert.read(0x10 + 8 * i))
                            for i in range(12)],
    }


def configure_video(display, vdma, width, height):
    # PYNQ's default 32-bit format is RGBA, which also swaps R and B in the
    # color matrix. Supply BGRX + identity explicitly instead of that default.
    from pynq.lib.video import VideoMode, PixelFormat
    display._vdma = vdma
    pixel_format = PixelFormat(32, IDENTITY, IDENTITY)
    display.configure(VideoMode(width, height, 32), pixelformat=pixel_format)
    config = read_video_config(display)
    expected = [int(v * 256) for v in IDENTITY]
    if config['pixel_unpack_mode'] != 1 or config['color_registers'] != expected:
        raise RuntimeError('HDMI pixel/color configuration readback failed: %s' % config)
    display.start()
    # Let the free-running HLS converters finish any invocation that captured
    # reset-time parameters. Only black priming frames are visible at this point.
    for _ in range(3):
        frame = display.newframe()
        frame[:] = 0
        frame[:, :, 3] = 255
        deadline = time.monotonic() + 2.0
        while not (vdma.read(4) & 0x1000):
            if time.monotonic() > deadline:
                frame.freebuffer()
                raise TimeoutError('HDMI VDMA did not complete a priming frame')
            time.sleep(.002)
        display.writeframe(frame)
    return config
