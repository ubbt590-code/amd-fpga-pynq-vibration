# -*- coding: utf-8 -*-
"""Hot-plug USB HID mouse support using Linux evdev, without extra packages."""
import errno
import os
from pathlib import Path
import struct
import time

EVENT = struct.Struct('@llHHi')  # Native Linux input_event: 16 bytes on ARMv7.


class MouseInput:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.x, self.y = width // 2, height // 2
        self.devices = {}
        self.next_scan = 0
        self.names = []

    def scan(self):
        if time.monotonic() < self.next_scan:
            return
        self.next_scan = time.monotonic() + 1
        present = set()
        for device in Path('/sys/class/input').glob('event*'):
            try:
                rel = int((device / 'device/capabilities/rel').read_text().strip().split()[-1], 16)
                if (rel & 3) != 3:
                    continue
                path = '/dev/input/' + device.name
                present.add(path)
                if path not in self.devices:
                    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                    self.devices[path] = {'fd': fd, 'buffer': b'',
                                          'name': (device / 'device/name').read_text().strip()}
            except (OSError, ValueError, IndexError):
                continue
        for path in list(self.devices):
            if path not in present:
                os.close(self.devices.pop(path)['fd'])
        self.names = [d['name'] for d in self.devices.values()]

    def feed(self, event_type, code, value):
        """Translate an evdev event; also usable with recorded input events."""
        if event_type == 2:
            if code == 0:
                self.x = max(0, min(self.width - 1, self.x + value))
            elif code == 1:
                self.y = max(0, min(self.height - 1, self.y + value))
            elif code == 8 and value:
                return ('previous' if value > 0 else 'next', None)
        elif event_type == 1 and value == 1:
            if code == 272:
                return ('click', (self.x, self.y))
            if code == 273:
                return ('key3', None)
            if code == 274:
                return ('key4', None)
        return None

    def poll(self):
        self.scan()
        old_xy = self.x, self.y
        actions = []
        for path, dev in list(self.devices.items()):
            try:
                while True:
                    payload = os.read(dev['fd'], EVENT.size * 64)
                    if not payload:
                        break
                    dev['buffer'] += payload
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    os.close(dev['fd'])
                    self.devices.pop(path, None)
                    continue
            while len(dev['buffer']) >= EVENT.size:
                _, _, event_type, code, value = EVENT.unpack(dev['buffer'][:EVENT.size])
                dev['buffer'] = dev['buffer'][EVENT.size:]
                action = self.feed(event_type, code, value)
                if action:
                    actions.append(action)
        return old_xy != (self.x, self.y), actions

    def close(self):
        for dev in self.devices.values():
            os.close(dev['fd'])
        self.devices.clear()
