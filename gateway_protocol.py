"""Small TCP protocol for PC multi-site vibration frames.

The wire format is deliberately independent of NumPy and Python versions:
network byte order header, UTF-8 site id, int16 C-order payload, CRC32.
"""
from __future__ import absolute_import

import collections
import socket
import struct
import threading
import time
import zlib

import numpy as np

MAGIC = b'VIB1'
VERSION = 1
CHANNELS = 3
FRAME_SAMPLES = 1024
DEFAULT_FS_HZ = 12000
HEADER = struct.Struct('!4sBBH I Q B B H I I')
CRC = struct.Struct('!I')
MAX_SITE_BYTES = 63
MAX_PACKET_BYTES = 1024 * 1024


def pack_frame(site_id, sequence, frame, fs_hz=DEFAULT_FS_HZ, timestamp_ns=None):
    site = str(site_id).encode('utf-8')
    if not site or len(site) > MAX_SITE_BYTES:
        raise ValueError('site_id must be 1..63 UTF-8 bytes')
    values = np.asarray(frame)
    if values.shape != (CHANNELS, FRAME_SAMPLES) or values.dtype != np.int16:
        raise ValueError('gateway frame must be (3,1024) int16')
    payload = values.astype('<i2', copy=False).tobytes(order='C')
    stamp = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
    header = HEADER.pack(MAGIC, VERSION, 0, HEADER.size, int(sequence) & 0xffffffff,
                         stamp, len(site), CHANNELS, FRAME_SAMPLES, int(fs_hz), len(payload))
    body = site + payload
    checksum = CRC.pack(zlib.crc32(header + body) & 0xffffffff)
    return header + body + checksum


def pack_site_select(site_id, timestamp_ns=None):
    """Control packet telling the board which logical source HDMI should follow."""
    site = str(site_id).encode('utf-8')
    if not site or len(site) > MAX_SITE_BYTES:
        raise ValueError('site_id must be 1..63 UTF-8 bytes')
    stamp = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
    header = HEADER.pack(MAGIC, VERSION, 1, HEADER.size, 0, stamp,
                         len(site), 0, 0, 0, 0)
    checksum = CRC.pack(zlib.crc32(header + site) & 0xffffffff)
    return header + site + checksum


def _receive_exact(sock, count):
    chunks = []
    remaining = int(count)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError('gateway peer closed connection')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def recv_frame(sock):
    header_bytes = _receive_exact(sock, HEADER.size)
    magic, version, flags, header_len, sequence, timestamp_ns, site_len, channels, samples, fs_hz, payload_len = HEADER.unpack(header_bytes)
    if magic != MAGIC or version != VERSION or header_len != HEADER.size:
        raise ValueError('unsupported gateway header')
    if flags == 1:
        if site_len < 1 or site_len > MAX_SITE_BYTES or channels != 0 or samples != 0 or fs_hz != 0 or payload_len != 0:
            raise ValueError('invalid site selection packet')
        body = _receive_exact(sock, site_len)
        checksum = CRC.unpack(_receive_exact(sock, CRC.size))[0]
        if (zlib.crc32(header_bytes + body) & 0xffffffff) != checksum:
            raise ValueError('gateway CRC mismatch')
        return {'type': 'select_site', 'site_id': body.decode('utf-8')}
    if flags != 0:
        raise ValueError('unsupported gateway flags')
    if site_len < 1 or site_len > MAX_SITE_BYTES or channels != CHANNELS or samples != FRAME_SAMPLES:
        raise ValueError('invalid gateway frame dimensions')
    if payload_len != CHANNELS * FRAME_SAMPLES * 2 or payload_len > MAX_PACKET_BYTES:
        raise ValueError('invalid gateway payload length')
    body = _receive_exact(sock, site_len + payload_len)
    checksum = CRC.unpack(_receive_exact(sock, CRC.size))[0]
    if (zlib.crc32(header_bytes + body) & 0xffffffff) != checksum:
        raise ValueError('gateway CRC mismatch')
    site = body[:site_len].decode('utf-8')
    payload = np.frombuffer(body[site_len:], dtype='<i2').reshape((CHANNELS, FRAME_SAMPLES)).copy()
    return {'site_id': site, 'sequence': int(sequence), 'timestamp_ns': int(timestamp_ns),
            'fs_hz': int(fs_hz), 'frame': payload}


class GatewayState(object):
    """Thread-safe latest-frame and eight-frame history per site."""
    def __init__(self, history_size=16):
        self.lock = threading.Lock()
        self.history_size = int(history_size)
        self.history = {}
        self.latest_by_site = {}
        self.preferred_site = None

    def push(self, packet):
        if packet['fs_hz'] != DEFAULT_FS_HZ:
            raise ValueError('gateway currently accepts only 12000 Hz')
        with self.lock:
            site = packet['site_id']
            self.history.setdefault(site, collections.deque(maxlen=self.history_size)).append(packet)
            self.latest_by_site[site] = packet

    def sites(self):
        with self.lock:
            return sorted(self.latest_by_site.keys())

    def select_site(self, site_id):
        with self.lock:
            self.preferred_site = str(site_id)

    def selected_site(self):
        with self.lock:
            return self.preferred_site

    def snapshot(self, site_id=None, frame_offset=0):
        with self.lock:
            if not self.latest_by_site:
                return None
            if site_id is None:
                site_id = self.preferred_site
            if site_id not in self.latest_by_site:
                site_id = max(self.latest_by_site, key=lambda name: self.latest_by_site[name]['timestamp_ns'])
            packets = list(self.history[site_id])
            selected_index = max(0, len(packets) - 1 - int(frame_offset))
            selected = packets[selected_index]
            visible = packets[:selected_index + 1]
            return {'site_id': site_id, 'latest': selected,
                    'frames': np.stack([item['frame'] for item in visible], axis=0),
                    'sequence': selected['sequence'], 'count': len(packets),
                    'selected_index': selected_index,
                    'frame_offset': len(packets) - 1 - selected_index}


class GatewayReceiver(object):
    """Background TCP listener. It never touches the PYNQ Overlay."""
    def __init__(self, host='0.0.0.0', port=9000):
        self.host, self.port = host, int(port)
        self.state = GatewayState()
        self.stop_event = threading.Event()
        self.server = None
        self.thread = None
        self.client_threads = []

    def start(self):
        if self.thread is not None:
            return
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(4)
        self.server.settimeout(.5)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._accept_loop, name='vibration-gateway', daemon=True)
        self.thread.start()

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                client, address = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(5.)
            worker = threading.Thread(target=self._client_loop, args=(client, address), daemon=True)
            self.client_threads.append(worker)
            worker.start()

    def _client_loop(self, client, address):
        try:
            while not self.stop_event.is_set():
                packet = recv_frame(client)
                if packet.get('type') == 'select_site':
                    self.state.select_site(packet['site_id'])
                else:
                    self.state.push(packet)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def snapshot(self, site_id=None, frame_offset=0):
        return self.state.snapshot(site_id, frame_offset)

    def sites(self):
        return self.state.sites()

    def selected_site(self):
        return self.state.selected_site()

    def close(self):
        self.stop_event.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.)
