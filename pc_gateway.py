"""PC multi-site vibration gateway for the AX7020.

Examples:
  python pc_gateway.py --board 192.168.2.99 --dataset demo_set.npz --sites motor-A,motor-B,motor-C
  python pc_gateway.py --board 192.168.2.99 --dataset dataset_test_3ch.npz --loop
"""
from __future__ import print_function
import argparse
import socket
import time
from pathlib import Path

import numpy as np

from gateway_protocol import DEFAULT_FS_HZ, pack_frame


def load_frames(path):
    archive = np.load(str(path), allow_pickle=False)
    try:
        key = 'X' if 'X' in archive.files else 'X_test_i16'
        data = np.asarray(archive[key])
        if data.ndim != 3 or data.shape[1:] != (3, 1024):
            raise ValueError('dataset must contain (N,3,1024)')
        if data.dtype != np.int16:
            raise ValueError('gateway dataset must be int16 Q1.15')
        return data
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', default='192.168.2.99')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--dataset', default='demo_set.npz')
    parser.add_argument('--sites', default='motor-A,motor-B,motor-C')
    parser.add_argument('--interval', type=float, default=512. / DEFAULT_FS_HZ)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()
    data = load_frames(Path(args.dataset))
    sites = [item.strip() for item in args.sites.split(',') if item.strip()]
    if not sites:
        raise SystemExit('at least one --sites entry is required')
    sock = socket.create_connection((args.board, args.port), timeout=8.)
    sock.settimeout(8.)
    sequence = 0
    index = 0
    sent = 0
    started = time.monotonic()
    try:
        print('CONNECTED %s:%d; %d frames; sites=%s' % (args.board, args.port, len(data), ','.join(sites)), flush=True)
        while True:
            for offset, site in enumerate(sites):
                frame = data[(index + offset * max(1, len(data) // len(sites))) % len(data)]
                sock.sendall(pack_frame(site, sequence, frame))
                sequence += 1
                sent += 1
            index += 1
            if sent % max(1, len(sites) * 25) == 0:
                print('SENT %d packets %.1f s' % (sent, time.monotonic() - started), flush=True)
            if args.max_frames and sent >= args.max_frames:
                break
            if not args.loop and index >= len(data):
                break
            time.sleep(max(0., args.interval))
    except KeyboardInterrupt:
        print('STOPPED by user', flush=True)
    finally:
        sock.close()
    print('DONE packets=%d' % sent, flush=True)


if __name__ == '__main__':
    main()
