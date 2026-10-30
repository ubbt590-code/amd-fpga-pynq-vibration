import socket
import sys
import unittest
import numpy as np

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gateway_protocol import GatewayState, pack_frame, recv_frame


class GatewayTests(unittest.TestCase):
    def test_round_trip_crc_and_shape(self):
        left, right = socket.socketpair()
        try:
            frame = np.arange(3 * 1024, dtype=np.int16).reshape(3, 1024)
            left.sendall(pack_frame('motor-A', 7, frame))
            packet = recv_frame(right)
            self.assertEqual(packet['site_id'], 'motor-A')
            self.assertEqual(packet['sequence'], 7)
            self.assertEqual(packet['fs_hz'], 12000)
            np.testing.assert_array_equal(packet['frame'], frame)
        finally:
            left.close(); right.close()

    def test_site_history(self):
        state = GatewayState(history_size=8)
        frame = np.zeros((3, 1024), dtype=np.int16)
        for sequence in range(10):
            state.push({'site_id':'A', 'sequence':sequence, 'timestamp_ns':sequence,
                        'fs_hz':12000, 'frame':frame + sequence})
        snapshot = state.snapshot('A')
        self.assertEqual(snapshot['frames'].shape, (8, 3, 1024))
        self.assertEqual(snapshot['latest']['sequence'], 9)


if __name__ == '__main__':
    unittest.main()
