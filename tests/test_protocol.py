import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from qmint.protocol import (
    job_id,
    recv_authenticated,
    recv_pickle,
    remove_port,
    send_authenticated,
    send_pickle,
    write_port,
)


class MemorySocket:
    def __init__(self):
        self.buffer = bytearray()

    def sendall(self, data):
        self.buffer.extend(data)

    def recv(self, size):
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data


class ProtocolTests(unittest.TestCase):
    def test_framed_pickle_round_trip(self):
        stream = MemorySocket()
        send_pickle(stream, {"value": [1, 2, 3]})
        self.assertEqual(recv_pickle(stream), {"value": [1, 2, 3]})

    def test_authentication_happens_before_payload_decode(self):
        token = "a" * 64
        stream = MemorySocket()
        send_authenticated(stream, token, {"task": "energy"})
        self.assertEqual(recv_authenticated(stream, token), {"task": "energy"})

    def test_wrong_token_is_rejected(self):
        stream = MemorySocket()
        stream.sendall(b"x" * 64)
        with self.assertRaises(PermissionError):
            recv_authenticated(stream, "y" * 64)

    def test_server_file_is_private(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"TMPDIR": directory, "MLP_JOBID": "test"}
        ):
            write_port(1234, token="x" * 64)
            path = Path(directory) / "qmint_test.json"
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            remove_port()

    def test_job_id_is_safe_for_a_filename(self):
        with patch.dict("os.environ", {"MLP_JOBID": "queue/user 1"}):
            self.assertEqual(job_id(), "queue_user_1")


if __name__ == "__main__":
    unittest.main()
