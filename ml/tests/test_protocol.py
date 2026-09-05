import io

import numpy as np

from rtsml.protocol import Kind, decode_frame, encode_frame, read_frame


def test_round_trip_with_arrays():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([1, 0, 1], dtype=np.uint8)
    c = np.array([[-1, 7]], dtype=np.int32)
    frame = decode_frame(encode_frame(Kind.STEP, {"maxObserved": 2}, [("a", a), ("b", b), ("c", c)]))
    assert frame.kind == Kind.STEP
    assert frame.header["maxObserved"] == 2
    assert [d["name"] for d in frame.header["arrays"]] == ["a", "b", "c"]
    np.testing.assert_array_equal(frame.arrays[0], a)
    np.testing.assert_array_equal(frame.arrays[1], b)
    np.testing.assert_array_equal(frame.arrays[2], c)


def test_header_only_frame():
    frame = decode_frame(encode_frame(Kind.HELLO, {"specVersion": 1}))
    assert frame.kind == Kind.HELLO and frame.arrays == []


class ByteAtATime(io.RawIOBase):
    """A stream that hands back one byte per read, however many were asked for."""

    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def read(self, n: int = -1) -> bytes:
        if self.at >= len(self.data):
            return b""
        chunk = self.data[self.at : self.at + 1]
        self.at += 1
        return chunk


def test_read_frame_survives_fragmented_reads():
    payload = np.random.default_rng(0).random((5, 7), dtype=np.float32)
    data = encode_frame(Kind.OBS, {"slots": []}, [("x", payload)]) + encode_frame(Kind.CLOSE, {})
    stream = ByteAtATime(data)
    first = read_frame(stream)
    second = read_frame(stream)
    assert first.kind == Kind.OBS
    np.testing.assert_array_equal(first.arrays[0], payload)
    assert second.kind == Kind.CLOSE
