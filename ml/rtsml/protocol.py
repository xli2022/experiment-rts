"""The frame protocol on the pipe to a Bun environment process.

Mirrors `tools/ml/protocol.ts`: `u32 length | u8 kind | u32 headerLength |
header JSON | payload`, little-endian, with the header's `arrays` naming the
payload's arrays in order.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, BinaryIO

import numpy as np


class Kind(IntEnum):
    HELLO = 1
    RESET = 2
    STEP = 3
    OBS = 4
    ERROR = 5
    CLOSE = 6


DTYPES = {"f32": np.dtype("<f4"), "u8": np.dtype("u1"), "i32": np.dtype("<i4")}
NAMES = {np.dtype("float32"): "f32", np.dtype("uint8"): "u8", np.dtype("int32"): "i32"}


@dataclass
class Frame:
    kind: Kind
    header: dict[str, Any]
    arrays: list[np.ndarray]


def encode_frame(kind: Kind, header: dict[str, Any], arrays: list[tuple[str, np.ndarray]] | None = None) -> bytes:
    arrays = arrays or []
    descs = []
    payload = bytearray()
    for name, array in arrays:
        array = np.ascontiguousarray(array)
        if array.dtype not in NAMES:
            raise TypeError(f"unsupported dtype {array.dtype} for {name}")
        descs.append({"name": name, "dtype": NAMES[array.dtype], "shape": list(array.shape)})
        payload += array.astype(DTYPES[NAMES[array.dtype]], copy=False).tobytes()
    header_bytes = json.dumps({**header, "arrays": descs}).encode("utf-8")
    length = 1 + 4 + len(header_bytes) + len(payload)
    return struct.pack("<IBI", length, int(kind), len(header_bytes)) + header_bytes + bytes(payload)


def decode_frame(data: bytes) -> Frame:
    length, kind, header_len = struct.unpack_from("<IBI", data, 0)
    if len(data) != 4 + length:
        raise ValueError(f"frame is {len(data)} bytes, header says {4 + length}")
    header = json.loads(data[9 : 9 + header_len].decode("utf-8"))
    arrays = []
    at = 9 + header_len
    for desc in header.get("arrays", []):
        dtype = DTYPES[desc["dtype"]]
        count = int(np.prod(desc["shape"])) if desc["shape"] else 1
        size = count * dtype.itemsize
        arrays.append(np.frombuffer(data[at : at + size], dtype=dtype).reshape(desc["shape"]))
        at += size
    if at != len(data):
        raise ValueError(f"frame has {len(data) - at} trailing bytes")
    return Frame(Kind(kind), header, arrays)


def read_frame(stream: BinaryIO) -> Frame:
    head = _read_exactly(stream, 4)
    (length,) = struct.unpack("<I", head)
    return decode_frame(head + _read_exactly(stream, length))


def write_frame(stream: BinaryIO, frame: bytes) -> None:
    stream.write(frame)
    stream.flush()


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise EOFError("environment process closed the pipe")
        out += chunk
    return bytes(out)
