import struct

import pytest

from bge_m3_lite import _proto


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def test_iter_fields_roundtrip():
    payload = (
        _varint((1 << 3) | 2)
        + _varint(5)
        + b"hello"
        + _varint((2 << 3) | 5)
        + struct.pack("<f", -1.5)
        + _varint((3 << 3) | 0)
        + _varint(300)
        + _varint((4 << 3) | 1)
        + struct.pack("<d", 2.0)
    )
    fields = list(_proto.iter_fields(payload))
    assert fields[0] == (1, 2, b"hello")
    assert fields[1][:2] == (2, 5) and fields[2] == (3, 0, 300)
    assert fields[3][:2] == (4, 1)
    grouped = _proto.as_message(payload)
    assert _proto.get_bytes(grouped, 1) == b"hello"
    assert _proto.as_float32(_proto.get_bytes(grouped, 2)) == -1.5
    assert _proto.get_int(grouped, 3) == 300
    assert _proto.get_int(grouped, 9, 7) == 7 and _proto.get_bytes(grouped, 9) == b""
    with pytest.raises(ValueError):
        _proto.get_int(grouped, 1)
