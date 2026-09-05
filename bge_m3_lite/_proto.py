"""Minimal protobuf wire-format decoder.

Only what is needed to read a SentencePiece ``ModelProto`` file. Fields are
returned as ``(field_number, wire_type, raw_value)`` triples; interpreting the
payload (string, sub-message, float, ...) is left to the caller.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator

_VARINT = 0
_FIXED64 = 1
_LENGTH = 2
_FIXED32 = 5


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("malformed varint")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield ``(field_number, wire_type, value)`` for every field in ``buf``."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == _VARINT:
            value, pos = read_varint(buf, pos)
        elif wire == _LENGTH:
            length, pos = read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == _FIXED32:
            value = buf[pos : pos + 4]
            pos += 4
        elif wire == _FIXED64:
            value = buf[pos : pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def as_float32(value: bytes) -> float:
    return struct.unpack("<f", value)[0]


Message = dict[int, list[int | bytes]]


def as_message(buf: bytes) -> Message:
    """Group the fields of one message by field number (repeated fields keep order)."""
    out: Message = {}
    for field, _wire, value in iter_fields(buf):
        out.setdefault(field, []).append(value)
    return out


def get_bytes(msg: Message, field: int, default: bytes = b"") -> bytes:
    values = msg.get(field)
    if not values:
        return default
    value = values[0]
    if not isinstance(value, bytes):
        raise ValueError(f"field {field}: expected length-delimited, got varint")
    return value


def get_int(msg: Message, field: int, default: int = 0) -> int:
    values = msg.get(field)
    if not values:
        return default
    value = values[0]
    if not isinstance(value, int):
        raise ValueError(f"field {field}: expected varint, got bytes")
    return value
