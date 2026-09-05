"""Load tensors from a ``torch.save`` zip archive without importing torch.

The archive layout (``torch>=1.6``) is a zip with ``<name>/data.pkl`` and one
raw little-endian storage per tensor under ``<name>/data/<key>``. The pickle
references storages via ``persistent_load`` and rebuilds tensors through
``torch._utils._rebuild_tensor_v2``; we intercept both and return numpy arrays.
"""

from __future__ import annotations

import collections
import os
import pickle
import zipfile

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Opaque:
    """Stand-in for any torch class we do not need to materialise."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def load_state_dict(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Return ``{tensor_name: ndarray}`` from a ``torch.save``-d dict."""
    with zipfile.ZipFile(path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if len(pkl_names) != 1:
            raise ValueError(f"not a torch archive (no data.pkl): {path}")
        pkl_name = pkl_names[0]
        prefix = pkl_name[: -len("data.pkl")]

        def rebuild_tensor(storage, offset, size, stride, *_rest):
            dtype, key = storage
            raw = np.frombuffer(zf.read(f"{prefix}data/{key}"), dtype=dtype)
            itemsize = raw.itemsize
            view = np.lib.stride_tricks.as_strided(
                raw[offset:],
                shape=tuple(size),
                strides=tuple(s * itemsize for s in stride),
                writeable=False,
            )
            return np.ascontiguousarray(view)

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return rebuild_tensor
                if name in _STORAGE_DTYPES:
                    return name
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                if module == "torch" and name in ("Size", "device"):
                    return _Opaque
                # Never fall back to the stock unpickler: that would let a
                # crafted file execute arbitrary callables.
                raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

            def persistent_load(self, pid):
                # ('storage', <StorageClass name>, key, location, numel)
                _tag, storage_type, key, _location, _numel = pid
                if storage_type not in _STORAGE_DTYPES:
                    raise ValueError(f"unsupported storage type {storage_type}")
                return _STORAGE_DTYPES[storage_type], key

        with zf.open(pkl_name) as fh:
            obj = Unpickler(fh).load()
    if not isinstance(obj, dict):
        raise ValueError("expected a state_dict (dict of tensors)")
    return {str(k): np.asarray(v) for k, v in obj.items()}
