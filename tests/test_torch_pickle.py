import numpy as np

from bge_m3_lite._torch_pickle import load_state_dict


def test_load_heads(head_paths):
    sparse = load_state_dict(str(head_paths["sparse_linear.pt"]))
    colbert = load_state_dict(str(head_paths["colbert_linear.pt"]))
    assert set(sparse) == {"weight", "bias"}
    assert sparse["weight"].shape == (1, 1024) and sparse["weight"].dtype == np.float16
    assert sparse["bias"].shape == (1,)
    assert colbert["weight"].shape == (1024, 1024) and colbert["bias"].shape == (1024,)
    assert np.isfinite(colbert["weight"].astype(np.float32)).all()
    assert abs(float(sparse["bias"][0]) - 0.0452) < 1e-3


def test_refuses_arbitrary_callables(tmp_path):
    import pickle
    import zipfile

    import pytest

    class Evil:
        def __reduce__(self):
            return (print, ("must not run",))

    path = tmp_path / "evil.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("evil/data.pkl", pickle.dumps({"w": Evil()}))
    with pytest.raises(pickle.UnpicklingError):
        load_state_dict(str(path))
    with zipfile.ZipFile(tmp_path / "junk.zip", "w") as zf:
        zf.writestr("hello.txt", "x")
    with pytest.raises(ValueError):
        load_state_dict(str(tmp_path / "junk.zip"))
