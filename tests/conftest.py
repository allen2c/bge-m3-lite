import os
from pathlib import Path

import pytest

from bge_m3_lite import hub

FIXTURES = Path(__file__).parent / "fixtures"


def _try_ensure(files):
    try:
        return hub.ensure_files(files, quiet=True)
    except Exception as exc:  # offline, proxy, ...
        pytest.skip(f"model files unavailable: {exc}")


@pytest.fixture(scope="session")
def tokenizer_path() -> Path:
    return _try_ensure(hub.TOKENIZER_FILES)["sentencepiece.bpe.model"]


@pytest.fixture(scope="session")
def head_paths() -> dict[str, Path]:
    return _try_ensure(hub.HEAD_FILES)


@pytest.fixture(scope="session")
def tokenizer(tokenizer_path):
    from bge_m3_lite.tokenizer import XLMRobertaTokenizer

    return XLMRobertaTokenizer.from_file(str(tokenizer_path))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BGE_M3_LITE_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set BGE_M3_LITE_RUN_SLOW=1 to run full-model tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
