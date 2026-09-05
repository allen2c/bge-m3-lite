import pytest

from bge_m3_lite.quantize import QuantConfig


def test_default_config_is_shipped_profile():
    cfg = QuantConfig()
    assert cfg.method == "dynamic" and cfg.quantize_embeddings


def test_embedder_rejects_unknown_precision(tokenizer_path, head_paths):
    from bge_m3_lite.embedder import BGEM3Embedder

    with pytest.raises(ValueError):
        BGEM3Embedder(precision="int4")  # type: ignore[arg-type]


def test_cli_quantize_help():
    from bge_m3_lite.cli import build_parser

    args = build_parser().parse_args(["quantize", "--method", "nbits", "--bits", "8"])
    assert args.method == "nbits" and args.bits == 8 and args.accuracy_level == 4
