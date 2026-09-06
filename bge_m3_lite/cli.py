"""Command line entry point: ``bge-m3-lite download|info|encode|quantize|fuse``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bge_m3_lite import __version__, hub
from bge_m3_lite.embedder import BATCH_SIZE, MAX_BATCH_TOKENS, MAX_LENGTH
from bge_m3_lite.quantize import ATTENTION_CHUNK


def _cmd_download(args: argparse.Namespace) -> int:
    files = hub.ensure_files(
        hub.ALL_FILES + hub.FUSED_FILES, args.cache_dir, quiet=args.quiet
    )
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    cache = hub.default_cache_dir() if args.cache_dir is None else args.cache_dir
    print(f"bge-m3-lite {__version__}")
    print(f"model: {hub.REPO_ID}@{hub.REVISION[:12]}")
    print(f"cache: {cache}")
    for remote in (*hub.ALL_FILES, *hub.FUSED_FILES, hub.INT8_FILE):
        path = hub.Path(cache) / remote.name
        state = "ok" if hub.is_complete(path, remote) else "missing"
        print(f"  {remote.name:26s} {remote.size / (1 << 20):9.1f} MiB  {state}")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    from bge_m3_lite.quantize import QuantConfig, load_calibration_texts, quantize

    hub.ensure_files(hub.MODEL_FILES, args.cache_dir, quiet=args.quiet)
    source = hub.FUSED_FILES if not args.raw else hub.MODEL_FILES
    files = hub.ensure_files(source, args.cache_dir, quiet=args.quiet)
    cache = hub.default_cache_dir() if args.cache_dir is None else Path(args.cache_dir)
    out = args.output or cache / hub.INT8_FILE.name
    config = QuantConfig(
        method=args.method,
        bits=args.bits,
        block_size=args.block_size,
        accuracy_level=args.accuracy_level,
        quantize_embeddings=not args.keep_embeddings,
        smooth_alpha=None if args.no_smooth else args.alpha,
        symmetric=args.symmetric,
        attention_chunk=args.attention_chunk,
        keep_fp32=tuple(args.keep_fp32 or ()),
    )
    texts = load_calibration_texts(args.calibration) if args.calibration else None
    tokenizer = hub.ensure_files(hub.TOKENIZER_FILES, args.cache_dir, quiet=True)
    size, digest = quantize(
        files[source[0].name],
        out,
        config,
        tokenizer_path=tokenizer["sentencepiece.bpe.model"],
        calibration_texts=texts,
    )
    print(f"{out}: {size} bytes sha256={digest}")
    return 0


def _cmd_fuse(args: argparse.Namespace) -> int:
    from bge_m3_lite.fuse import fuse

    files = hub.ensure_files(hub.MODEL_FILES, args.cache_dir, quiet=args.quiet)
    result = fuse(
        files["model.onnx"], args.output, attention_chunk=args.attention_chunk
    )
    out = args.output or files["model.onnx"].parent
    for name, size, digest in (
        ("model_fused.onnx", result.graph_size, result.graph_sha256),
        ("model_fused.onnx_data", result.data_size, result.data_sha256),
    ):
        print(f"{Path(out) / name}: {size} bytes sha256={digest}")
    print(f"tensors: {result.shared} shared with model.onnx_data, {result.fused} fused")
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    from bge_m3_lite.embedder import BGEM3Embedder

    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    embedder = BGEM3Embedder(
        args.cache_dir,
        num_threads=args.threads,
        quiet=args.quiet,
        precision="int8" if args.int8 else "fp32",
        fused=not args.raw,
        model_path=args.model,
    )
    out = embedder.encode(
        texts,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=args.sparse,
        return_colbert_vecs=args.colbert,
    )
    for i, text in enumerate(texts):
        record = {"text": text, "dense_vecs": out["dense_vecs"][i].tolist()}
        if args.sparse:
            lw = out["lexical_weights"][i]
            record["lexical_weights"] = (
                embedder.convert_id_to_token(lw) if args.tokens else lw
            )
        if args.colbert:
            record["colbert_vecs"] = out["colbert_vecs"][i].tolist()
        print(json.dumps(record, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bge-m3-lite")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--cache-dir", default=None, help="model cache directory")
    parser.add_argument("--quiet", action="store_true", help="no download progress")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch model files into the cache").set_defaults(
        func=_cmd_download
    )
    sub.add_parser("info", help="show version and cache state").set_defaults(
        func=_cmd_info
    )
    enc = sub.add_parser(
        "encode", help="embed texts (args or stdin lines) as JSON lines"
    )
    enc.add_argument("text", nargs="*")
    enc.add_argument(
        "--sparse", action="store_true", help="also output lexical weights"
    )
    enc.add_argument(
        "--colbert", action="store_true", help="also output colbert vectors"
    )
    enc.add_argument(
        "--tokens", action="store_true", help="lexical keys as tokens, not ids"
    )
    enc.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="texts per batch"
    )
    enc.add_argument(
        "--max-batch-tokens",
        type=int,
        default=MAX_BATCH_TOKENS,
        help="padded tokens per batch (bounds memory on long inputs)",
    )
    enc.add_argument("--max-length", type=int, default=MAX_LENGTH)
    enc.add_argument("--threads", type=int, default=None)
    enc.add_argument("--int8", action="store_true", help="use the int8 backbone")
    enc.add_argument(
        "--raw", action="store_true", help="fp32 without the fused graph (slower)"
    )
    enc.add_argument("--model", default=None, help="path to a custom backbone .onnx")
    enc.set_defaults(func=_cmd_encode)
    q = sub.add_parser(
        "quantize",
        help='build the int8 backbone (needs pip install "bge-m3-lite[quant]")',
    )
    q.add_argument("--output", type=Path, default=None, help="default: cache dir")
    q.add_argument(
        "--method", choices=["rowwise", "dynamic", "nbits"], default="rowwise"
    )
    q.add_argument("--bits", type=int, default=8)
    q.add_argument("--block-size", type=int, default=128, help="nbits only")
    q.add_argument(
        "--accuracy-level", type=int, default=4, help="nbits: 0=fp32, 4=int8"
    )
    q.add_argument(
        "--keep-embeddings", action="store_true", help="leave Gather in fp32"
    )
    q.add_argument(
        "--raw",
        action="store_true",
        help="quantize the raw export, not the fused graph",
    )
    q.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant strength")
    q.add_argument("--no-smooth", action="store_true", help="skip SmoothQuant")
    q.add_argument(
        "--calibration", type=Path, default=None, help="one text per line (utf-8)"
    )
    q.add_argument(
        "--attention-chunk",
        type=int,
        default=ATTENTION_CHUNK,
        metavar="N",
        help="query rows per attention pass (0 = whole sequence)",
    )
    q.add_argument(
        "--symmetric",
        action="store_true",
        help="rowwise only: symmetric per-row activations (faster, less exact)",
    )
    q.add_argument(
        "--keep-fp32",
        action="append",
        default=None,
        metavar="REGEX",
        help="rowwise only: leave MatMul/Attention nodes matching this node-name "
        "regex in fp32 (repeatable)",
    )
    q.set_defaults(func=_cmd_quantize)
    f = sub.add_parser(
        "fuse",
        help='build the fused fp32 graph (needs pip install "bge-m3-lite[quant]")',
    )
    f.add_argument("--output", type=Path, default=None, help="default: cache dir")
    f.add_argument(
        "--attention-chunk",
        type=int,
        default=ATTENTION_CHUNK,
        metavar="N",
        help="query rows per attention pass (0 = whole sequence, ORT Attention op)",
    )
    f.set_defaults(func=_cmd_fuse)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
