"""Command line entry point: ``bge-m3-lite download|encode|info``."""

from __future__ import annotations

import argparse
import json
import sys

from bge_m3_lite import __version__, hub


def _cmd_download(args: argparse.Namespace) -> int:
    files = hub.ensure_files(hub.ALL_FILES, args.cache_dir, quiet=args.quiet)
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    cache = hub.default_cache_dir() if args.cache_dir is None else args.cache_dir
    print(f"bge-m3-lite {__version__}")
    print(f"model: {hub.REPO_ID}@{hub.REVISION[:12]}")
    print(f"cache: {cache}")
    for remote in hub.ALL_FILES:
        path = hub.Path(cache) / remote.name
        state = "ok" if hub.is_complete(path, remote) else "missing"
        print(f"  {remote.name:26s} {remote.size / (1 << 20):9.1f} MiB  {state}")
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    from bge_m3_lite.embedder import BGEM3Embedder

    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    embedder = BGEM3Embedder(args.cache_dir, num_threads=args.threads, quiet=args.quiet)
    out = embedder.encode(
        texts,
        batch_size=args.batch_size,
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
    enc.add_argument("--batch-size", type=int, default=12)
    enc.add_argument("--max-length", type=int, default=8192)
    enc.add_argument("--threads", type=int, default=None)
    enc.set_defaults(func=_cmd_encode)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
