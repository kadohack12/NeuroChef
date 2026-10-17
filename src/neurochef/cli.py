"""Command-line interface for NeuroChef."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurochef.core import SearchEngine
from neurochef.detectors import detect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroChef — automatic byte-stream decoder")
    parser.add_argument("input", nargs="?", help="Input text; omit to read raw bytes from stdin")
    parser.add_argument("--file", help="Read raw bytes from a file")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.file:
        data = Path(args.file).read_bytes()
    elif args.input is not None:
        data = args.input.encode()
    else:
        data = sys.stdin.buffer.read()
    result = SearchEngine(max_depth=args.max_depth, max_nodes=args.max_nodes).solve(data)
    print(result.data.decode("utf-8", errors="replace"))
    print(f"\nsteps: {' -> '.join(step.transform for step in result.steps) or 'none'}", file=sys.stderr)
    print(f"detection: {detect(result.data)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
