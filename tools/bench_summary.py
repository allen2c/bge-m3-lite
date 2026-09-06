"""Turn `tools/eval_model.py` output into a Markdown table for the CI job summary.

Usage:  uv run tools/bench_summary.py EVAL.LOG RUNNER "CPU model"
"""

import re
import sys
from pathlib import Path

COLS = (
    "dense min/mean",
    "sparse top-5",
    "colbert p5",
    "held-out dense",
    "held-out top-5",
    "16-tok",
    "128-tok",
    "512-tok",
    "cpu-s/ktok (128)",
    "short query wall/cpu ms",
    "idle ms/s",
    "rss load/peak MiB",
    "start-up s",
)


def parse(log: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    variant = ""
    for line in log.splitlines():
        if m := re.match(r"##### (.*)", line):
            variant = m.group(1).strip()
        elif m := re.match(r"== (\S+) \((\d+) MiB\)(.*)", line):
            name = Path(m.group(1)).name
            if variant:
                name = variant
                variant = ""
            rows.append({"model": f"`{name}` ({m.group(2)} MiB)"})
            if s := re.search(r"session (\S+)s rss (\d+) MiB", m.group(3)):
                rows[-1]["start-up s"] = s.group(1)
                rows[-1]["rss load/peak MiB"] = s.group(2)
        elif not rows:
            continue
        elif m := re.match(r"(?:\[(\S+)\] )?dense cos: min (\S+) mean (\S+)", line):
            col = "held-out dense" if m.group(1) == "held-out" else COLS[0]
            rows[-1][col] = f"{m.group(2)} / {m.group(3)}"
        elif m := re.search(r"(?:\[(\S+)\] )?sparse:.*top-5 identical (\d+/\d+)", line):
            col = "held-out top-5" if m.group(1) == "held-out" else COLS[1]
            rows[-1][col] = m.group(2)
        elif m := re.search(r"colbert token cos: .*p5 (\S+)", line):
            rows[-1].setdefault(COLS[2], m.group(1))
        elif m := re.match(
            r"(\d+)tok x\d+\s+(\d+) tok/s(?:\s+(\S+) cpu-s/ktok)?", line
        ):
            rows[-1][f"{m.group(1)}-tok"] = m.group(2)
            if m.group(1) == "128" and m.group(3):
                rows[-1]["cpu-s/ktok (128)"] = m.group(3)
        elif m := re.match(r"short query\s+(\S+) ms wall\s+(\S+) ms cpu", line):
            rows[-1]["short query wall/cpu ms"] = f"{m.group(1)} / {m.group(2)}"
        elif m := re.match(r"idle cpu (\d+) ms/s .*peak rss (\d+) MiB", line):
            rows[-1]["idle ms/s"] = m.group(1)
            load = rows[-1].get("rss load/peak MiB", "?")
            rows[-1]["rss load/peak MiB"] = f"{load} / {m.group(2)}"
    return rows


def main(argv: list[str]) -> int:
    log_path, runner, cpu = argv[1], argv[2], argv[3] if len(argv) > 3 else ""
    log = Path(log_path).read_text(encoding="utf-8") if Path(log_path).exists() else ""
    print(f"### {runner}: {cpu}\n")
    rows = parse(log)
    if not rows:
        print("no eval output")
        return 0
    print("| model | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for row in rows:
        print(f"| {row['model']} | " + " | ".join(row.get(c, "") for c in COLS) + " |")
    print(
        "\n(tok/s columns: 16 tokens x 32 texts, 128 x 16, 512 x 4; short query = "
        "one 9-token text per call; idle = CPU burnt per second between calls)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
