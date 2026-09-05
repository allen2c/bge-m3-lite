"""Regenerate tests/fixtures/tokenizer_cases.json with the reference tokenizers.

Run with the ``ref`` dependency group:  uv run --group ref tools/make_tokenizer_fixtures.py
"""

# pyright: reportMissingImports=false

import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

CASES = [
    "",
    " ",
    "   ",
    "Hello world!",
    "  leading and trailing spaces  ",
    "multiple    internal     spaces",
    "tab\tseparated\tand\nnewline\r\nmixed",
    "BGE-M3 是一個多語言 embedding 模型。",
    "這是一段很長的中文測試文字，用來測試效能。" * 30,
    "日本語のテキストをトークン化する。",
    "한국어 문장을 토큰화합니다.",
    "Русский текст для проверки токенизатора.",
    "النص العربي للاختبار",
    "עברית מימין לשמאל",
    "हिन्दी पाठ परीक्षण के लिए",
    "ภาษาไทยไม่มีช่องว่างระหว่างคำ",
    "Tiếng Việt có dấu thanh điệu phức tạp",
    "Ελληνικά κείμενα",
    "Türkçe İstanbul ışık ğüşiöç",
    "Deutsch: Straße, Größe, Übermäßig",
    "Français: garçon, naïve, coeur, cœur",
    "ﬁｒｅ ①  café  ＡＢＣ １２３ ㎞ ㍿",
    "combining é vs precomposed é",
    "emoji 😀🚀👨‍👩‍👧‍👦 🇹🇼 ✨",
    "https://example.com/path?q=1&b=2#frag",
    "user@example.com, +886-2-1234-5678",
    "code: def f(x): return x**2  # comment",
    "MiXeD CaSe AND CAPS lock",
    "numbers 3.14159 1,000,000 1e-5 0x1F",
    "quotes “smart” ‘single’ \"straight\" 'apos'",
    "dashes - – — and ellipsis … and ‥",
    "zero​width‌joiner‍ and nbsp here",
    "control\x01chars\x7f\x1f here",
    "soft­hyphen and ﻿BOM",
    "<s> </s> <pad> <unk> <mask> fake special tokens",
    "▁ literal lower one eighth block ▁▁",
    "̀ starts with combining mark",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝒷𝑜𝓁𝒹",
    "ǅ ǈ ǋ titlecase digraphs",
    "ｶﾀｶﾅ半角 ﾊﾟﾋﾟﾌﾟ",
    "ᄀᄁ jamo 각 precomposed",
    "supplementary 𠀀𠀁 𡈽 CJK ext",
    "a" * 300,
    "中" * 300,
    "x y " * 200,
    "mixed 中文 English 日本語 한국어 in one sentence, 混合語言測試。",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog.",
    "機器學習與深度學習有什麼差別？",
]

# Deterministic random unicode soup to widen coverage.
rng = random.Random(20260905)
ranges = [
    (0x20, 0x7E),
    (0xA0, 0x24F),
    (0x370, 0x3FF),
    (0x400, 0x4FF),
    (0x600, 0x6FF),
    (0x900, 0x97F),
    (0xE00, 0xE7F),
    (0x3040, 0x30FF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7A3),
    (0xFF00, 0xFFEF),
    (0x1F300, 0x1F5FF),
    (0x2000, 0x206F),
]
for _ in range(60):
    n = rng.randint(1, 40)
    s = "".join(chr(rng.randint(*rng.choice(ranges))) for _ in range(n))
    CASES.append(s)

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
assert tok.is_fast
out = []
for text in CASES:
    ids = tok(text, add_special_tokens=True)["input_ids"]
    out.append({"text": text, "ids": ids})
dest = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "tokenizer_cases.json"
)
dest.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
print(f"wrote {len(out)} cases to {dest}", file=sys.stderr)
