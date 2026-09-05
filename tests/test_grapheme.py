from pathlib import Path

import pytest

from bge_m3_lite._grapheme import graphemes

FIXTURE = Path(__file__).parent / "fixtures" / "GraphemeBreakTest.txt"


def _cases():
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        parts = body.split()
        text = ""
        expected: list[str] = []
        current = ""
        for tok in parts[1:]:  # skip leading ÷
            if tok == "÷":
                expected.append(current)
                current = ""
            elif tok == "×":
                continue
            else:
                ch = chr(int(tok, 16))
                text += ch
                current += ch
        yield text, expected


@pytest.mark.parametrize("text,expected", list(_cases()))
def test_unicode_grapheme_break_test(text, expected):
    assert graphemes(text) == expected


def test_empty_and_single():
    assert graphemes("") == []
    assert graphemes("a") == ["a"]
    assert graphemes("éx") == ["é", "x"]
    assert graphemes("👨‍👩‍👧‍👦🇹🇼") == ["👨‍👩‍👧‍👦", "🇹🇼"]
