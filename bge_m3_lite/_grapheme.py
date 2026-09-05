"""Extended grapheme cluster segmentation (UAX #29), pure Python.

Needed because Hugging Face's ``Precompiled`` normalizer applies the
SentencePiece charsmap per grapheme cluster rather than by longest prefix.
Tables come from ``_grapheme_data`` (generated from the Unicode database).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator

from bge_m3_lite import _grapheme_data as _d

# Grapheme_Cluster_Break values (indices into _d.GCB_NAMES)
_OTHER, _CR, _LF, _CONTROL, _EXTEND, _ZWJ, _RI, _PREPEND, _SPACINGMARK = range(9)
_L, _V, _T, _LV, _LVT = range(9, 14)
# Indic_Conjunct_Break values
_INCB_NONE, _INCB_CONSONANT, _INCB_EXTEND, _INCB_LINKER = range(4)


def _lookup(cp: int, starts, ends, values) -> int:
    i = bisect_right(starts, cp) - 1
    if i >= 0 and cp <= ends[i]:
        return values[i]
    return 0


_gcb_cache: dict[int, int] = {}
_incb_cache: dict[int, int] = {}
_ext_cache: dict[int, bool] = {}


def gcb(cp: int) -> int:
    v = _gcb_cache.get(cp)
    if v is None:
        v = _lookup(cp, _d.GCB_STARTS, _d.GCB_ENDS, _d.GCB_VALUES)
        if len(_gcb_cache) < 65536:
            _gcb_cache[cp] = v
    return v


def incb(cp: int) -> int:
    v = _incb_cache.get(cp)
    if v is None:
        v = _lookup(cp, _d.INCB_STARTS, _d.INCB_ENDS, _d.INCB_VALUES)
        if len(_incb_cache) < 65536:
            _incb_cache[cp] = v
    return v


def is_extended_pictographic(cp: int) -> bool:
    v = _ext_cache.get(cp)
    if v is None:
        v = bool(_lookup(cp, _d.EXTPICT_STARTS, _d.EXTPICT_ENDS, _d.EXTPICT_VALUES))
        if len(_ext_cache) < 65536:
            _ext_cache[cp] = v
    return v


def grapheme_boundaries(text: str) -> Iterator[int]:
    """Yield every index ``i`` (0 < i < len) where a grapheme cluster boundary falls."""
    n = len(text)
    if n < 2:
        return
    cps = [ord(c) for c in text]
    props = [gcb(cp) for cp in cps]
    ri_run = 0  # number of consecutive Regional_Indicator chars ending at i-1
    for i in range(1, n):
        prev, cur = props[i - 1], props[i]
        if prev == _RI:
            ri_run += 1
        else:
            ri_run = 0
        # GB3
        if prev == _CR and cur == _LF:
            continue
        # GB4 / GB5
        if prev in (_CONTROL, _CR, _LF) or cur in (_CONTROL, _CR, _LF):
            yield i
            continue
        # GB6 / GB7 / GB8 (Hangul)
        if prev == _L and cur in (_L, _V, _LV, _LVT):
            continue
        if prev in (_LV, _V) and cur in (_V, _T):
            continue
        if prev in (_LVT, _T) and cur == _T:
            continue
        # GB9 / GB9a / GB9b
        if cur in (_EXTEND, _ZWJ, _SPACINGMARK) or prev == _PREPEND:
            continue
        # GB9c: Consonant [Extend Linker]* Linker [Extend Linker]* x Consonant
        if incb(cps[i]) == _INCB_CONSONANT:
            j = i - 1
            seen_linker = False
            while j >= 0:
                v = incb(cps[j])
                if v == _INCB_LINKER:
                    seen_linker = True
                elif v != _INCB_EXTEND:
                    break
                j -= 1
            if seen_linker and j >= 0 and incb(cps[j]) == _INCB_CONSONANT:
                continue
        # GB11: ExtPict Extend* ZWJ x ExtPict
        if prev == _ZWJ and is_extended_pictographic(cps[i]):
            j = i - 2
            while j >= 0 and props[j] == _EXTEND:
                j -= 1
            if j >= 0 and is_extended_pictographic(cps[j]):
                continue
        # GB12 / GB13: keep RI pairs together
        if prev == _RI and cur == _RI and ri_run % 2 == 1:
            continue
        # GB999
        yield i


def graphemes(text: str) -> list[str]:
    """Split ``text`` into extended grapheme clusters."""
    out: list[str] = []
    start = 0
    for b in grapheme_boundaries(text):
        out.append(text[start:b])
        start = b
    if text:
        out.append(text[start:])
    return out
