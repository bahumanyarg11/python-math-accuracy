"""Exhaustive input enumeration for binary32 (all finite values, both signs,
as in check_exhaustive.c: 2*0x7f800000 = 4278190080 inputs)."""
import numpy as np
MAXN = 0x7F800000
CH = 1 << 24
NCH = 2 * ((MAXN + CH - 1) // CH)   # 256 chunks


def chunk_inputs(c):
    """chunk c in [0, NCH): positive values for c < NCH/2, negative after"""
    h = NCH // 2
    s, k = (0, c) if c < h else (0x80000000, c - h)
    lo = k * CH
    hi = min(MAXN, lo + CH)
    b = np.arange(lo, hi, dtype=np.uint32) | np.uint32(s)
    return b.view(np.float32)
