"""Validate the CORE-MATH fast reference against MPFR (direct rounding to the
target format) on random inputs, for every function and format."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, acc
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
rng = np.random.default_rng(12345)
UNI = "exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh cbrt erf erfc lgamma tgamma".split()
BI = "pow atan2 hypot".split()
def rand_inputs(fmt, n):
    # uniform over bit patterns of the format (finite values)
    if fmt == "binary32":
        b = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
        x = b.view(np.float32).astype(np.float64)
    elif fmt == "binary16":
        x = rng.integers(0, 2**16, n).astype(np.uint16).view(np.float16).astype(np.float64)
    else:
        b = (rng.integers(0, 2**16, n).astype(np.uint32) << 16).astype(np.uint32)
        x = b.view(np.float32).astype(np.float64)
    return np.where(np.isfinite(x), x, 1.5)
out = {}
for fmt in ["binary32", "binary16", "bfloat16"]:
    for f in UNI + BI:
        x1 = rand_inputs(fmt, N); x2 = rand_inputs(fmt, N) if f in BI else None
        # mix in "interesting" inputs: small-magnitude values
        x1[: N // 4] = rng.uniform(-4, 4, N // 4).astype(np.float32 if fmt=="binary32" else np.float16).astype(np.float64)
        t = time.time(); zf = acc.cr_array(f, fmt, x1, x2, fast=True); tf = time.time() - t
        t = time.time(); zm = acc.cr_array(f, fmt, x1, x2, fast=False); tm = time.time() - t
        same = (zf == zm) | (np.isnan(zf) & np.isnan(zm))
        nbad = int((~same).sum())
        out[f"{fmt}/{f}"] = dict(n=N, mismatches=nbad, t_fast=tf, t_mpfr=tm)
        print(fmt, f, "mismatch", nbad, f"fast {tf/N*1e9:.0f}ns mpfr {tm/N*1e9:.0f}ns", flush=True)
        if nbad:
            i = np.flatnonzero(~same)[:3]
            print("  e.g.", x1[i], x2[i] if x2 is not None else "", zf[i], zm[i])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "validation", f"cm_vs_mpfr_{os.uname().machine}.json"), "w"), indent=1)
