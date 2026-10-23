"""Independent check of every reported maximum (same platform):
  * the library is called again at the reported input, and its output must
    equal the stored output bit for bit;
  * the error is recomputed in Python with gmpy2 (MPFR) at 256 bits, rising to
    1024 bits until two precisions agree to 1e-12 relative (e.g. near zeros of
    f), with the ulp definition of Gladman et al. written independently of
    harness/acc.c:  err = |f(x) - y| / 2^max(e - p, emin_sub),  2^(e-1) <= |f(x)| < 2^e,
    y = +-Inf replaced by +-2^emax.
usage: verify.py --tag x86_cpu --device cpu
"""
import sys, os, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import numpy as np
import gmpy2
from gmpy2 import mpfr
import libs

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
R = os.path.join(HERE, "..", "results")
FMTP = {"binary64": (53, -1074, 1024), "binary32": (24, -149, 128), "binary16": (11, -24, 16),
        "bfloat16": (8, -133, 128)}
DT = {"binary64": "float64", "binary32": "float32", "binary16": "binary16", "bfloat16": "bfloat16"}


def G(f, a, b):
    if f == "lgamma":
        return gmpy2.lgamma(a)[0]
    if f == "tgamma":
        return gmpy2.gamma(a)
    if f == "rsqrt":   # IEEE 754 rSqrt: rSqrt(-0) = -Inf
        return mpfr("-inf") if (gmpy2.is_zero(a) and gmpy2.is_signed(a)) else gmpy2.rec_sqrt(a)
    if f == "pow":
        return a ** b
    if f == "atan2":
        return gmpy2.atan2(a, b)
    if f == "hypot":
        return gmpy2.hypot(a, b)
    if f in ("asin", "acos", "atan", "asinh", "acosh", "atanh"):
        return getattr(gmpy2, f)(a)
    return getattr(gmpy2, f)(a)


def value(f, x1, x2, prec):
    ctx = gmpy2.context(precision=prec, emin=-(1 << 30), emax=1 << 30)
    ctx.trap_divzero = False; ctx.trap_invalid = False
    with gmpy2.context(ctx):
        a = mpfr(x1); b = mpfr(x2) if x2 is not None else None
        return G(f, a, b)


def err(f, fmt, x1, x2, y):
    p, emin, emax = FMTP[fmt]
    prev = None
    for prec in (256, 512, 1024):
        ctx = gmpy2.context(precision=prec, emin=-(1 << 30), emax=1 << 30)
        with gmpy2.context(ctx):
            v = value(f, x1, x2, prec)
            if gmpy2.is_nan(v):
                return float("nan"), prec
            Y = mpfr(y) if np.isfinite(y) else (mpfr(2) ** emax) * (1 if y > 0 else -1)
            if gmpy2.is_infinite(v):
                return float("inf"), prec
            if v == 0:
                e = emin + p  # ulp = 2^emin_sub
            else:
                e, m = gmpy2.frexp(v)  # gmpy2 returns (exponent, mantissa); v = m 2^e, 0.5 <= |m| < 1
            ue = max(e - p, emin)
            r = float(abs(v - Y) * mpfr(2) ** (-ue))
        if prev is not None and (r == prev or abs(r - prev) <= 1e-12 * abs(r)):
            return r, prec
        prev = r
    return r, prec


_IMPL = {}
LENGTHS = list(range(1, 18)) + [31, 64, 1000, 4096]


def call_contexts(lib, fmt, f, x1, x2):
    """outputs of the library for this input in several calling contexts:
    array lengths LENGTHS, input at the first and at the last position,
    other elements = 1.5 (a harmless value).  Returns {context: output}"""
    key = (lib, fmt)
    if key not in _IMPL:
        _IMPL[key] = libs.impls(lib, args.device, DT[fmt])
    I = _IMPL[key][f]
    dt = np.float64 if fmt == "binary64" else np.float32
    res = {}
    for n in LENGTHS:
        for pos in sorted({0, n - 1}):
            a = np.full(n, 1.5, dt); a[pos] = x1
            if x2 is not None:
                b = np.full(n, 1.5, dt); b[pos] = x2
                y = I(a, b)
            else:
                y = I(a)
            res[f"n={n},pos={pos}"] = float(y[pos])
    return res


out = []
for kind in ["binary32", "binary16", "bfloat16", "binary16_bivar", "bfloat16_bivar",
             "binary64_search", "binary32_search"]:
    for p in sorted(glob.glob(os.path.join(R, kind, args.tag, "*_*.json"))):
        b = os.path.basename(p)[:-5]
        if b in ("versions", "available"):
            continue
        d = json.load(open(p))
        rec = d.get("final", d) if "final" in d else d
        if not rec or rec.get("x1") is None:
            continue
        fmt = d.get("fmt") or kind.split("_")[0]
        lib, f = d["lib"], d["func"]
        x1 = float.fromhex(rec["x1"])
        x2 = float.fromhex(rec["x2"]) if (f in libs.BI and rec.get("x2")) else None
        ystored = float.fromhex(rec["y"])
        try:
            ctx = call_contexts(lib, fmt, f, x1, x2)
        except Exception as ex:
            ctx = {}
        eq = lambda a, b: a == b or (np.isnan(a) and np.isnan(b))
        repro = [k for k, v in ctx.items() if eq(v, ystored)]
        same = len(repro) > 0
        distinct = sorted({float(v).hex() for v in ctx.values()})
        ynow = ctx.get("n=64,pos=0")
        e, prec = err(f, fmt, x1, x2, ystored)
        if np.isinf(e) and np.isinf(rec["max_err"]):
            rel = 0.0            # both infinite ("Inf" in the tables, as theirs)
        else:
            rel = abs(e - rec["max_err"]) / max(abs(e), 1e-300) if np.isfinite(e) else None
        ok = same and rel is not None and rel < 1e-9
        r = dict(kind=kind, lib=lib, func=f, fmt=fmt, x1=rec["x1"], x2=rec.get("x2") if x2 is not None else None,
                 y_stored=rec["y"], y_recomputed=float(ynow).hex() if ynow is not None else None,
                 output_reproduced=bool(same), contexts_reproducing=len(repro), contexts_tried=len(ctx),
                 context_dependent=len(distinct) > 1, outputs_seen=distinct, err_reported=rec["max_err"], err_independent=e,
                 mpfr_bits=prec, rel_diff=rel, ok=bool(ok))
        out.append(r)
        if not ok:
            print("CHECK FAILED:", r, flush=True)
os.makedirs(os.path.join(R, "verify"), exist_ok=True)
json.dump(out, open(os.path.join(R, "verify", f"{args.tag}.json"), "w"), indent=1)
print(f"verify {args.tag}: {sum(r['ok'] for r in out)}/{len(out)} maxima confirmed")
