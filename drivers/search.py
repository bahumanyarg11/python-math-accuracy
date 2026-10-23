"""ATTRIBUTION: the recursive search below is a port of the algorithm of
check_sample2.c from https://gitlab.inria.fr/zimmerma/math_accuracy (commit
c2b897fb, GNU General Public License), by Paul Zimmermann and co-authors.
This file is distributed under GPL-3.0-or-later; see LICENSE.

Lower bounds of the maximal error (binary64 uni/bivariate, binary32
bivariate), following Gladman et al.'s methodology:
  1. the published worst-case inputs (all libraries, Aug 2026 paper, and
     glibc 2.39 inputs) are evaluated;
  2. uniform random sampling over bit patterns;
  3. a vectorised port of the recursive search of check_sample2.c (mode 0):
     the input range (as ordered integers) is repeatedly split in two, each
     half is sampled with T random points, the LIST_ALLOC=20 halves with the
     largest error are kept, and when an interval has at most T points it
     is searched exhaustively.
Errors: distance() port (binary64) / ulp_error_double port (binary32).
Inputs where the library returns NaN and the reference is a number (or the
converse) are counted separately and excluded from the maxima (their
distance() returns 2^1024 for them).
Resumable: one JSON per (lib, func) under results/<fmt>_search/<tag>/.

usage: search.py --fmt binary64 --device cpu --tag x86_cpu [--libs] [--funcs]
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import numpy as np
import acc, libs

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="binary64")       # binary64 | binary32
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", required=True)
ap.add_argument("--libs", default="numpy,scipy,jax,torch")
ap.add_argument("--funcs", default=None)
ap.add_argument("--T", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--nrandom", type=int, default=1 << 20)
args = ap.parse_args()
B64 = args.fmt == "binary64"
DT = "float64" if B64 else "float32"
OUT = os.path.join(HERE, "..", "results", args.fmt + "_search", args.tag)
os.makedirs(OUT, exist_ok=True)
json.dump(libs.versions(), open(os.path.join(OUT, "versions.json"), "w"), indent=1)
if args.funcs:
    FUNCS = args.funcs.split(",")
else:
    FUNCS = libs.ALL if B64 else libs.BI
SEEDS = json.load(open(os.path.join(HERE, "..", "data", "seeds_b64.json" if B64 else "seeds_b32_bivar.json")))

# ordered-integer encoding
if B64:
    UT, IT, FT, MAXB, SIGN = np.uint64, np.int64, np.float64, 0x7FF0000000000000, np.uint64(1 << 63)
else:
    UT, IT, FT, MAXB, SIGN = np.uint32, np.int32, np.float32, 0x7F800000, np.uint32(1 << 31)


def o2f(o):
    o = np.asarray(o, np.int64)
    neg = o < 0
    b = np.where(neg, (-o).astype(np.uint64), o.astype(np.uint64)).astype(UT)
    b = np.where(neg, b | SIGN, b)
    return b.view(FT)


def errors(f, impl, x1, x2, mode=None):
    """per-point errors; NaN mismatches (library NaN xor reference NaN) are
    returned as nan in e and flagged in nanmis"""
    y = impl(x1, x2) if x2 is not None else impl(x1)
    if B64:
        e = acc.err64(f, x1, y, x2)       # 2^1024 (= inf) for NaN mismatch
        inf = np.flatnonzero(np.isinf(e))
        nanmis = np.zeros(len(e), bool)
        if len(inf):
            # is the exact result NaN?  (NaN-ness does not depend on precision)
            z = acc.cr_array(f, "binary32", x1[inf], x2[inf] if x2 is not None else None, fast=False)
            nanmis[inf] = np.isnan(y[inf]) != np.isnan(z)
    else:
        e = acc.err_mt(f, "binary32", x1, y, x2)   # inf only for NaN mismatch
        nanmis = np.isinf(e)
    # flush-to-zero cases (same rule as the exhaustive runs): zero returned
    # where the exact result is a nonzero number below the smallest normal,
    # or a subnormal input giving the library's own result for a zero input.
    minnorm = 2.0 ** -1022 if B64 else 2.0 ** -126
    sub1 = (x1 != 0) & (np.abs(x1) < minnorm)
    sub2 = ((x2 != 0) & (np.abs(x2) < minnorm)) if x2 is not None else np.zeros(len(x1), bool)
    flush = np.zeros(len(e), bool)
    ulp_min = 2.0 ** -1074 if B64 else 2.0 ** -149
    z0 = (y == 0) & np.isfinite(e) & (e > 0.5)
    # for y == 0 the error is |f(x)| / ulp_min: subnormal f(x) <=> e < 2^(p-1)
    mode = mode or {}
    if mode.get("ftz"):
        flush |= z0 & (e < (2.0 ** 52 if B64 else 2.0 ** 23))
    if mode.get("daz") and (sub1.any() or sub2.any()):
        i = np.flatnonzero(sub1 | sub2)
        a = np.where(sub1[i], np.copysign(0.0, x1[i]), x1[i]).astype(x1.dtype)
        b = None
        if x2 is not None:
            b = np.where(sub2[i], np.copysign(0.0, x2[i]), x2[i]).astype(x2.dtype)
        y0 = impl(a, b) if b is not None else impl(a)
        same = (y[i] == y0) | (np.isnan(y[i]) & np.isnan(y0))
        bad = (e[i] > 0.5) | nanmis[i] | np.isnan(e[i])
        flush[i] |= same & bad
    nanmis = nanmis & ~flush
    ef = np.where(flush, e, np.nan)
    e = np.where(nanmis | flush, np.nan, e)
    return e, y, nanmis, flush, ef


class Best:
    def __init__(self):
        self.err = -1.0; self.x1 = self.x2 = self.y = None; self.nanmis = 0; self.nanx = None
        self.nflush = 0; self.flushx = None; self.flush_err = -1.0; self.flushmx = None
        self.nevals = 0

    def update(self, e, x1, x2, y, nanmis, flush, ef):
        self.nevals += len(e)
        if flush.any():
            self.nflush += int(flush.sum())
            efm = np.where(np.isnan(ef), -1.0, ef)
            m = np.nanmax(efm)
            if m > self.flush_err:
                self.flush_err = float(m)
                i = int(np.argmax(efm))
                self.flushmx = [float(x1[i]).hex(), float(x2[i]).hex() if x2 is not None else None,
                                float(y[i]).hex()]
            if self.flushx is None:
                i = int(np.flatnonzero(flush)[0])
                self.flushx = [float(x1[i]).hex(), float(x2[i]).hex() if x2 is not None else None]
        if nanmis.any():
            self.nanmis += int(nanmis.sum())
            if self.nanx is None:
                i = int(np.flatnonzero(nanmis)[0])
                self.nanx = [float(x1[i]).hex(), float(x2[i]).hex() if x2 is not None else None]
        if np.all(np.isnan(e)):
            return
        i = int(np.nanargmax(e))
        if e[i] > self.err:
            self.err = float(e[i]); self.x1 = float(x1[i]); self.y = float(y[i])
            self.x2 = float(x2[i]) if x2 is not None else None


def search(f, impl, best, rng, T, bivar):
    LIST = 20
    full = (-MAXB + 1, MAXB)            # ordered ints, finite values
    L = [(full, full)] if bivar else [(full, None)]
    batch = 2 * LIST * T

    def width(c):
        w = c[0][1] - c[0][0]
        return w * (c[1][1] - c[1][0]) if bivar else w

    while width(L[0]) > T:
        halves = []
        for cx, cy in L:
            if not bivar or (cx[1] - cx[0]) >= (cy[1] - cy[0]):
                m = cx[0] + (cx[1] - cx[0]) // 2
                halves += [((cx[0], m), cy), ((m, cx[1]), cy)]
            else:
                m = cy[0] + (cy[1] - cy[0]) // 2
                halves += [(cx, (cy[0], m)), (cx, (m, cy[1]))]
        halves = [h for h in halves if h[0][1] > h[0][0] and (not bivar or h[1][1] > h[1][0])]
        o1 = np.concatenate([rng.integers(h[0][0], h[0][1], T) for h in halves])
        x1 = o2f(o1)
        x2 = None
        if bivar:
            x2 = o2f(np.concatenate([rng.integers(h[1][0], h[1][1], T) for h in halves]))
        # pad to a fixed batch size (avoids JAX recompilation)
        n = len(x1)
        if n < batch:
            x1 = np.concatenate([x1, np.full(batch - n, x1[0])])
            if bivar:
                x2 = np.concatenate([x2, np.full(batch - n, x2[0])])
        e, y, nm, fl, ef = errors(f, impl, x1, x2, MODE.get(LIBNAME))
        e, y, nm, fl, ef, x1 = e[:n], y[:n], nm[:n], fl[:n], ef[:n], x1[:n]
        if bivar:
            x2 = x2[:n]
        best.update(e, x1, x2, y, nm, fl, ef)
        d = [np.nanmax(e[k * T:(k + 1) * T]) if not np.all(np.isnan(e[k * T:(k + 1) * T])) else -1.0
             for k in range(len(halves))]
        order = np.argsort(d)[::-1][:LIST]
        L = [halves[k] for k in order]
    # exhaustive search of the remaining intervals
    pts1, pts2 = [], []
    for cx, cy in L:
        a = np.arange(cx[0], cx[1])
        if bivar:
            b = np.arange(cy[0], cy[1])
            pts1.append(np.repeat(a, len(b))); pts2.append(np.tile(b, len(a)))
        else:
            pts1.append(a)
    x1 = o2f(np.concatenate(pts1))
    x2 = o2f(np.concatenate(pts2)) if bivar else None
    e, y, nm, fl, ef = errors(f, impl, x1, x2, MODE.get(LIBNAME))
    best.update(e, x1, x2, y, nm, fl, ef)


impl = {lib: libs.impls(lib, args.device, DT) for lib in args.libs.split(",")}
MODE = {lib: libs.denormal_mode(lib, args.device, DT) for lib in impl}
LIBNAME = None
print("denormal modes:", MODE, flush=True)
json.dump({lib: sorted(d) for lib, d in impl.items()}, open(os.path.join(OUT, "available.json"), "w"), indent=1)
for f in FUNCS:
    bivar = f in libs.BI
    for lib in impl:
        if f not in impl[lib]:
            continue
        p = os.path.join(OUT, f"{lib}_{f}.json")
        if os.path.exists(p):
            continue
        t0 = time.time()
        best = Best()
        rng = np.random.default_rng(20261020)
        # 1. published inputs
        sd = SEEDS.get(f, [])
        if sd:
            x1 = np.array([float.fromhex(s[0]) for s in sd], FT)
            x2 = np.array([float.fromhex(s[1]) for s in sd], FT) if bivar else None
            e, y, nm, fl, ef = errors(f, impl[lib][f], x1, x2, MODE.get(lib))
            best.update(e, x1, x2, y, nm, fl, ef)
        seed_err = best.err
        # 2. uniform random bit patterns
        o = rng.integers(-MAXB + 1, MAXB, args.nrandom)
        x2 = o2f(rng.integers(-MAXB + 1, MAXB, args.nrandom)) if bivar else None
        e, y, nm, fl, ef = errors(f, impl[lib][f], o2f(o), x2, MODE.get(lib))
        best.update(e, o2f(o), x2, y, nm, fl, ef)
        # 3. recursive search, several seeds
        LIBNAME = lib
        for s in range(args.seeds):
            search(f, impl[lib][f], best, np.random.default_rng(1000 + s), args.T, bivar)
        r = dict(func=f, lib=lib, fmt=args.fmt, device=args.device, max_err=best.err,
                 max_str=acc.roundup3(best.err) if best.err >= 0 else None,
                 x1=float(best.x1).hex() if best.x1 is not None else None,
                 x2=float(best.x2).hex() if best.x2 is not None else None,
                 y=float(best.y).hex() if best.y is not None else None,
                 seed_err=seed_err, nanmis=best.nanmis, nan_example=best.nanx,
                 n_flush=best.nflush, flush_example=best.flushx, flush_max=best.flush_err, denormal_mode=MODE.get(lib),
                 flush_x1=best.flushmx[0] if best.flushmx else None, flush_x2=best.flushmx[1] if best.flushmx else None,
                 flush_y=best.flushmx[2] if best.flushmx else None,
                 nevals=best.nevals, seconds=time.time() - t0, T=args.T, nseeds=args.seeds,
                 nrandom=args.nrandom)
        json.dump(r, open(p, "w"), indent=1)
        print(f"{args.fmt} {args.tag} {lib:6s} {f:7s} {r['max_str']} x={r['x1']} {r['x2'] or ''} "
              f"(seeds {acc.roundup3(seed_err) if seed_err >= 0 else '-'}) nanmis={best.nanmis} "
              f"evals={best.nevals} {r['seconds']:.0f}s", flush=True)
print("DONE", args.fmt, args.tag)
