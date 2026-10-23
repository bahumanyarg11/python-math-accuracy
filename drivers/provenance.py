"""Backend provenance by bitwise comparison (evidence for 'where each result
comes from').  For each library function and dtype, on 2^22 inputs (random
bit patterns + random values in [-10,10]; all finite inputs for 16-bit
formats), the library output is compared bit-for-bit with candidates:
  libm32 / libm64 : the platform C math library (float / double functions)
  via_f64         : the library's own float64 result rounded to float32
  via_f32         : (16-bit) the library's own float32 result rounded to the format
  numpy           : NumPy's result for the same dtype
A candidate 'explains' the output if 100% of outputs agree (NaNs equal).
usage: provenance.py --device cpu --tag x86_cpu
"""
import sys, os, json, ctypes, argparse, ctypes.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import numpy as np
import libs

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", required=True)
ap.add_argument("--libs", default="numpy,scipy,jax,torch")
args = ap.parse_args()
OUT = os.path.join(HERE, "..", "results", "provenance")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
N = 1 << 22

# C library
lm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
CNAMES = {f: f for f in libs.UNI}
CNAMES.update(pow="pow", atan2="atan2", hypot="hypot")


def libm(f, x, x2, single):
    name = CNAMES[f] + ("f" if single else "")
    try:
        fn = getattr(lm, name)
    except AttributeError:
        return None
    T = ctypes.c_float if single else ctypes.c_double
    fn.restype = T
    fn.argtypes = [T] * (2 if x2 is not None else 1)
    if x2 is not None:
        return np.array([fn(a, b) for a, b in zip(x, x2)], np.float32 if single else np.float64)
    return np.array([fn(a) for a in x], np.float32 if single else np.float64)


def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean((a == b) | (np.isnan(a) & np.isnan(b))))


def inputs32(n):
    b = rng.integers(0, 2**32, n // 2, dtype=np.uint64).astype(np.uint32).view(np.float32)
    v = rng.uniform(-10, 10, n - n // 2).astype(np.float32)
    x = np.concatenate([b, v]); x[~np.isfinite(x)] = 1.0
    return x


def inputs64(n):
    b = rng.integers(0, 2**63, n // 2, dtype=np.uint64).view(np.float64) * rng.choice([-1.0, 1.0], n // 2)
    v = rng.uniform(-10, 10, n - n // 2)
    x = np.concatenate([b, v]); x[~np.isfinite(x)] = 1.0
    return x


def small(fmt):
    if fmt == "binary16":
        b = np.arange(0, 0x7C00, dtype=np.uint16)
        return np.concatenate([b, b | 0x8000]).view(np.float16).astype(np.float32)
    b = np.arange(0, 0x7F80, dtype=np.uint32)
    return (np.concatenate([b, b | 0x8000]) << 16).astype(np.uint32).view(np.float32)


def round_small(y32, fmt):
    if fmt == "binary16":
        return y32.astype(np.float16).astype(np.float32)
    import ml_dtypes
    return y32.astype(ml_dtypes.bfloat16).astype(np.float32)


res = {}
NLIBM = 1 << 16   # libm comparison via ctypes is slow: first 65536 points
for lib in args.libs.split(","):
    I32 = libs.impls(lib, args.device, "float32")
    I64 = libs.impls(lib, args.device, "float64")
    I16 = libs.impls(lib, args.device, "binary16")
    IBF = libs.impls(lib, args.device, "bfloat16")
    N32 = libs.impls("numpy", "cpu", "float32") if lib != "numpy" else {}
    for f in libs.ALL:
        bi = f in libs.BI
        r = {}
        if f in I32:
            x = inputs32(N); x2 = inputs32(N) if bi else None
            y = I32[f](x, x2) if bi else I32[f](x)
            c = libm(f, x[:NLIBM], x2[:NLIBM] if bi else None, True) if args.device == "cpu" else None
            if c is not None:
                r["f32_eq_libm32"] = same(y[:NLIBM], c)
                c64 = libm(f, x[:NLIBM].astype(np.float64), x2[:NLIBM].astype(np.float64) if bi else None, False)
                r["f32_eq_libm64_rounded"] = same(y[:NLIBM], c64.astype(np.float32))
            if f in I64:
                y64 = I64[f](x.astype(np.float64), x2.astype(np.float64)) if bi else I64[f](x.astype(np.float64))
                r["f32_eq_own_f64_rounded"] = same(y, y64.astype(np.float32))
            if f in N32:
                r["f32_eq_numpy"] = same(y, N32[f](x, x2) if bi else N32[f](x))
        if f in I64 and args.device == "cpu":
            x = inputs64(NLIBM); x2 = inputs64(NLIBM) if bi else None
            y = I64[f](x, x2) if bi else I64[f](x)
            c = libm(f, x, x2, False)
            if c is not None:
                r["f64_eq_libm64"] = same(y, c)
        for fmt, I in [("binary16", I16), ("bfloat16", IBF)]:
            if f in I:
                x = small(fmt)
                if bi:
                    x2 = rng.choice(x, len(x)); x = rng.choice(x, len(x))
                y = I[f](x, x2) if bi else I[f](x)
                if f in I32:
                    y32 = I32[f](x, x2) if bi else I32[f](x)
                    r[f"{fmt}_eq_own_f32_rounded"] = same(y, round_small(y32, fmt))
                if f in I64:
                    y64 = I64[f](x.astype(np.float64), x2.astype(np.float64)) if bi else I64[f](x.astype(np.float64))
                    r[f"{fmt}_eq_own_f64_rounded"] = same(y, round_small(y64.astype(np.float32), fmt))
        if r:
            res[f"{lib}/{f}"] = r
            print(lib, f, r, flush=True)
json.dump(dict(tag=args.tag, device=args.device, versions=libs.versions(), results=res),
          open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
print("DONE provenance", args.tag)
