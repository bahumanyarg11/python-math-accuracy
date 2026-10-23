"""Stage-1 gate: reproduce published numbers of Gladman et al. with our
error engine (harness/acc.c), on x86_64 Linux.
  glibc (system, 2.39 on Ubuntu 24.04) vs latex/glibc239.tex (math_accuracy c2b897fb)
  OpenLibm 0.8.7 vs the August 2026 paper (Tables 1 and 7)."""
import sys, os, json, ctypes, time, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import numpy as np, acc, inputs
LIB = sys.argv[1]            # glibc | openlibm
WHICH = sys.argv[2] if len(sys.argv) > 2 else "all"   # b32 | b64 | all
L = ctypes.CDLL(os.path.join(HERE, f"libgate_{LIB}.so"))
F = ctypes.POINTER(ctypes.c_float); D = ctypes.POINTER(ctypes.c_double)
B32 = {"glibc": {"exp": "0.502", "log": "0.818", "sin": "0.561", "cos": "0.561",
                 "tanh": "2.19", "log10": "2.07", "cosh": "1.89", "expm1": "0.813",
                 "erfc": "3.13", "acosh": "2.01"},
       "openlibm": {"acos": "0.918", "cosh": "1.36", "erf": "0.943", "exp": "0.911",
                    "log10": "0.832", "tanh": "2.19", "asinh": "1.78", "log": "0.888"}}
B64 = json.load(open(os.path.join(HERE, {"glibc": "glibc239_b64.json",
                                          "openlibm": "openlibm087_b64.json"}[LIB])))
out = {"lib": LIB, "platform": platform.platform(), "machine": platform.machine(),
       "libc": platform.libc_ver(), "b32": {}, "b64": {}}
ok = True
if WHICH in ("b32", "all"):
    for f, exp in B32[LIB].items():
        t = time.time(); res = acc.Res()
        y = np.empty(inputs.CH, np.float32)
        for c in range(inputs.NCH):
            x = inputs.chunk_inputs(c)
            yy = y[:len(x)]
            L.call32(f.encode(), x.ctypes.data_as(F), None, yy.ctypes.data_as(F), ctypes.c_long(len(x)))
            acc.chunk(f, "binary32", x, yy, res=res)
        r = acc.finalize(f, "binary32", res)
        match = r["max_str"] == exp
        ok &= match
        r.update(published=exp, match=match, seconds=time.time() - t)
        out["b32"][f] = r
        print(f"b32 {LIB} {f}: ours {r['max_str']} (x={r.get('x1')}) published {exp} {'OK' if match else 'MISMATCH'}"
              f"  [n={r['n']} mis={r['n_mis']} flush={r['n_flush']} spnan={r['n_spnan']}] {r['seconds']:.0f}s", flush=True)
if WHICH in ("b64", "all"):
    for f, e in B64.items():
        if f == "sqrt": continue
        if LIB == "openlibm" and f == "exp10": continue
        x = np.array([float.fromhex(e["x"])]); x2 = np.array([float.fromhex(e["y"])]) if e.get("y") else None
        z = np.empty(1)
        if L.call64(f.encode(), x.ctypes.data_as(D), x2.ctypes.data_as(D) if x2 is not None else None,
                    z.ctypes.data_as(D), ctypes.c_long(1)) < 0:
            continue
        err = acc.err64(f, x, z, x2)[0]
        s = acc.roundup3(err); pub = e["err"].rstrip(".")
        match = s == pub or (pub.endswith("(CR)") and err <= 0.5)
        ok &= match
        out["b64"][f] = dict(x=e["x"], y=e.get("y"), z=float(z[0]).hex(), err=err, ours=s, published=e["err"], match=match)
        print(f"b64 {LIB} {f}: ours {s} ({err:.6g}) published {e['err']} {'OK' if match else 'MISMATCH'}", flush=True)
out["all_match"] = bool(ok)
json.dump(out, open(os.path.join(HERE, f"gate_{LIB}_{WHICH}.json"), "w"), indent=1)
print("GATE", LIB, WHICH, "ALL MATCH" if ok else "SOME MISMATCH")
