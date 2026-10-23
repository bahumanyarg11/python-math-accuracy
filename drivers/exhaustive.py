"""Exhaustive univariate check (binary32: all 4278190080 finite inputs;
binary16 / bfloat16: all finite inputs) and exhaustive bivariate check for
the 16-bit formats (all pairs of finite inputs).  Resumable: state is saved
per (lib, func) after every few chunks under results/<fmt>/<tag>/.

usage: exhaustive.py --fmt binary32 --device cpu --tag x86_cpu [--libs ...] [--funcs ...]
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import numpy as np
import acc, inputs, libs

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", default="binary32")
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", required=True)
ap.add_argument("--libs", default="numpy,scipy,jax,torch")
ap.add_argument("--funcs", default=",".join(libs.UNI))
ap.add_argument("--bivariate", action="store_true", help="16-bit formats: all pairs")
ap.add_argument("--chunks", default=None, help="a:b subset of chunks (testing)")
ap.add_argument("--save_every", type=int, default=8)
args = ap.parse_args()
DT = {"binary32": "float32", "binary16": "binary16", "bfloat16": "bfloat16"}[args.fmt]
OUT = os.path.join(HERE, "..", "results", args.fmt + ("_bivar" if args.bivariate else ""), args.tag)
os.makedirs(OUT, exist_ok=True)
json.dump(libs.versions(), open(os.path.join(OUT, "versions.json"), "w"), indent=1)


def small_inputs(fmt):
    if fmt == "binary16":
        b = np.arange(0, 0x7C00, dtype=np.uint16)
        x = np.concatenate([b, b | 0x8000]).view(np.float16).astype(np.float32)
    else:
        b = np.arange(0, 0x7F80, dtype=np.uint32)
        x = (np.concatenate([b, b | 0x8000]) << 16).astype(np.uint32).view(np.float32)
    return x


if args.fmt == "binary32":
    NCH = inputs.NCH
    get = lambda c: (inputs.chunk_inputs(c), None)
elif not args.bivariate:
    NCH = 1
    get = lambda c: (small_inputs(args.fmt), None)
else:
    XS = small_inputs(args.fmt)
    ROWS = 256
    NCH = (len(XS) + ROWS - 1) // ROWS
    def get(c):
        a = XS[c * ROWS:(c + 1) * ROWS]
        return np.repeat(a, len(XS)), np.tile(XS, len(a))

impl = {lib: libs.impls(lib, args.device, DT) for lib in args.libs.split(",")}
avail = {lib: sorted(d) for lib, d in impl.items()}
json.dump(avail, open(os.path.join(OUT, "available.json"), "w"), indent=1)
print("available:", avail, flush=True)
funcs = [f for f in args.funcs.split(",") if (f in libs.BI) == args.bivariate]
c0, c1 = (0, NCH) if not args.chunks else map(int, args.chunks.split(":"))

for f in funcs:
    todo = [lib for lib in impl if f in impl[lib]]
    if not todo:
        continue
    st = {}
    for lib in todo:
        p = os.path.join(OUT, f"{lib}_{f}.json")
        st[lib] = json.load(open(p)) if os.path.exists(p) else None
    todo = [lib for lib in todo if not (st[lib] and st[lib].get("final"))]
    if not todo:
        continue
    res = {lib: acc.res_from_dict(st[lib]["res"]) if st[lib] else acc.Res() for lib in todo}
    done = {lib: st[lib]["done"] if st[lib] else c0 for lib in todo}
    secs = {lib: st[lib].get("seconds", 0) if st[lib] else 0 for lib in todo}
    start = min(done.values())
    t0 = time.time()
    for c in range(start, c1):
        x1, x2 = get(c)
        zref = None
        if not acc.has_fast(f):
            zref = acc.cr_mt(f, args.fmt, x1, x2)
        for lib in todo:
            if done[lib] > c:
                continue
            t = time.time()
            y = impl[lib][f](x1, x2) if x2 is not None else impl[lib][f](x1)
            secs[lib] += time.time() - t
            acc.chunk(f, args.fmt, x1, y, x2=x2, res=res[lib], zref=zref)
            done[lib] = c + 1
        if (c + 1) % args.save_every == 0 or c + 1 == c1:
            for lib in todo:
                p = os.path.join(OUT, f"{lib}_{f}.json")
                d = dict(func=f, lib=lib, fmt=args.fmt, device=args.device, done=done[lib],
                         nchunks=NCH, seconds=secs[lib], res=acc.res_to_dict(res[lib]))
                if done[lib] == NCH:
                    d["final"] = acc.finalize(f, args.fmt, res[lib])
                json.dump(d, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
    for lib in todo:
        p = os.path.join(OUT, f"{lib}_{f}.json")
        d = json.load(open(p))
        fin = d.get("final")
        print(f"{args.fmt} {args.tag} {lib:6s} {f:7s}", (fin or {}).get("max_str"), "x=", (fin or {}).get("x1"),
              f"mis={(fin or {}).get('n_mis')} flush={(fin or {}).get('n_flush')} spnan={(fin or {}).get('n_spnan')} misnan={(fin or {}).get('n_misnan')}",
              f"[{time.time()-t0:.0f}s]", flush=True)
print("DONE", args.fmt, args.tag)
