"""Regenerate every table (LaTeX + CSV) from the raw results in results/.
usage: python3 make_tables.py   -> tables/*.tex, tables/*.csv, tables/summary.md
"""
import os, sys, json, glob, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness"))
from acc import roundup3
ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
T = os.path.join(ROOT, "tables")
os.makedirs(T, exist_ok=True)
UNI = ("exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan "
       "sinh cosh tanh asinh acosh atanh cbrt sqrt rsqrt erf erfc lgamma tgamma j0 j1 y0 y1").split()
BI = ["atan2", "hypot", "pow"]
LIBN = {"numpy": "NumPy", "scipy": "SciPy", "jax": "JAX", "torch": "PyTorch"}
PLAT = {"x86_cpu": "x86-64 CPU", "mac_arm64_cpu": "Arm64 CPU (M4)", "t4_cuda": "CUDA (T4)",
        "mac_mps": "MPS (M4)"}
PLAT_ORDER = ["x86_cpu", "mac_arm64_cpu", "t4_cuda", "mac_mps"]


def denormal_modes():
    m = {}
    for p in glob.glob(os.path.join(R, "provenance", "denormals_*.json")):
        m[os.path.basename(p)[len("denormals_"):-5]] = json.load(open(p))
    return m


DMODE = denormal_modes()
KDT = {"binary32": "float32", "binary16": "binary16", "bfloat16": "bfloat16",
       "binary16_bivar": "binary16", "bfloat16_bivar": "bfloat16"}


def load(kind):
    """kind: binary32, binary16, bfloat16, binary16_bivar, bfloat16_bivar,
    binary64_search, binary32_search -> {(plat, lib, func): record}"""
    out = {}
    for p in glob.glob(os.path.join(R, kind, "*", "*_*.json")):
        plat = os.path.basename(os.path.dirname(p))
        b = os.path.basename(p)[:-5]
        if b in ("versions", "available"):
            continue
        lib, func = b.split("_", 1)
        d = json.load(open(p))
        if "nchunks" in d:          # exhaustive record
            if d.get("done") != d.get("nchunks") or not d.get("final"):
                continue            # partial run: shown as "?"
            rec = dict(d["final"]); rec["exhaustive"] = True
            # flush category only where the backend really flushes (probe);
            # otherwise zero results / subnormal-input cases are ordinary errors
            mode = (DMODE.get(plat, {}).get(lib) or {}).get(KDT.get(kind, ""))
            if rec.get("n_flush") and not (mode and (mode.get("ftz") or mode.get("daz"))):
                fm = rec.get("flush_max")
                if fm is not None and fm != float("inf") and fm > rec.get("max_err", 0):
                    rec["max_err"] = fm
                    rec["max_str"] = roundup3(fm)
                    rec.update(x1=rec.get("flush_x1"), x2=rec.get("flush_x2"), y=rec.get("flush_y"))
                elif fm == float("inf"):
                    rec["n_spnan"] = (rec.get("n_spnan") or 0) + 1
                rec["n_flush"] = 0
                rec["flush_reclassified"] = True
        else:
            rec = dict(d); rec["exhaustive"] = False
        # value under the rules of Gladman et al. (their Sec. 1): an input where
        # exactly one of library/exact result is NaN gives an infinite error, and
        # flushed results are ordinary errors.  max_err itself excludes the cases
        # caused by a documented FTZ/DAZ mode and the NaN mismatches.
        fm = rec.get("flush_max") if rec.get("n_flush") else None
        if fm is not None and fm != fm:      # NaN: a flushed case where only one side is NaN
            fm = float("inf")
        if rec.get("n_spnan") or rec.get("nanmis") or rec.get("n_misnan"):
            rec["strict"] = float("inf")
        else:
            rec["strict"] = max(0.5, rec.get("max_err") or 0.0, fm if fm is not None else 0.0)
        out[(plat, lib, func)] = rec
    return out


def avail(kind):
    a = {}
    for p in glob.glob(os.path.join(R, kind, "*", "available.json")):
        a[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    return a


def cell(rec, tex=True):
    if rec is None:
        return None
    s = rec.get("max_str") or "0.500"
    st = roundup3(rec["strict"])
    if st != s:
        s = f"{st} ({s})"
    marks = ""
    if rec.get("n_flush"):
        marks += "f"
    if rec.get("n_spnan") or rec.get("nanmis"):
        marks += "n"
    if rec.get("n_misnan"):
        marks += "m"
    if tex and marks:
        return f"{s}$^{{\\rm {marks}}}$"
    return s + (f" [{marks}]" if marks else "")


def table(kind, funcs, caption, label, fname, plats=None):
    data = load(kind)
    av = avail(kind)
    plats = [p for p in (plats or PLAT_ORDER) if p in av]
    cols = []
    for p in plats:
        for lib in ["numpy", "scipy", "jax", "torch"]:
            if lib in av[p] and av[p][lib]:
                cols.append((p, lib))
    plats = [p for p in plats if any(c[0] == p for c in cols)]   # drop platforms with no column
    rows = []
    for f in funcs:
        if not any((p, l, f) in data or f in av[p].get(l, []) for p, l in cols):
            continue
        r = [f]
        for p, l in cols:
            if (p, l, f) in data:
                r.append(cell(data[(p, l, f)]))
            elif f in av[p].get(l, []):
                r.append("?")       # available but not (yet) measured
            else:
                r.append("--")
        rows.append(r)
    # CSV
    with open(os.path.join(T, fname + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function"] + [f"{LIBN[l]} {PLAT.get(p, p)}" for p, l in cols])
        for r in rows:
            w.writerow([x.replace("$^{\\rm ", " [").replace("}$", "]") if isinstance(x, str) else x for x in r])
    # inputs CSV (hex)
    with open(os.path.join(T, fname + "_inputs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["platform", "library", "function", "value_gladman_rules", "max_err_excluding_flush_and_nan",
                    "x", "y_input2", "library_output", "n_mismatch", "n_flush", "flush_max", "flush_x", "flush_y_input2",
                    "flush_library_output", "n_spurious_nan", "spurious_nan_example", "n_missing_nan",
                    "missing_nan_example", "exhaustive"])
        for (p, l, f), rec in sorted(data.items()):
            ne = rec.get("nan_example") or [rec.get("spnan_x1"), rec.get("spnan_x2")]
            w.writerow([p, l, f, roundup3(rec["strict"]), rec.get("max_err"), rec.get("x1"), rec.get("x2"), rec.get("y"),
                        rec.get("n_mis"), rec.get("n_flush"), rec.get("flush_max"), rec.get("flush_x1"),
                        rec.get("flush_x2"), rec.get("flush_y"), rec.get("n_spnan", rec.get("nanmis")),
                        " ".join(str(v) for v in ne if v), rec.get("n_misnan"),
                        rec.get("misnan_x1"), rec["exhaustive"]])
    # LaTeX
    ncol = len(cols)
    lines = ["\\begin{table}", "\\centering", "\\begin{small}",
             "\\begin{tabular}{|l|" + "|".join(["c" * sum(1 for c in cols if c[0] == p) for p in plats]) + "|}",
             "\\hline"]
    lines.append("platform & " + " & ".join(
        f"\\multicolumn{{{sum(1 for c in cols if c[0] == p)}}}{{c|}}{{{PLAT.get(p, p)}}}" for p in plats) + " \\\\")
    lines.append("library & " + " & ".join(LIBN[l] for _, l in cols) + " \\\\ \\hline")
    for r in rows:
        if r[0] == BI[0] and funcs[0] != BI[0]:
            lines.append("\\hline")
        lines.append(" & ".join([r[0]] + r[1:]) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{small}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\end{table}"]
    open(os.path.join(T, fname + ".tex"), "w").write("\n".join(lines) + "\n")
    return cols, rows


def md(cols, rows, title):
    s = [f"### {title}", "", "| function | " + " | ".join(f"{LIBN[l]} {PLAT.get(p, p)}" for p, l in cols) + " |",
         "|---" * (len(cols) + 1) + "|"]
    for r in rows:
        s.append("| " + " | ".join(str(x).replace("$^{\\rm ", " [").replace("}$", "]") for x in r) + " |")
    return "\n".join(s) + "\n"


NOTE = ("Values: largest error in ulps under the rules of Gladman et al. (3 significant digits rounded up; "
        "an input where only one of the library result and the exact result is NaN gives Inf). "
        "In parentheses, when different: the largest error excluding results affected by a flush-to-zero/"
        "denormals-are-zero mode of the backend and NaN mismatches. "
        "0.500 = correctly rounded on all inputs; -- = function not provided for this format/backend; "
        "marks: f = some results flushed by the backend's FTZ/DAZ mode, "
        "n = spurious NaN for some inputs, m = NaN missing (number returned where the result is NaN). ")
out = ["# PYACC tables (generated by make_tables.py)", "", NOTE, ""]
c, r = table("binary32", UNI, "Binary32: largest error over all inputs (exhaustive). " + NOTE,
             "tab:b32", "binary32_exhaustive")
out.append(md(c, r, "binary32, univariate, exhaustive (true maxima)"))
c, r = table("binary32_search", BI, "Binary32 bivariate: largest known error (lower bound). " + NOTE,
             "tab:b32bi", "binary32_bivariate")
out.append(md(c, r, "binary32, bivariate (lower bounds)"))
c, r = table("binary64_search", UNI + BI, "Binary64: largest known error (lower bounds). " + NOTE,
             "tab:b64", "binary64")
out.append(md(c, r, "binary64 (lower bounds)"))
for fmt in ["binary16", "bfloat16"]:
    c, r = table(fmt, UNI, f"{fmt}: largest error over all inputs (exhaustive). " + NOTE,
                 f"tab:{fmt}", f"{fmt}_exhaustive")
    out.append(md(c, r, f"{fmt}, univariate, exhaustive"))
    c, r = table(fmt + "_bivar", BI, f"{fmt} bivariate: largest error over all pairs (exhaustive). " + NOTE,
                 f"tab:{fmt}bi", f"{fmt}_bivariate")
    out.append(md(c, r, f"{fmt}, bivariate, exhaustive over all pairs"))
open(os.path.join(T, "summary.md"), "w").write("\n".join(out))
print("\n".join(out))

# ---- tables for the paper (paper/tables/*.tex): CPU and GPU platforms split
PT = os.path.join(ROOT, "paper", "tables")
os.makedirs(PT, exist_ok=True)
_T = T
T = PT
CPU, GPU = ["x86_cpu", "mac_arm64_cpu"], ["t4_cuda", "mac_mps"]
SHORT = ("Largest error in ulps (rules of Gladman et al.); in parentheses, when different, "
         "the value excluding FTZ/DAZ-affected results and NaN mismatches. "
         "Marks: f flushed by the FTZ/DAZ mode, n spurious NaN, m missing NaN; -- not provided.")
for kind, funcs, what, lab in [("binary32", UNI, "binary32, all inputs (exhaustive)", "b32"),
                               ("binary32_search", BI, "binary32 bivariate, largest known error (lower bounds)", "b32bi"),
                               ("binary64_search", UNI + BI, "binary64, largest known error (lower bounds)", "b64"),
                               ("binary16", UNI, "binary16, all inputs (exhaustive)", "b16"),
                               ("binary16_bivar", BI, "binary16 bivariate, all pairs of inputs (exhaustive)", "b16bi"),
                               ("bfloat16", UNI, "bfloat16, all inputs (exhaustive)", "bf16"),
                               ("bfloat16_bivar", BI, "bfloat16 bivariate, all pairs of inputs (exhaustive)", "bf16bi")]:
    for side, plats in [("cpu", CPU), ("gpu", GPU)]:
        av = avail(kind)
        if not any(p in av for p in plats):
            continue
        table(kind, funcs, f"{'CPUs' if side == 'cpu' else 'GPUs'}: {what}. {SHORT}",
              f"tab:{lab}{side}", f"{lab}_{side}", plats=plats)
T = _T
