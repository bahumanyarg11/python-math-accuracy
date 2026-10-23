"""ctypes wrapper of libacc (see acc.c)."""
import ctypes, os, math
from decimal import Decimal, ROUND_CEILING
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = "dylib" if os.uname().sysname == "Darwin" else "so"
L = ctypes.CDLL(os.path.join(ROOT, "build", f"libacc.{EXT}"))
NTHREADS = int(os.environ.get("ACC_THREADS", os.cpu_count()))
FMT = {"binary32": 0, "binary16": 1, "bfloat16": 2}
K = 32


class Res(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in
                "n n_mis n_err2 n_spnan n_misnan n_flush n_yinf n_badinf n_mpfr".split()] + [
        ("k", ctypes.c_int),
        ("top_u", ctypes.c_uint64 * K), ("top_d", ctypes.c_double * K),
        ("top_x1", ctypes.c_float * K), ("top_x2", ctypes.c_float * K),
        ("top_y", ctypes.c_float * K),
        ("fl_d", ctypes.c_double), ("fl_x1", ctypes.c_float), ("fl_x2", ctypes.c_float),
        ("fl_y", ctypes.c_float),
        ("sn_x1", ctypes.c_float), ("sn_x2", ctypes.c_float), ("sn_y", ctypes.c_float),
        ("mn_x1", ctypes.c_float), ("mn_x2", ctypes.c_float), ("mn_y", ctypes.c_float)]


assert ctypes.sizeof(Res) == L.acc_res_size(), (ctypes.sizeof(Res), L.acc_res_size())
_F = ctypes.POINTER(ctypes.c_float)
_D = ctypes.POINTER(ctypes.c_double)
L.acc_err_exact.restype = ctypes.c_double
L.acc_err_exact.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.POINTER(ctypes.c_uint64), _D]


def fid(name):
    i = L.acc_func_id(name.encode())
    if i < 0:
        raise KeyError(name)
    return i


def bivariate(name):
    return bool(L.acc_is_bivariate(fid(name)))


def chunk(name, fmt, x1, y, x2=None, res=None, zref=None):
    """accumulate errors of library outputs y (float32 array) at inputs x1 (x2);
    zref: optional precomputed correctly rounded values (float32)"""
    if res is None:
        res = Res()
    pz = None
    if zref is not None:
        zref = np.ascontiguousarray(zref, np.float32); pz = zref.ctypes.data_as(_F)
    x1 = np.ascontiguousarray(x1, np.float32)
    y = np.ascontiguousarray(y, np.float32)
    p2 = None
    if x2 is not None:
        x2 = np.ascontiguousarray(x2, np.float32)
        p2 = x2.ctypes.data_as(_F)
    L.acc_chunk(fid(name), FMT[fmt], x1.ctypes.data_as(_F), p2, y.ctypes.data_as(_F), pz,
                ctypes.c_long(len(x1)), ctypes.c_int(NTHREADS), ctypes.byref(res))
    return res


def err_exact(name, fmt, x1, y, x2=0.0):
    u = ctypes.c_uint64(); z = ctypes.c_double()
    d = L.acc_err_exact(fid(name), FMT[fmt], float(x1), float(x2), float(y),
                        ctypes.byref(u), ctypes.byref(z))
    return d, u.value, z.value


def cr_array(name, fmt, x1, x2=None, fast=True):
    x1 = np.ascontiguousarray(x1, np.float64)
    z = np.empty(len(x1))
    p2 = None
    if x2 is not None:
        x2 = np.ascontiguousarray(x2, np.float64); p2 = x2.ctypes.data_as(_D)
    if not fast or L.acc_cr_fast_array(fid(name), FMT[fmt], x1.ctypes.data_as(_D), p2,
                                       z.ctypes.data_as(_D), ctypes.c_long(len(x1))) < 0:
        L.acc_cr_array(fid(name), FMT[fmt], x1.ctypes.data_as(_D), p2,
                       z.ctypes.data_as(_D), ctypes.c_long(len(x1)))
    return z


def err64(name, x1, z, x2=None):
    """distance() of check_sample2.c for library outputs z (float64)"""
    x1 = np.ascontiguousarray(x1, np.float64)
    z = np.ascontiguousarray(z, np.float64)
    e = np.empty(len(x1))
    p2 = None
    if x2 is not None:
        x2 = np.ascontiguousarray(x2, np.float64); p2 = x2.ctypes.data_as(_D)
    L.acc_err64(fid(name), x1.ctypes.data_as(_D), p2, z.ctypes.data_as(_D),
                e.ctypes.data_as(_D), ctypes.c_long(len(x1)), ctypes.c_int(NTHREADS))
    return e


def mpfr_str(name, x1, x2=0.0, prec=256):
    buf = ctypes.create_string_buffer(200)
    L.acc_mpfr_str(fid(name), ctypes.c_double(x1), ctypes.c_double(x2), prec, buf, 200)
    return buf.value.decode()


def roundup3(e):
    """3 significant digits rounded up, as their %.2RUe"""
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "NaN"
    if math.isinf(e):
        return "Inf"
    if e == 0:
        return "0"
    d = Decimal(repr(float(e)))
    r = d.quantize(Decimal(1).scaleb(d.adjusted() - 2), rounding=ROUND_CEILING)
    r = r.quantize(Decimal(1).scaleb(r.adjusted() - 2), rounding=ROUND_CEILING)
    a = r.adjusted()
    if -3 <= a <= 2:
        return f"{r:.{max(0, 2 - a)}f}"
    m = r.scaleb(-a)
    return f"{m:.2f}e{a}"


def finalize(name, fmt, res):
    """exact recomputation of the top candidates; returns dict"""
    cands = []
    for i in range(res.k):
        x1, x2, y = res.top_x1[i], res.top_x2[i], res.top_y[i]
        d, u, z = err_exact(name, fmt, x1, y, x2)
        cands.append((u, d, x1, x2, y, z))
    out = {k: getattr(res, k) for k in
           "n n_mis n_err2 n_spnan n_misnan n_flush n_yinf n_badinf n_mpfr".split()}
    if cands:
        u, d, x1, x2, y, z = max(cands, key=lambda c: (c[0], c[1]))
        out.update(max_err=d, max_u=u, x1=float(x1).hex(), x2=float(x2).hex(),
                   y=float(y).hex(), cr=float(z).hex(), max_str=roundup3(d))
    else:
        out.update(max_err=0.0, max_str="0.500" if res.n else None)
        # all outputs correctly rounded: error <= 0.5 ulp ("0.500" in their tables)
    if res.n_flush:
        d, u, z = err_exact(name, fmt, res.fl_x1, res.fl_y, res.fl_x2)
        out.update(flush_max=d, flush_x1=float(res.fl_x1).hex(), flush_x2=float(res.fl_x2).hex(),
                   flush_y=float(res.fl_y).hex())
    if res.n_spnan:
        out.update(spnan_x1=float(res.sn_x1).hex(), spnan_x2=float(res.sn_x2).hex())
    if res.n_misnan:
        out.update(misnan_x1=float(res.mn_x1).hex(), misnan_x2=float(res.mn_x2).hex(),
                   misnan_y=float(res.mn_y).hex())
    return out


def res_to_dict(res):
    return {f: (list(getattr(res, f)) if hasattr(getattr(res, f), "__len__") else getattr(res, f))
            for f, _ in Res._fields_}


def res_from_dict(d):
    r = Res()
    for f, _ in Res._fields_:
        v = d[f]
        if isinstance(v, list):
            arr = getattr(r, f)
            for i, a in enumerate(v):
                arr[i] = a
        else:
            setattr(r, f, v)
    return r


def has_fast(name):
    return bool(L.acc_has_fast(fid(name)))


def cr_mt(name, fmt, x1, x2=None):
    x1 = np.ascontiguousarray(x1, np.float32); z = np.empty(len(x1), np.float32)
    p2 = None
    if x2 is not None:
        x2 = np.ascontiguousarray(x2, np.float32); p2 = x2.ctypes.data_as(_F)
    L.acc_cr_mt(fid(name), FMT[fmt], x1.ctypes.data_as(_F), p2, z.ctypes.data_as(_F),
                ctypes.c_long(len(x1)), ctypes.c_int(NTHREADS))
    return z


def err_mt(name, fmt, x1, y, x2=None):
    """exact per-point errors (ulp_error_double); NaN mismatch -> inf"""
    x1 = np.ascontiguousarray(x1, np.float32); y = np.ascontiguousarray(y, np.float32)
    e = np.empty(len(x1))
    p2 = None
    if x2 is not None:
        x2 = np.ascontiguousarray(x2, np.float32); p2 = x2.ctypes.data_as(_F)
    L.acc_err_mt(fid(name), FMT[fmt], x1.ctypes.data_as(_F), p2, y.ctypes.data_as(_F),
                 e.ctypes.data_as(_D), ctypes.c_long(len(x1)), ctypes.c_int(NTHREADS))
    return e
