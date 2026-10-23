"""Library adapters: impls(lib, device, dtype) -> {func: callable}.

Every callable takes numpy arrays (float32 for binary32 / binary16-as-float32 /
bfloat16-as-float32 inputs, float64 for binary64) and returns the library
result converted *exactly* to float32 (small formats) or float64.  Libraries
are always called in batched, vectorised form."""
import os
import numpy as np
np.seterr(all="ignore")

UNI = ("exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan "
       "sinh cosh tanh asinh acosh atanh cbrt sqrt rsqrt erf erfc lgamma tgamma j0 j1 y0 y1").split()
BI = ["pow", "atan2", "hypot"]
ALL = UNI + BI

NP_NAMES = dict(exp="exp", exp2="exp2", expm1="expm1", log="log", log2="log2", log10="log10",
                log1p="log1p", sin="sin", cos="cos", tan="tan", asin="arcsin", acos="arccos",
                atan="arctan", sinh="sinh", cosh="cosh", tanh="tanh", asinh="arcsinh",
                acosh="arccosh", atanh="arctanh", cbrt="cbrt", sqrt="sqrt", pow="power", atan2="arctan2",
                hypot="hypot")
SC_NAMES = dict(exp2="exp2", exp10="exp10", expm1="expm1", log1p="log1p", cbrt="cbrt", erf="erf",
                erfc="erfc", lgamma="gammaln", tgamma="gamma", j0="j0", j1="j1", y0="y0", y1="y1")
TORCH_NAMES = dict(exp="exp", exp2="exp2", expm1="expm1", log="log", log2="log2", log10="log10",
                   log1p="log1p", sin="sin", cos="cos", tan="tan", asin="asin", acos="acos",
                   atan="atan", sinh="sinh", cosh="cosh", tanh="tanh", asinh="asinh", sqrt="sqrt", rsqrt="rsqrt",
                   acosh="acosh", atanh="atanh", erf="special.erf", erfc="special.erfc",
                   lgamma="lgamma", j0="special.bessel_j0", j1="special.bessel_j1",
                   y0="special.bessel_y0", y1="special.bessel_y1", pow="pow", atan2="atan2",
                   hypot="hypot")
JAX_SPECIAL = dict(erf="erf", erfc="erfc", lgamma="gammaln", tgamma="gamma")


def _bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_bits(x):  # x is exactly representable in bfloat16
    b = np.ascontiguousarray(x, np.float32).view(np.uint32)
    assert not np.any(b & 0xFFFF), "input not a bfloat16 value"
    return (b >> 16).astype(np.uint16)


# ------------------------------------------------------------------ numpy
def numpy_impls(device, dtype):
    if device != "cpu" or dtype == "bfloat16":
        return {}
    dt = {"float32": np.float32, "float64": np.float64, "binary16": np.float16}.get(dtype)
    out = {}
    for f, n in NP_NAMES.items():
        u = getattr(np, n)
        if f in BI:
            out[f] = (lambda u: lambda x, y: u(x.astype(dt), y.astype(dt)).astype(
                np.float64 if dt == np.float64 else np.float32))(u)
        else:
            out[f] = (lambda u: lambda x: u(x.astype(dt)).astype(
                np.float64 if dt == np.float64 else np.float32))(u)
    return out


# ------------------------------------------------------------------ scipy
def scipy_impls(device, dtype):
    if device != "cpu" or dtype in ("bfloat16", "binary16"):
        return {}   # scipy.special has no float16/bfloat16 loops
    import scipy.special as sc
    dt = np.float32 if dtype == "float32" else np.float64
    out = {}
    for f, n in SC_NAMES.items():
        u = getattr(sc, n)
        sig = "f->f" if dt == np.float32 else "d->d"
        if sig not in u.types:
            continue
        out[f] = (lambda u: lambda x: u(x.astype(dt)).astype(dt))(u)
    return out


# ------------------------------------------------------------------ torch
_torch_ready = False


def torch_setup():
    global _torch_ready
    import torch
    if not _torch_ready:
        torch.set_num_threads(int(os.environ.get("ACC_THREADS", os.cpu_count())))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _torch_ready = True
    return torch


def torch_impls(device, dtype):
    torch = torch_setup()
    if device == "cuda" and not torch.cuda.is_available():
        return {}
    if device == "mps" and not torch.backends.mps.is_available():
        return {}
    if device == "mps" and dtype == "float64":
        return {}   # MPS has no float64
    tdt = {"float32": torch.float32, "float64": torch.float64, "binary16": torch.float16,
           "bfloat16": torch.bfloat16}[dtype]

    def to_t(x):
        if dtype == "bfloat16":
            t = torch.from_numpy(_f32_to_bf16_bits(x).view(np.int16)).view(torch.bfloat16)
        elif dtype == "binary16":
            t = torch.from_numpy(x.astype(np.float16))
        elif dtype == "float32":
            t = torch.from_numpy(np.ascontiguousarray(x, np.float32))
        else:
            t = torch.from_numpy(np.ascontiguousarray(x, np.float64))
        return t.to(device)

    def from_t(t):
        assert t.dtype == tdt, (t.dtype, tdt)
        t = t.cpu()
        if dtype == "bfloat16":
            return _bf16_bits_to_f32(t.view(torch.int16).numpy().view(np.uint16))
        if dtype == "binary16":
            return t.numpy().astype(np.float32)
        return t.numpy()

    out = {}
    for f, n in TORCH_NAMES.items():
        obj = torch
        for part in n.split("."):
            obj = getattr(obj, part)
        u = obj
        # probe support for this dtype/device
        try:
            a = to_t(np.array([0.5, 1.5], np.float32 if dtype != "float64" else np.float64))
            r = u(a, a) if f in BI else u(a)
            if r.dtype != tdt:
                continue
        except Exception:
            continue
        if f in BI:
            out[f] = (lambda u: lambda x, y: from_t(u(to_t(x), to_t(y))))(u)
        else:
            out[f] = (lambda u: lambda x: from_t(u(to_t(x))))(u)
    if device == "cuda":
        torch.cuda.synchronize()
    return out


# ------------------------------------------------------------------ jax
def jax_impls(device, dtype):
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jss
    import ml_dtypes
    plat = {"cpu": "cpu", "cuda": "gpu"}.get(device)
    if plat is None:
        return {}
    try:
        dev = jax.devices(plat)[0]
    except Exception:
        return {}
    if dtype == "float64" and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)
    ndt = {"float32": np.float32, "float64": np.float64, "binary16": np.float16,
           "bfloat16": ml_dtypes.bfloat16}[dtype]
    fns = {f: getattr(jnp, n) for f, n in NP_NAMES.items()}
    fns.update({f: getattr(jss, n) for f, n in JAX_SPECIAL.items()})
    import jax.lax
    fns["rsqrt"] = jax.lax.rsqrt
    fns["j0"] = lambda x: jss.bessel_jn(x, v=0)[0]
    fns["j1"] = lambda x: jss.bessel_jn(x, v=1)[1]
    out = {}
    for f, u in fns.items():
        j = jax.jit(u)

        def call(*args, j=j):
            a = [jax.device_put(np.asarray(v).astype(ndt), dev) for v in args]
            r = np.asarray(j(*a))
            assert r.dtype == ndt, (r.dtype, ndt)
            return r.astype(np.float64 if dtype == "float64" else np.float32)
        try:
            t = np.array([0.5, 1.5])
            call(t, t) if f in BI else call(t)
        except Exception:
            continue
        out[f] = call
    return out


IMPLS = {"numpy": numpy_impls, "scipy": scipy_impls, "torch": torch_impls, "jax": jax_impls}


def impls(lib, device, dtype):
    return IMPLS[lib](device, dtype)


def versions():
    v = {}
    import platform
    v["python"] = platform.python_version()
    v["platform"] = platform.platform()
    v["machine"] = platform.machine()
    for m in ["numpy", "scipy", "jax", "jaxlib", "torch", "ml_dtypes"]:
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["torch_config"] = torch.__config__.show()
        v["torch_parallel"] = torch.__config__.parallel_info()
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import numpy as np
        import io, contextlib
        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            np.show_config()
        v["numpy_config"] = s.getvalue()
        v["numpy_simd"] = {k: list(map(str, val)) if isinstance(val, (list, tuple)) else str(val)
                           for k, val in getattr(np._core._multiarray_umath, "__cpu_features__", {}).items()
                           if val} if hasattr(np, "_core") else None
    except Exception:
        pass
    try:
        import jax
        v["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    v["env"] = {k: os.environ[k] for k in os.environ if k.startswith(("XLA", "OMP", "MKL", "JAX", "TF_", "NVIDIA_TF32", "TORCH", "ACC_", "OPENBLAS", "NPY", "CUDA"))}
    try:
        cpu = [l for l in open("/proc/cpuinfo") if l.startswith(("model name", "flags"))][:2]
        v["cpu"] = cpu
    except Exception:
        import subprocess
        try:
            v["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
    return v


def denormal_mode(lib, device, dtype):
    """Probe whether the backend flushes subnormals: DAZ (subnormal inputs
    read as zero) and FTZ (subnormal results written as zero), using the
    library's own elementwise multiply on non-constant arrays."""
    import ml_dtypes
    tiny = {"float32": np.float32(2.0 ** -126), "float64": np.float64(2.0 ** -1022),
            "binary16": np.float32(2.0 ** -14), "bfloat16": np.float32(2.0 ** -126)}[dtype]
    a = np.array([tiny * 0.75, tiny], dtype=np.float64 if dtype == "float64" else np.float32)
    b = np.array([1.0, 0.5], dtype=a.dtype)
    try:
        if lib in ("numpy", "scipy"):
            if device != "cpu":
                return None
            dt = {"float32": np.float32, "float64": np.float64, "binary16": np.float16}.get(dtype)
            if dt is None:
                return None
            r = np.multiply(a.astype(dt), b.astype(dt)).astype(np.float64)
        elif lib == "torch":
            torch = torch_setup()
            tdt = {"float32": torch.float32, "float64": torch.float64, "binary16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
            if device == "mps" and dtype == "float64":
                return None
            ta = torch.tensor(a.astype(np.float64)).to(tdt).to(device)
            tb = torch.tensor(b.astype(np.float64)).to(tdt).to(device)
            r = torch.mul(ta, tb).cpu().to(torch.float64).numpy()
        else:
            import jax, jax.numpy as jnp
            plat = {"cpu": "cpu", "cuda": "gpu"}[device]
            dev = jax.devices(plat)[0]
            if dtype == "float64" and not jax.config.read("jax_enable_x64"):
                jax.config.update("jax_enable_x64", True)
            ndt = {"float32": np.float32, "float64": np.float64, "binary16": np.float16,
                   "bfloat16": ml_dtypes.bfloat16}[dtype]
            f = jax.jit(jnp.multiply)
            r = np.asarray(f(jax.device_put(a.astype(ndt), dev), jax.device_put(b.astype(ndt), dev))).astype(np.float64)
    except Exception:
        return None
    return {"daz": bool(r[0] == 0), "ftz": bool(r[1] == 0)}
