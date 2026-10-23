"""Diagnose machine dependence of PyTorch x86 outputs: CPU, torch capability,
and outputs at the 11 inputs whose stored outputs were not reproduced."""
import json, subprocess, numpy as np, torch
cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo; grep -m1 -o -w -E 'avx512f|avx512dq' /proc/cpuinfo | sort -u | tr '\\n' ' '; nproc", shell=True, capture_output=True, text=True).stdout
print("CPU:", cpu.replace("\n", " | "))
print("torch capability:", torch.backends.cpu.get_cpu_capability(), "threads:", torch.get_num_threads())
cases = [("acos", "f4", "0x1.f2493e0000000p-2"), ("asin", "f4", "0x1.0ef0060000000p-1"), ("exp", "f4", "-0x1.73cfba0000000p+0"),
         ("log", "f4", "0x1.559d2e0000000p+1"), ("log2", "f4", "0x1.00b1100000000p+0"), ("tan", "f4", "0x1.a4e17c0000000p+22"),
         ("acos", "f8", "-0x1.0e41589c345eep-1"), ("asin", "f8", "-0x1.0e40420dc8057p-1"), ("atan", "f8", "-0x1.606dca52331c7p-1"),
         ("erfc", "f8", "0x1.a8cf81fa2a28fp+4"), ("tan", "f8", "-0x1.7b2f3a40a3982p+678")]
for f, dt, xs in cases:
    x = float.fromhex(xs); out = {}
    for n in (1, 7, 8, 16, 64, 1 << 20):
        for pos in sorted({0, n // 2, n - 1}):
            a = np.full(n, 0.5, dtype=dt); a[pos] = x
            fn = getattr(torch.special, f) if f == "erfc" else getattr(torch, f)
            out[f"n={n},pos={pos}"] = float(fn(torch.from_numpy(a))[pos]).hex()
    print(f, dt, xs, "distinct outputs:", sorted(set(out.values())), json.dumps(out) if len(set(out.values())) > 1 else "")
