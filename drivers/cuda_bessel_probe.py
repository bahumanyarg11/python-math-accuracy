"""Record exactly what happens when torch.special Bessel functions are called
on CUDA in each dtype (the availability probe found float32 missing)."""
import json, traceback, torch
out = {}
for name in ["bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1"]:
    for dt in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        k = f"{name}/{str(dt)[6:]}"
        try:
            x = torch.tensor([0.5, 1.5, 100.0], dtype=dt, device="cuda")
            y = getattr(torch.special, name)(x)
            torch.cuda.synchronize()
            out[k] = {"ok": True, "dtype_out": str(y.dtype), "values": [float(v) for v in y.cpu().double()]}
        except Exception as e:
            out[k] = {"ok": False, "error": "".join(traceback.format_exception_only(type(e), e)).strip()[:2000]}
        print(k, out[k], flush=True)
json.dump(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), results=out),
          open("results/provenance/cuda_bessel_probe.json", "w"), indent=1)
