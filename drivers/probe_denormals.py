"""Record whether each backend flushes subnormals (FTZ) / treats subnormal
inputs as zero (DAZ), per library and dtype.  usage: probe_denormals.py --tag T --device D"""
import sys, os, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import libs
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cpu")
ap.add_argument("--libs", default="numpy,scipy,jax,torch")
a = ap.parse_args()
out = {lib: {dt: libs.denormal_mode(lib, a.device, dt) for dt in ["float32", "float64", "binary16", "bfloat16"]}
       for lib in a.libs.split(",")}
os.makedirs(os.path.join(HERE, "..", "results", "provenance"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "provenance", f"denormals_{a.tag}.json"), "w"), indent=1)
print(a.tag, out)
