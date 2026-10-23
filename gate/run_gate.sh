#!/bin/bash
# Stage-1 gate on x86_64 Linux. Needs gcc, libmpfr-dev, python3+numpy.
set -x
cd "$(dirname "$0")/.."
uname -a; ldd --version | head -1; gcc --version | head -1; grep -m1 "model name" /proc/cpuinfo; nproc
./harness/build.sh
gcc -O2 -fPIC -shared gate/gatelib.c -o gate/libgate_glibc.so -lm
[ -d gate/openlibm-0.8.7 ] || (cd gate && wget -q https://github.com/JuliaMath/openlibm/archive/refs/tags/v0.8.7.tar.gz -O ol.tgz && tar xzf ol.tgz)
(cd gate/openlibm-0.8.7 && make -j"$(nproc)" libopenlibm.a >/dev/null 2>&1)
gcc -O2 -fPIC -shared -DNO_EXP10 -Igate/openlibm-0.8.7/include gate/gatelib.c -o gate/libgate_openlibm.so \
    -Wl,-Bsymbolic -Wl,--whole-archive gate/openlibm-0.8.7/libopenlibm.a -Wl,--no-whole-archive
nm -D gate/libgate_openlibm.so | grep -c " T " 
python3 gate/gate.py glibc b64
python3 gate/gate.py openlibm b64
python3 gate/gate.py glibc b32
python3 gate/gate.py openlibm b32
# their own exhaustive checker on glibc expf, as an independent cross-check of the harness
M=gate/math_accuracy
[ -d $M ] || git clone -q https://gitlab.inria.fr/zimmerma/math_accuracy.git $M
(cd $M && git checkout -q c2b897fb393ed8df33452b225953ccd2003cd389)
for f in exp cosh; do
gcc -DSTR=$f -DGLIBC -O3 $M/binary32_exhaustive/check_exhaustive.c -lmpfr -lgmp -lm -fopenmp -o gate/ce_$f
./gate/ce_$f
done
