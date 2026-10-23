#!/bin/bash
# Build CORE-MATH binary64 objects (correctly rounded references) into build/cm/
set -e
cd "$(dirname "$0")/.."
CC=${CC:-cc}
ARCHF=""; [ "$(uname -m)" = x86_64 ] && ARCHF="-march=native"
mkdir -p build/cm
# the 28 CORE-MATH binary64 functions used as references by harness/acc.c
for f in exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan sinh cosh tanh \
         asinh acosh atanh cbrt erf erfc lgamma tgamma pow atan2 hypot; do
  d=ext/core-math/src/binary64/$f; [ -d $d ] || d=ext/core-math/binary64/$f   # upstream layout, or a flat copy
  $CC -O2 -fPIC -ffp-contract=off $ARCHF -I$d -c $d/$f.c -o build/cm/$f.o
done
