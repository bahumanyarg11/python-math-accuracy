#!/bin/bash
# Fetch the third-party sources used by this project (they are not redistributed here):
#  - CORE-MATH (correctly rounded binary64 references), commit 7ebab739
#  - math_accuracy (Gladman, Innocente, Mather, Ozaki, Zimmermann), commit c2b897fb,
#    whose error functions are ported in harness/acc.c and whose search is ported in drivers/search.py
set -e
cd "$(dirname "$0")/.."
mkdir -p ext
[ -d ext/core-math/.git ] || git clone https://gitlab.inria.fr/core-math/core-math.git ext/core-math
git -C ext/core-math checkout --quiet 7ebab73913c5a33a25402ee51c96ec06b3a18095
[ -d ext/math_accuracy/.git ] || git clone https://gitlab.inria.fr/zimmerma/math_accuracy.git ext/math_accuracy
git -C ext/math_accuracy checkout --quiet c2b897fb393ed8df33452b225953ccd2003cd389
echo "dependencies ready in ext/"
