#!/bin/bash
# build libacc (error engine + CORE-MATH references)
set -e
cd "$(dirname "$0")/.."
CC=${CC:-cc}
INC=""; LIBS="-lmpfr -lgmp -lm -lpthread"
if [ "$(uname)" = Darwin ]; then INC="-I/opt/homebrew/include -L/opt/homebrew/lib"; EXT=dylib; else EXT=so; fi
[ -f build/cm/exp.o ] || CC=$CC ./harness/build_cm.sh
$CC -O2 -fPIC -shared -ffp-contract=off -Wall -Wno-unused-function $INC harness/acc.c build/cm/*.o -o build/libacc.$EXT.tmp $LIBS && mv build/libacc.$EXT.tmp build/libacc.$EXT
echo built build/libacc.$EXT
