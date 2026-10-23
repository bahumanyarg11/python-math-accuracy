#!/bin/bash
# Run every measurement for one platform.  usage: run_all.sh <tag> <device> [libs]
# Resumable (each driver skips finished (lib,func) results).
TAG=$1; DEV=$2; LIBS=${3:-numpy,scipy,jax,torch}
cd "$(dirname "$0")/.."
P=python3
set -x
$P drivers/exhaustive.py --fmt binary16 --device $DEV --tag $TAG --libs $LIBS
$P drivers/exhaustive.py --fmt bfloat16 --device $DEV --tag $TAG --libs $LIBS
$P drivers/exhaustive.py --fmt binary32 --device $DEV --tag $TAG --libs $LIBS --funcs exp,exp2,exp10,expm1,log,log2,log10,log1p,sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,asinh,acosh,atanh,cbrt,erf,erfc,lgamma,tgamma
$P drivers/search.py --fmt binary64 --device $DEV --tag $TAG --libs $LIBS
$P drivers/search.py --fmt binary32 --device $DEV --tag $TAG --libs $LIBS
$P drivers/exhaustive.py --fmt binary16 --bivariate --device $DEV --tag $TAG --libs $LIBS --funcs pow,atan2,hypot
$P drivers/exhaustive.py --fmt bfloat16 --bivariate --device $DEV --tag $TAG --libs $LIBS --funcs pow,atan2,hypot
[ -z "$NOBESSEL" ] && $P drivers/exhaustive.py --fmt binary32 --device $DEV --tag $TAG --libs $LIBS --funcs j0,j1,y0,y1
$P -W ignore drivers/verify.py --tag $TAG --device $DEV
echo ALLDONE $TAG
