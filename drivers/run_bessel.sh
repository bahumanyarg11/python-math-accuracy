#!/bin/bash
TAG=$1; DEV=$2; LIBS=${3:-numpy,scipy,jax,torch}
cd "$(dirname "$0")/.."
python3 drivers/exhaustive.py --fmt binary32 --device $DEV --tag $TAG --libs $LIBS --funcs j0,j1,y0,y1
python3 -W ignore drivers/verify.py --tag $TAG --device $DEV
echo ALLDONE $TAG
