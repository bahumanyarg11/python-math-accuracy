#!/bin/bash
python3 --version; nvidia-smi
pip install -q numpy==2.2.6 scipy==1.17.1 "jax[cuda12]==0.11.2" ml_dtypes==0.5.4 gmpy2 2>&1 | tail -2
pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
python3 -c "import numpy,scipy,jax,torch;print(numpy.__version__,scipy.__version__,jax.__version__,torch.__version__,torch.cuda.is_available(),jax.devices())"
./harness/build.sh
