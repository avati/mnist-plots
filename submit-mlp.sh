#!/bin/sh

set -x

export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64

mkdir ./mnist-dim-$DQ_HDIM-datafrac-$DQ_DATA_FRAC-nlayers-$DQ_NLAYERS.dir
THEANO_FLAGS="device=gpu0,floatX=float32" python mnist-mlp.py --epochs=${DQ_EPOCHS:-10000000} --hdim=${DQ_HDIM:-500} --nlayers=${DQ_NLAYERS:-2} --data_frac=${DQ_DATA_FRAC:-1.0} --expdir=./mnist-dim-$DQ_HDIM-datafrac-$DQ_DATA_FRAC-nlayers-$DQ_NLAYERS.dir --optimizer=adam --lr=0.001 --batch_size 10 | tee log-hdim-$DQ_HDIM-nlayers-$DQ_NLAYERS-data_frac-$DQ_DATA_FRAC.log

