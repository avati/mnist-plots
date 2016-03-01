#!/bin/sh

set -x

export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64

mkdir ./mnist-softmax-$DQ_DATA_FRAC.dir -p
THEANO_FLAGS="device=gpu0,floatX=float32" python mnist-softmax.py --epochs=${DQ_EPOCHS:-10000000} --data_frac=${DQ_DATA_FRAC:-1.0} --expdir=./mnist-softmax-$DQ_DATA_FRAC.dir --optimizer=adam --lr=0.001 --batch_size 10 | tee log-softmax-data_frac-$DQ_DATA_FRAC.log

