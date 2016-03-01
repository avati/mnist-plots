#!/bin/sh

for HDIM_NLAYER in 16_1 16_2 16_3 32_1 32_2 32_3 64_1 64_2 64_3 128_1 128_2 128_3; do
    DQ_HDIM=`echo $HDIM_NLAYER | cut -f1 -d_`;
    DQ_NLAYERS=`echo $HDIM_NLAYER | cut -f2 -d_`;
    for DQ_DATA_FRAC in `seq 0.001 0.001 0.009; seq 0.01 0.01 0.09; seq 0.1 0.1 1.0`; do
	DQ_HDIM=$DQ_HDIM DQ_NLAYERS=$DQ_NLAYERS DQ_DATA_FRAC=$DQ_DATA_FRAC sh ./submit-mlp.sh;
    done;
done;

for DQ_DATA_FRAC in `seq 0.001 0.001 0.009; seq 0.01 0.01 0.09; seq 0.1 0.1 0.9`; do
    DQ_DATA_FRAC=$DQ_DATA_FRAC sh ./submit-softmax.sh;
done;
