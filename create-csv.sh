#!/bin/sh

for hdim_nlayers in 16_1 16_2 16_3 32_1 32_2 32_3 64_1 64_2 64_3 128_1 128_2 128_3; do
    hdim=`echo $hdim_nlayers | cut -f1 -d_`;
    nlayers=`echo $hdim_nlayers | cut -f2 -d_`;
    > hdim-$hdim-nlayers-$nlayers.csv
    for data_frac in `seq 0.001 0.001 0.009; seq 0.01 0.01 0.09; seq 0.1 0.1 0.9`; do
	perf=`grep Best log-hdim-$hdim-nlayers-$nlayers-data_frac-$data_frac.log | cut -f3 -d' '`
	echo "$data_frac, $perf" >> hdim-$hdim-nlayers-$nlayers.csv
    done
done

> softmax.csv
for data_frac in `seq 0.001 0.001 0.009; seq 0.01 0.01 0.09; seq 0.1 0.1 0.9`; do
    perf=`grep Best log-softmax-data_frac-$data_frac.log | cut -f3 -d' '`
    echo "$data_frac, $perf" >> softmax.csv
done